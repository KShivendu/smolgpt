use core::f32;

use candle_core::{DType, Device, Error as CandleError, IndexOp, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_b, linear_no_bias,
    ops::{Dropout, softmax},
    Embedding, Init, LayerNorm, Linear, Module, VarBuilder, VarMap,
};

use crate::error::{SmolError, SmolResult};

const DROPOUT: f32 = 0.1;

pub struct Gpt {
    pub token_embeddings: Embedding,
    pub position_embeddings: Embedding,
    transformer_blocks: Vec<TransformerBlock>, // Holds TransformerBlock modules
    pub lm_head: Linear,
    pub var_map: VarMap,
    pub block_size: usize,
    /// Stored so `LanguageModel::snapshot` can re-load a frozen reference
    /// copy with the same arch without the caller having to thread the
    /// constructor params back through. Not serialized into the `.bin`
    /// (only `var_map` is); reconstructed from the constructor args on load.
    pub vocab_size: usize,
    pub embed_dims: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
}

impl Gpt {
    pub fn new(
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        if embed_dims % num_heads != 0 {
            return Err(SmolError::invalid_argument(&format!(
                "hidden_size ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Create the token embedding layer
        let token_embeddings = Embedding::new(
            var_builder.get_with_hints(
                (vocab_size, embed_dims),
                "token_embeddings",
                Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )?,
            embed_dims,
        );
        let position_embeddings = Embedding::new(
            var_builder.get_with_hints(
                (block_size, embed_dims),
                "position_embeddings",
                Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )?,
            embed_dims,
        );
        let transformer_blocks = build_blocks(
            embed_dims,
            block_size,
            num_heads,
            num_blocks,
            &var_builder,
            device,
        )?;
        // Convert embedding dimension back to vocabulary size
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder.pp("lm_head"))?;

        Ok(Gpt {
            token_embeddings,
            position_embeddings,
            transformer_blocks,
            lm_head,
            var_map,
            block_size,
            vocab_size,
            embed_dims,
            num_heads,
            num_blocks,
        })
    }

    pub fn save(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load(
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        if embed_dims % num_heads != 0 {
            return Err(SmolError::invalid_argument(&format!(
                "hidden_size ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        // Build the exact same structure as `new` so the variable names/shapes
        // line up with the saved checkpoint; `var_map.load` then overwrites the
        // freshly-initialized values below with the trained ones.
        let token_embeddings = var_builder.get_with_hints(
            (vocab_size, embed_dims),
            "token_embeddings",
            Init::Const(0.0),
        )?;
        let position_embeddings = var_builder.get_with_hints(
            (block_size, embed_dims),
            "position_embeddings",
            Init::Const(0.0),
        )?;
        let transformer_blocks = build_blocks(
            embed_dims,
            block_size,
            num_heads,
            num_blocks,
            &var_builder,
            device,
        )?;
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder.pp("lm_head"))?;
        if let Err(underlying) = var_map.load(path) {
            return Err(SmolError::invalid_argument(&format!(
                "Model file {} does not match the requested architecture \
                 (block_size={block_size}, hidden_size={embed_dims}, \
                 num_heads={num_heads}, num_blocks={num_blocks}). \
                 Re-run with --block-size/--hidden-size/--num-heads/--num-blocks \
                 matching the saved model. Underlying error: {underlying}",
                path.display()
            )));
        }

        Ok(Gpt {
            token_embeddings: Embedding::new(token_embeddings, embed_dims),
            position_embeddings: Embedding::new(position_embeddings, embed_dims),
            transformer_blocks,
            lm_head,
            var_map,
            block_size,
            vocab_size,
            embed_dims,
            num_heads,
            num_blocks,
        })
    }

    /// Forward pass with an explicit `is_training` flag. When `is_training`
    /// is `false`, all `Dropout` layers are no-ops, which makes inference
    /// deterministic given the same inputs and weights. The `Module::forward`
    /// impl below delegates to this with `is_training = false` so that the
    /// `&dyn Module` callers (e.g. `LanguageModel::generate`) get the
    /// inference/deterministic behaviour, while `LanguageModel::train` calls
    /// this directly with `is_training = true` to keep dropout active during
    /// training.
    pub fn forward_with_training(
        &self,
        input: &Tensor, // (batch_size, seq_len)
        is_training: bool,
    ) -> Result<Tensor, CandleError> {
        // (b, t, c) => b = batch_size, t = seq_len, c = embed_dims
        let (_, t) = input.shape().dims2()?;
        if t > self.block_size {
            candle_core::bail!(
                "Gpt::forward_with_training: input seq_len {t} exceeds block_size {} \
                 (the position-embedding table only has {} rows); truncate the input first",
                self.block_size,
                self.block_size
            );
        }

        let token_embedding = self.token_embeddings.forward(input)?; // (batch_size, seq_len, embed_dims)

        let position_embedding =
            self.position_embeddings
                .forward(&Tensor::arange(0, t as u32, input.device())?)?; // (seq_len, embed_dims)

        let combined_embedding = token_embedding.broadcast_add(&position_embedding)?; // (batch_size, seq_len, embed_dims)

        let mut x_blocks = combined_embedding; // (batch_size, seq_len, embed_dims)
        for block in &self.transformer_blocks {
            x_blocks = block.forward_with_training(&x_blocks, is_training)?;
        }

        let logits = self.lm_head.forward(&x_blocks)?; // (batch_size, seq_len, vocab_size)

        Ok(logits)
    }
}

impl Module for Gpt {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len)
    ) -> Result<Tensor, CandleError> {
        // Default to inference mode so generation is deterministic.
        self.forward_with_training(input, false)
    }
}

/// Build the stack of transformer blocks under a shared `blocks` prefix so that
/// `new` and `load` produce identical variable names.
fn build_blocks(
    embed_dims: usize,
    block_size: usize,
    num_heads: usize,
    num_blocks: usize,
    vb: &VarBuilder,
    device: &Device,
) -> Result<Vec<TransformerBlock>, CandleError> {
    let blocks_vb = vb.pp("blocks");
    let mut blocks = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let block = TransformerBlock::new(
            embed_dims,
            block_size,
            num_heads,
            blocks_vb.pp(format!("block_{block_idx}")),
            device,
        )?;
        blocks.push(block);
    }
    Ok(blocks)
}

struct TransformerBlock {
    multi_head_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(
        embed_dims: usize,
        block_size: usize,
        num_heads: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let head_size = embed_dims / num_heads;
        let multi_head_attn =
            MultiHeadAttention::new(embed_dims, head_size, block_size, num_heads, vb.pp("attn"), device)?;
        let feed_forward = FeedForward::new(embed_dims, vb.pp("ffwd"))?;
        let layer_norm1 = layer_norm(embed_dims, 1e-5, vb.pp("ln1"))?;
        let layer_norm2 = layer_norm(embed_dims, 1e-5, vb.pp("ln2"))?;

        Ok(TransformerBlock {
            multi_head_attn,
            feed_forward,
            layer_norm1,
            layer_norm2,
        })
    }

    pub fn forward_with_training(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
        is_training: bool,
    ) -> Result<Tensor, CandleError> {
        // Pre-norm residual connections (nanoGPT style).
        let ln1 = self.layer_norm1.forward(input)?; // (batch_size, seq_len, embed_dims)
        let self_attn = self.multi_head_attn.forward_with_training(&ln1, is_training)?; // (batch_size, seq_len, embed_dims)
        let attn_output = input.broadcast_add(&self_attn)?; // (batch_size, seq_len, embed_dims)
        let ln2 = self.layer_norm2.forward(&attn_output)?; // (batch_size, seq_len, embed_dims)
        let ff_output = self.feed_forward.forward_with_training(&ln2, is_training)?; // (batch_size, seq_len, embed_dims)
        let output = attn_output.broadcast_add(&ff_output)?; // (batch_size, seq_len, embed_dims)
        Ok(output)
    }
}

impl Module for TransformerBlock {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
    ) -> Result<Tensor, CandleError> {
        self.forward_with_training(input, false)
    }
}

struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(
        embed_dims: usize,
        head_size: usize,
        block_size: usize,
        num_heads: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let mut heads = Vec::with_capacity(num_heads);
        for head_idx in 0..num_heads {
            heads.push(Head::new(
                embed_dims,
                head_size,
                block_size,
                DROPOUT,
                vb.pp(format!("head_{head_idx}")),
                device,
            )?);
        }
        // Project the concatenated heads back to the embedding dimension.
        let proj = linear(num_heads * head_size, embed_dims, vb.pp("proj"))?;

        Ok(MultiHeadAttention {
            heads,
            proj,
            dropout: Dropout::new(DROPOUT),
        })
    }

    pub fn forward_with_training(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
        is_training: bool,
    ) -> Result<Tensor, CandleError> {
        let head_outputs = self
            .heads
            .iter()
            .map(|head| head.forward_with_training(input, is_training))
            .collect::<Result<Vec<_>, _>>()?;
        // Concatenate along the channel dim: (batch_size, seq_len, num_heads * head_size)
        let concatenated = Tensor::cat(&head_outputs, 2)?;
        let projected = self.proj.forward(&concatenated)?; // (batch_size, seq_len, embed_dims)
        self.dropout.forward(&projected, is_training)
    }
}

impl Module for MultiHeadAttention {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
    ) -> Result<Tensor, CandleError> {
        self.forward_with_training(input, false)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(embed_dims: usize, vb: VarBuilder) -> Result<Self, CandleError> {
        // Standard 4x inner expansion.
        let fc1 = linear(embed_dims, 4 * embed_dims, vb.pp("fc1"))?;
        let fc2 = linear(4 * embed_dims, embed_dims, vb.pp("fc2"))?;
        Ok(FeedForward {
            fc1,
            fc2,
            dropout: Dropout::new(DROPOUT),
        })
    }

    pub fn forward_with_training(
        &self,
        input: &Tensor,
        is_training: bool,
    ) -> Result<Tensor, CandleError> {
        let hidden = self.fc1.forward(input)?.relu()?;
        let out = self.fc2.forward(&hidden)?;
        self.dropout.forward(&out, is_training)
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor, CandleError> {
        self.forward_with_training(input, false)
    }
}

struct Head {
    /// one head of self-attention:
    key: Linear,
    query: Linear,
    value: Linear,
    tril: Tensor,
    neg_inf: Tensor,
    dropout: Dropout,
}

impl Head {
    pub fn new(
        embed_size: usize,
        head_size: usize,
        block_size: usize,
        dropout_rate: f32,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let key = linear_no_bias(embed_size, head_size, vb.push_prefix("key"))?;
        let query = linear_no_bias(embed_size, head_size, vb.push_prefix("query"))?;
        let value = linear_no_bias(embed_size, head_size, vb.push_prefix("value"))?;

        let tril = Tensor::tril2(block_size, DType::U32, device)?;
        let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.to_device(device)?;

        Ok(Head {
            key,
            query,
            value,
            tril,
            neg_inf,
            dropout: Dropout::new(dropout_rate),
        })
    }

    pub fn forward_with_training(
        &self,
        input: &Tensor,
        is_training: bool,
    ) -> Result<Tensor, CandleError> {
        let k = self.key.forward(input)?; // (batch_size, seq_len, head_size)
        let q = self.query.forward(input)?; // (batch_size, seq_len, head_size)

        let (batch_size, time_size, channel_size) = k.shape().dims3()?;

        // Scaled dot-product attention scores: (B, T, C) @ (B, C, T) -> (B, T, T)
        let mut weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            * (channel_size as f64).powf(-0.5))?;

        // Causal mask: keep the lower triangle, set the future to -inf.
        let mask = self.tril.i((..time_size, ..time_size))?; // (T, T)
        weights = mask
            .broadcast_as((batch_size, time_size, time_size))?
            .where_cond(
                &weights,
                &self.neg_inf.broadcast_as((batch_size, time_size, time_size))?,
            )?; // (B, T, T)
        weights = softmax(&weights, D::Minus1)?; // (B, T, T)
        weights = self.dropout.forward(&weights, is_training)?;

        // Weighted aggregation of the values.
        let v = self.value.forward(input)?; // (B, T, C)
        let out = weights.matmul(&v)?; // (B, T, T) @ (B, T, C) -> (B, T, C)

        Ok(out)
    }
}

impl Module for Head {
    fn forward(&self, input: &Tensor) -> Result<Tensor, CandleError> {
        self.forward_with_training(input, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_head() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let head = super::Head::new(32, 16, 8, 0.1, vb, &device).unwrap();
        let input = Tensor::randn(0f32, 1f32, (2, 8, 32), &device).unwrap();
        let output = head.forward(&input).unwrap();
        assert_eq!(output.shape(), &candle_core::Shape::from(&[2, 8, 16]));
    }

    #[test]
    fn test_gpt_forward_shape() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (8, 40, 32);
        let gpt = Gpt::new(block_size, vocab_size, embed_dims, 8, 6, &device).unwrap();
        // (batch_size=2, seq_len=8)
        let input = Tensor::zeros((2, block_size), DType::U32, &device).unwrap();
        let logits = gpt.forward(&input).unwrap();
        assert_eq!(logits.shape(), &candle_core::Shape::from(&[2, block_size, vocab_size]));
    }
}
