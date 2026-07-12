use core::f32;

use candle_core::{DType, Device, Error as CandleError, IndexOp, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_b, linear_no_bias,
    ops::{dropout, softmax},
    sequential, Embedding, Init, LayerNorm, Linear, Module, Sequential, VarBuilder, VarMap,
};

use crate::error::SmolResult;

const NUM_HEADS: usize = 8;
const NUM_BLOCKS: usize = 6;
const DROPOUT: f32 = 0.1;

pub struct Gpt {
    pub token_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub transformer_blocks: Sequential, // Holds TransformerBlock modules
    pub lm_head: Linear,
    pub var_map: VarMap,
    pub block_size: usize,
}

impl Gpt {
    pub fn new(
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        device: &Device,
    ) -> SmolResult<Self> {
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
        let transformer_blocks = build_blocks(embed_dims, block_size, &var_builder, device)?;
        // Convert embedding dimension back to vocabulary size
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder.pp("lm_head"))?;

        Ok(Gpt {
            token_embeddings,
            position_embeddings,
            transformer_blocks,
            lm_head,
            var_map,
            block_size,
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
        device: &Device,
    ) -> SmolResult<Self> {
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
        let transformer_blocks = build_blocks(embed_dims, block_size, &var_builder, device)?;
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder.pp("lm_head"))?;
        var_map.load(path)?;

        Ok(Gpt {
            token_embeddings: Embedding::new(token_embeddings, embed_dims),
            position_embeddings: Embedding::new(position_embeddings, embed_dims),
            transformer_blocks,
            lm_head,
            var_map,
            block_size,
        })
    }
}

impl Module for Gpt {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len)
    ) -> Result<Tensor, CandleError> {
        // (b, t, c) => b = batch_size, t = seq_len, c = embed_dims
        let (_, t) = input.shape().dims2()?;

        let token_embedding = self.token_embeddings.forward(input)?; // (batch_size, seq_len, embed_dims)

        let position_embedding =
            self.position_embeddings
                .forward(&Tensor::arange(0, t as u32, input.device())?)?; // (seq_len, embed_dims)

        let combined_embedding = token_embedding.broadcast_add(&position_embedding)?; // (batch_size, seq_len, embed_dims)

        let x_blocks = self.transformer_blocks.forward(&combined_embedding)?; // (batch_size, seq_len, embed_dims)

        let logits = self.lm_head.forward(&x_blocks)?; // (batch_size, seq_len, vocab_size)

        Ok(logits)
    }
}

/// Build the stack of transformer blocks under a shared `blocks` prefix so that
/// `new` and `load` produce identical variable names.
fn build_blocks(
    embed_dims: usize,
    block_size: usize,
    vb: &VarBuilder,
    device: &Device,
) -> Result<Sequential, CandleError> {
    let mut blocks = sequential::seq();
    let blocks_vb = vb.pp("blocks");
    for block_idx in 0..NUM_BLOCKS {
        let block = TransformerBlock::new(
            embed_dims,
            block_size,
            blocks_vb.pp(format!("block_{block_idx}")),
            device,
        )?;
        blocks = blocks.add(block);
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
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let head_size = embed_dims / NUM_HEADS;
        let multi_head_attn =
            MultiHeadAttention::new(embed_dims, head_size, block_size, vb.pp("attn"), device)?;
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
}

impl Module for TransformerBlock {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
    ) -> Result<Tensor, CandleError> {
        // Pre-norm residual connections (nanoGPT style).
        let ln1 = self.layer_norm1.forward(input)?; // (batch_size, seq_len, embed_dims)
        let self_attn = self.multi_head_attn.forward(&ln1)?; // (batch_size, seq_len, embed_dims)
        let attn_output = input.broadcast_add(&self_attn)?; // (batch_size, seq_len, embed_dims)
        let ln2 = self.layer_norm2.forward(&attn_output)?; // (batch_size, seq_len, embed_dims)
        let ff_output = self.feed_forward.forward(&ln2)?; // (batch_size, seq_len, embed_dims)
        let output = attn_output.broadcast_add(&ff_output)?; // (batch_size, seq_len, embed_dims)
        Ok(output)
    }
}

struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout_rate: f32,
}

impl MultiHeadAttention {
    pub fn new(
        embed_dims: usize,
        head_size: usize,
        block_size: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let mut heads = Vec::with_capacity(NUM_HEADS);
        for head_idx in 0..NUM_HEADS {
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
        let proj = linear(NUM_HEADS * head_size, embed_dims, vb.pp("proj"))?;

        Ok(MultiHeadAttention {
            heads,
            proj,
            dropout_rate: DROPOUT,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(
        &self,
        input: &Tensor, // (batch_size, seq_len, embed_dims)
    ) -> Result<Tensor, CandleError> {
        let head_outputs = self
            .heads
            .iter()
            .map(|head| head.forward(input))
            .collect::<Result<Vec<_>, _>>()?;
        // Concatenate along the channel dim: (batch_size, seq_len, num_heads * head_size)
        let concatenated = Tensor::cat(&head_outputs, 2)?;
        let projected = self.proj.forward(&concatenated)?; // (batch_size, seq_len, embed_dims)
        dropout(&projected, self.dropout_rate)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout_rate: f32,
}

impl FeedForward {
    pub fn new(embed_dims: usize, vb: VarBuilder) -> Result<Self, CandleError> {
        // Standard 4x inner expansion.
        let fc1 = linear(embed_dims, 4 * embed_dims, vb.pp("fc1"))?;
        let fc2 = linear(4 * embed_dims, embed_dims, vb.pp("fc2"))?;
        Ok(FeedForward {
            fc1,
            fc2,
            dropout_rate: DROPOUT,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor, CandleError> {
        let hidden = self.fc1.forward(input)?.relu()?;
        let out = self.fc2.forward(&hidden)?;
        dropout(&out, self.dropout_rate)
    }
}

struct Head {
    /// one head of self-attention:
    key: Linear,
    query: Linear,
    value: Linear,
    tril: Tensor,
    neg_inf: Tensor,
    dropout_rate: f32,
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
            dropout_rate,
        })
    }
}

impl Module for Head {
    fn forward(&self, input: &Tensor) -> Result<Tensor, CandleError> {
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
        weights = dropout(&weights, self.dropout_rate)?;

        // Weighted aggregation of the values.
        let v = self.value.forward(input)?; // (B, T, C)
        let out = weights.matmul(&v)?; // (B, T, T) @ (B, T, C) -> (B, T, C)

        Ok(out)
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
        let gpt = Gpt::new(block_size, vocab_size, embed_dims, &device).unwrap();
        // (batch_size=2, seq_len=8)
        let input = Tensor::zeros((2, block_size), DType::U32, &device).unwrap();
        let logits = gpt.forward(&input).unwrap();
        assert_eq!(logits.shape(), &candle_core::Shape::from(&[2, block_size, vocab_size]));
    }
}
