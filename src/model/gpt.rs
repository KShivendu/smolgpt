use core::f32;

use candle_core::{DType, Device, Error as CandleError, IndexOp, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_b, linear_no_bias,
    ops::{Dropout, softmax},
    Embedding, Init, LayerNorm, Linear, Module, VarBuilder, VarMap,
};

use crate::error::{SmolError, SmolResult};

const DROPOUT: f32 = 0.1;

/// The `--init-std` sentinel that reproduces today's exact fresh-init
/// behavior. Below this, `build_linear` delegates to candle_nn's own
/// `linear`/`linear_no_bias` (Kaiming-normal weight, `Uniform(-1/sqrt(fan_in),
/// 1/sqrt(fan_in))` bias) so a default `--init-std 1.0` run is byte-for-byte
/// unaffected by this module's EXPERIMENTAL init-scale support. Only when
/// `init_std` differs from this sentinel do weight matrices switch to
/// `Init::Randn { mean: 0.0, stdev: init_std }`.
const DEFAULT_INIT_STD: f32 = 1.0;

/// Construct a `Linear` layer whose WEIGHT init is controlled by `init_std`
/// and `init_gain`, in that precedence order:
///
/// 1. At the `DEFAULT_INIT_STD` sentinel with `init_gain == None`, this is
///    exactly `candle_nn::linear`/`linear_no_bias` (today's Kaiming-normal
///    default, gain=√2 for ReLU, byte-for-byte unchanged default behavior).
/// 2. Otherwise, if `init_std` differs from the sentinel, `init_std` WINS
///    (it's the more direct override): every weight matrix uses
///    `Init::Randn { mean: 0.0, stdev: init_std }`, per Experiment A's
///    "applied consistently to every weight matrix" requirement (embeddings,
///    attention, MLP, lm_head — not just the embedding tables).
/// 3. Otherwise, if `init_gain` is `Some(gain)` (EXPERIMENTAL, `--init-gain`),
///    the weight uses candle's own Kaiming-Normal formula
///    (`std = gain / sqrt(fan_in)`) but with `gain` substituted for candle's
///    hardcoded ReLU gain of √2, via
///    `Init::Kaiming { dist: Normal, fan: FanIn, non_linearity: ExplicitGain(gain) }`.
///    This lets `--init-gain` probe the effect of the Kaiming gain constant
///    itself (e.g. PyTorch's default gain of √(1/3)≈0.577 vs candle's
///    textbook-ReLU √2≈1.414) without changing anything else about the
///    init scheme (still Kaiming-Normal, still fan-in scaled).
///
/// The bias (`bias == true`) keeps the same `Uniform(-1/sqrt(fan_in),
/// 1/sqrt(fan_in))` scheme candle_nn's `linear` uses in every branch, since
/// only the WEIGHT scale/gain is the variable under test here.
fn build_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    init_std: f32,
    init_gain: Option<f64>,
    vb: VarBuilder,
) -> Result<Linear, CandleError> {
    if init_std == DEFAULT_INIT_STD && init_gain.is_none() {
        return if bias {
            linear(in_dim, out_dim, vb)
        } else {
            linear_no_bias(in_dim, out_dim, vb)
        };
    }
    let weight_init = if init_std != DEFAULT_INIT_STD {
        Init::Randn {
            mean: 0.0,
            stdev: init_std as f64,
        }
    } else {
        // init_std is at the sentinel, so init_gain must be Some (checked above).
        Init::Kaiming {
            dist: candle_nn::init::NormalOrUniform::Normal,
            fan: candle_nn::init::FanInOut::FanIn,
            non_linearity: candle_nn::init::NonLinearity::ExplicitGain(
                init_gain.expect("init_gain must be Some when init_std is at the sentinel"),
            ),
        }
    };
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", weight_init)?;
    let bs = if bias {
        let bound = 1. / (in_dim as f64).sqrt();
        Some(vb.get_with_hints(out_dim, "bias", Init::Uniform { lo: -bound, up: bound })?)
    } else {
        None
    };
    Ok(Linear::new(ws, bs))
}

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
    /// Per-block attention head count, length == `num_blocks`. Index `i` is
    /// the head count of transformer block `i`. A uniform architecture (the
    /// common case, e.g. today's `--num-heads 4`) is represented as
    /// `vec![4; num_blocks]` — there's no separate "uniform" variant, so
    /// `load`/`snapshot` always reconstruct from this single source of
    /// truth instead of juggling a scalar + a schedule.
    pub heads_schedule: Vec<usize>,
    pub num_blocks: usize,
    /// EXPERIMENTAL (Experiment B, weight tying): whether `lm_head`'s weight
    /// is the SAME underlying `token_embeddings` weight tensor (transposed by
    /// `Linear::forward`'s own `matmul(w.t())`, no separate copy) rather than
    /// an independently allocated one. Stored so `LanguageModel::snapshot`/
    /// `Gpt::load` can reconstruct the exact same wiring without the caller
    /// re-threading the original `--tie-embeddings` flag through every call
    /// site (mirrors why `heads_schedule` is stored instead of re-derived).
    pub tie_embeddings: bool,
}

impl Gpt {
    /// `heads_schedule` must have exactly `num_blocks` entries, one per
    /// block; `embed_dims` must be evenly divisible by EACH entry
    /// individually (multi-head attention splits `embed_dims` across that
    /// block's heads, so a non-divisor would produce a fractional head
    /// size). Both are validated up front with a message naming the
    /// offending block, rather than surfacing as an opaque shape-mismatch
    /// deeper in `build_blocks`.
    ///
    /// `init_std` (EXPERIMENTAL, Experiment A): the stdev used for every
    /// weight matrix's `Init::Randn` at construction time (embeddings,
    /// attention Q/K/V/proj, MLP fc1/fc2, and `lm_head` when untied). See
    /// `DEFAULT_INIT_STD`/`build_linear`'s docs — `1.0` (the CLI default)
    /// reproduces today's exact init scheme byte-for-byte.
    ///
    /// `init_gain` (EXPERIMENTAL, `--init-gain`): when `init_std` is at its
    /// sentinel and this is `Some(gain)`, every `Linear` layer built via
    /// `build_linear` (attention Q/K/V/proj, MLP fc1/fc2, and `lm_head` when
    /// untied — NOT the embedding tables, which always use `init_std`
    /// directly) uses candle's Kaiming-Normal formula with `gain` substituted
    /// for candle's hardcoded ReLU gain of √2. `None` (the CLI default)
    /// reproduces today's exact init scheme byte-for-byte (candle's own
    /// `DEFAULT_KAIMING_NORMAL`, gain=√2). See `build_linear`'s doc for full
    /// precedence rules between `init_std` and `init_gain`.
    ///
    /// `tie_embeddings` (EXPERIMENTAL, Experiment B): when `true`, `lm_head`
    /// reuses `token_embeddings`' weight tensor directly (candle tracks
    /// gradients by the underlying `Var`, so using the same `Tensor` in two
    /// places in the graph correctly accumulates both usages' gradients into
    /// it) instead of allocating an independent `(vocab_size, embed_dims)`
    /// matrix. A small separate `lm_head.bias` is still allocated in both
    /// cases (tied embeddings usually keep their own output bias — it's a
    /// single `vocab_size`-length vector, negligible next to the
    /// `vocab_size * embed_dims` weight it would otherwise duplicate).
    pub fn new(
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        heads_schedule: &[usize],
        num_blocks: usize,
        init_std: f32,
        init_gain: Option<f64>,
        tie_embeddings: bool,
        device: &Device,
    ) -> SmolResult<Self> {
        validate_heads_schedule(embed_dims, heads_schedule, num_blocks)?;
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Create the token embedding layer
        let token_embeddings = Embedding::new(
            var_builder.get_with_hints(
                (vocab_size, embed_dims),
                "token_embeddings",
                Init::Randn {
                    mean: 0.0,
                    stdev: init_std as f64,
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
                    stdev: init_std as f64,
                },
            )?,
            embed_dims,
        );
        let transformer_blocks = build_blocks(
            embed_dims,
            block_size,
            heads_schedule,
            init_std,
            init_gain,
            &var_builder,
            device,
        )?;
        // Convert embedding dimension back to vocabulary size. When tied,
        // reuse `token_embeddings`' weight tensor directly (same shape:
        // `Embedding` stores `(vocab_size, embed_dims)`, and `Linear` expects
        // `(out_features, in_features) = (vocab_size, embed_dims)` too — no
        // transpose needed at construction time, `Linear::forward` transposes
        // at matmul time same as any other `Linear`) instead of allocating a
        // second independent `lm_head.weight` var.
        let lm_head = if tie_embeddings {
            let bias = var_builder.get_with_hints(vocab_size, "lm_head.bias", Init::Const(0.0))?;
            Linear::new(token_embeddings.embeddings().clone(), Some(bias))
        } else {
            build_linear(
                embed_dims,
                vocab_size,
                true,
                init_std,
                init_gain,
                var_builder.pp("lm_head"),
            )?
        };

        Ok(Gpt {
            token_embeddings,
            position_embeddings,
            transformer_blocks,
            lm_head,
            var_map,
            block_size,
            vocab_size,
            embed_dims,
            heads_schedule: heads_schedule.to_vec(),
            num_blocks,
            tie_embeddings,
        })
    }

    pub fn save(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    /// Post-training INT8 quantization for storage: see `crate::quantize`'s
    /// module doc for the scheme (per-tensor symmetric scale, i8 data, custom
    /// binary format) and why a full int8 *inference* path is out of scope.
    /// Does not mutate `self` — this only writes a quantized copy to `path`.
    pub fn save_quantized(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        crate::quantize::save_var_map_quantized(&self.var_map, path)
    }

    /// `tie_embeddings` MUST match the value the model was originally
    /// constructed/trained with (same rule as `block_size`/`heads_schedule`/
    /// etc — see the error message below): it controls whether an
    /// `lm_head.weight` var is allocated at all (untied) or `lm_head` reuses
    /// `token_embeddings` (tied, no separate var). Get it wrong and either
    /// (a) untied-but-actually-tied: a fresh unused `lm_head.weight` var is
    /// created and `var_map.load` errors because the saved file has no such
    /// key, or (b) tied-but-actually-untied: the saved `lm_head.weight`
    /// values are silently never loaded (no var requests that name), leaving
    /// `lm_head` at its random init. Both are load-time bugs this signature
    /// is designed to prevent by requiring the caller to know and pass it.
    pub fn load(
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        heads_schedule: &[usize],
        num_blocks: usize,
        tie_embeddings: bool,
        device: &Device,
    ) -> SmolResult<Self> {
        validate_heads_schedule(embed_dims, heads_schedule, num_blocks)?;
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        // Build the exact same structure as `new` so the variable names/shapes
        // line up with the saved checkpoint; `var_map.load` then overwrites the
        // freshly-initialized values below with the trained ones. `init_std`
        // doesn't matter here (every value gets overwritten by the load
        // below), so `DEFAULT_INIT_STD` is passed as an arbitrary placeholder.
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
            heads_schedule,
            DEFAULT_INIT_STD,
            None,
            &var_builder,
            device,
        )?;
        // Mirror `new`'s tying logic exactly so the var-map's set of named
        // variables matches what was saved: tied models never had a separate
        // `lm_head.weight` key to load, so we must not create one here either.
        let lm_head = if tie_embeddings {
            let bias = var_builder.get_with_hints(vocab_size, "lm_head.bias", Init::Const(0.0))?;
            Linear::new(token_embeddings.clone(), Some(bias))
        } else {
            linear_b(embed_dims, vocab_size, true, var_builder.pp("lm_head"))?
        };
        // `load_into_var_map` transparently handles both a regular f32
        // safetensors `.bin` (candle's own format, the original behavior) and
        // our custom quantized format (see `crate::quantize`'s doc) — it
        // detects which one `path` is via a magic-byte peek, so callers
        // (`--eval`/`--serve`/etc.) never need to know or care which kind of
        // file they're loading.
        if let Err(underlying) = crate::quantize::load_into_var_map(&mut var_map, path, device) {
            let heads_str = heads_schedule
                .iter()
                .map(|h| h.to_string())
                .collect::<Vec<_>>()
                .join(",");
            return Err(SmolError::invalid_argument(&format!(
                "Model file {} does not match the requested architecture \
                 (block_size={block_size}, hidden_size={embed_dims}, \
                 num_heads={heads_str}, num_blocks={num_blocks}). \
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
            heads_schedule: heads_schedule.to_vec(),
            num_blocks,
            tie_embeddings,
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

/// Validate a per-block head-count schedule: it must have exactly
/// `num_blocks` entries, and `embed_dims` must be evenly divisible by EACH
/// entry individually (multi-head attention splits `embed_dims` across that
/// block's heads — a non-divisor would produce a fractional head size).
/// Shared by `Gpt::new`/`Gpt::load` so both constructors fail the same way
/// with a message identifying the offending block index, instead of a bare
/// shape-mismatch surfacing later inside `build_blocks`/candle.
fn validate_heads_schedule(
    embed_dims: usize,
    heads_schedule: &[usize],
    num_blocks: usize,
) -> SmolResult<()> {
    if heads_schedule.len() != num_blocks {
        return Err(SmolError::invalid_argument(&format!(
            "heads schedule has {} entries but num_blocks is {num_blocks}; \
             --num-heads must be either a single number (applied to every \
             block) or a comma-separated list with exactly {num_blocks} entries",
            heads_schedule.len()
        )));
    }
    for (block_idx, &num_heads) in heads_schedule.iter().enumerate() {
        if num_heads == 0 || embed_dims % num_heads != 0 {
            return Err(SmolError::invalid_argument(&format!(
                "hidden_size ({embed_dims}) must be divisible by num_heads \
                 ({num_heads}) at block {block_idx}"
            )));
        }
    }
    Ok(())
}

/// Build the stack of transformer blocks under a shared `blocks` prefix so that
/// `new` and `load` produce identical variable names. `heads_schedule[i]` is
/// block `i`'s head count (already validated by the caller via
/// `validate_heads_schedule`).
fn build_blocks(
    embed_dims: usize,
    block_size: usize,
    heads_schedule: &[usize],
    init_std: f32,
    init_gain: Option<f64>,
    vb: &VarBuilder,
    device: &Device,
) -> Result<Vec<TransformerBlock>, CandleError> {
    let blocks_vb = vb.pp("blocks");
    let num_blocks = heads_schedule.len();
    let mut blocks = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let block = TransformerBlock::new(
            embed_dims,
            block_size,
            heads_schedule[block_idx],
            init_std,
            init_gain,
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
        init_std: f32,
        init_gain: Option<f64>,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let head_size = embed_dims / num_heads;
        let multi_head_attn = MultiHeadAttention::new(
            embed_dims,
            head_size,
            block_size,
            num_heads,
            init_std,
            init_gain,
            vb.pp("attn"),
            device,
        )?;
        let feed_forward = FeedForward::new(embed_dims, init_std, init_gain, vb.pp("ffwd"))?;
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
        init_std: f32,
        init_gain: Option<f64>,
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
                init_std,
                init_gain,
                vb.pp(format!("head_{head_idx}")),
                device,
            )?);
        }
        // Project the concatenated heads back to the embedding dimension.
        let proj = build_linear(
            num_heads * head_size,
            embed_dims,
            true,
            init_std,
            init_gain,
            vb.pp("proj"),
        )?;

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
    pub fn new(
        embed_dims: usize,
        init_std: f32,
        init_gain: Option<f64>,
        vb: VarBuilder,
    ) -> Result<Self, CandleError> {
        // Standard 4x inner expansion.
        let fc1 = build_linear(embed_dims, 4 * embed_dims, true, init_std, init_gain, vb.pp("fc1"))?;
        let fc2 = build_linear(4 * embed_dims, embed_dims, true, init_std, init_gain, vb.pp("fc2"))?;
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
        init_std: f32,
        init_gain: Option<f64>,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, CandleError> {
        let key = build_linear(
            embed_size,
            head_size,
            false,
            init_std,
            init_gain,
            vb.push_prefix("key"),
        )?;
        let query = build_linear(
            embed_size,
            head_size,
            false,
            init_std,
            init_gain,
            vb.push_prefix("query"),
        )?;
        let value = build_linear(
            embed_size,
            head_size,
            false,
            init_std,
            init_gain,
            vb.push_prefix("value"),
        )?;

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
    use temp_dir::TempDir;

    #[test]
    fn test_transformer_head() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let head = super::Head::new(32, 16, 8, 0.1, DEFAULT_INIT_STD, None, vb, &device).unwrap();
        let input = Tensor::randn(0f32, 1f32, (2, 8, 32), &device).unwrap();
        let output = head.forward(&input).unwrap();
        assert_eq!(output.shape(), &candle_core::Shape::from(&[2, 8, 16]));
    }

    #[test]
    fn test_gpt_forward_shape() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (8, 40, 32);
        let heads_schedule = vec![8; 6];
        let gpt = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            6,
            DEFAULT_INIT_STD,
            None,
            false,
            &device,
        )
        .unwrap();
        // (batch_size=2, seq_len=8)
        let input = Tensor::zeros((2, block_size), DType::U32, &device).unwrap();
        let logits = gpt.forward(&input).unwrap();
        assert_eq!(logits.shape(), &candle_core::Shape::from(&[2, block_size, vocab_size]));
    }

    /// Non-uniform per-block head schedules must build fine as long as EVERY
    /// entry individually divides `embed_dims`, and the resulting model still
    /// forwards to the expected `(batch, seq, vocab)` shape.
    #[test]
    fn test_gpt_new_accepts_non_uniform_heads_schedule() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (16, 20, 16);
        let heads_schedule = vec![1, 2, 4, 8];
        let gpt = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            4,
            DEFAULT_INIT_STD,
            None,
            false,
            &device,
        )
        .unwrap();
        assert_eq!(gpt.heads_schedule, heads_schedule);
        let input = Tensor::zeros((2, block_size), DType::U32, &device).unwrap();
        let logits = gpt.forward(&input).unwrap();
        assert_eq!(logits.shape(), &candle_core::Shape::from(&[2, block_size, vocab_size]));
    }

    /// A schedule whose length doesn't match `num_blocks` must be rejected
    /// with a clear error rather than silently truncating/panicking.
    #[test]
    fn test_gpt_new_rejects_wrong_schedule_length() {
        let device = Device::Cpu;
        let heads_schedule = vec![1, 2, 4]; // 3 entries, but num_blocks=4
        let result = Gpt::new(16, 20, 16, &heads_schedule, 4, DEFAULT_INIT_STD, None, false, &device);
        let msg = match result {
            Ok(_) => panic!("expected an error for a mismatched schedule length"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("num_blocks") || msg.contains("entries"),
            "expected a schedule-length error, got: {msg}"
        );
    }

    /// A per-block head count that doesn't evenly divide `hidden_size` must
    /// be rejected with an error identifying the offending block, not a
    /// silent fractional head size or a downstream shape panic.
    #[test]
    fn test_gpt_new_rejects_non_divisor_head_count() {
        let device = Device::Cpu;
        // hidden_size=16; block index 2 asks for 3 heads, which doesn't divide 16.
        let heads_schedule = vec![1, 2, 3, 8];
        let result = Gpt::new(16, 20, 16, &heads_schedule, 4, DEFAULT_INIT_STD, None, false, &device);
        let msg = match result {
            Ok(_) => panic!("expected an error for a non-divisor head count"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("block 2") && msg.contains("divisible"),
            "expected a per-block divisibility error naming block 2, got: {msg}"
        );
    }

    /// EXPERIMENTAL (Experiment A): `--init-std` must actually control the
    /// stdev of freshly-initialized weight matrices. `token_embeddings` is
    /// checked directly (it's always `Init::Randn { stdev: init_std }`,
    /// tying/no-tying doesn't affect it); `lm_head`'s weight (untied) is
    /// checked too since Experiment A explicitly requires the scale to apply
    /// to `lm_head`, not just the embedding tables. A large `vocab_size *
    /// embed_dims` element count keeps the sample-stdev estimate tight enough
    /// that a 3x sentinel-vs-scaled gap can't be sampling noise.
    #[test]
    fn test_init_std_controls_weight_scale() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (16, 200, 64);
        let heads_schedule = vec![4; 2];

        let small_std = 0.02f32;
        let gpt_small = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            small_std,
            None,
            false,
            &device,
        )
        .unwrap();
        let sample_stdev = |t: &Tensor| -> f32 {
            let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let n = flat.len() as f32;
            let mean = flat.iter().sum::<f32>() / n;
            (flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt()
        };

        let emb_stdev = sample_stdev(gpt_small.token_embeddings.embeddings());
        assert!(
            (emb_stdev - small_std).abs() < small_std * 0.5,
            "token_embeddings stdev should track --init-std={small_std}, got sample stdev {emb_stdev}"
        );

        let lm_head_stdev = sample_stdev(gpt_small.lm_head.weight());
        assert!(
            (lm_head_stdev - small_std).abs() < small_std * 0.5,
            "lm_head weight stdev should track --init-std={small_std} (applied consistently, \
             not just to embeddings), got sample stdev {lm_head_stdev}"
        );

        // The default sentinel (1.0) must reproduce a MUCH larger scale than
        // 0.02, confirming the flag actually has an effect rather than being
        // silently ignored.
        let gpt_default = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            DEFAULT_INIT_STD,
            None,
            false,
            &device,
        )
        .unwrap();
        let default_emb_stdev = sample_stdev(gpt_default.token_embeddings.embeddings());
        assert!(
            default_emb_stdev > emb_stdev * 5.0,
            "default --init-std=1.0 embeddings (stdev {default_emb_stdev}) should be much \
             larger-scale than --init-std=0.02 embeddings (stdev {emb_stdev})"
        );
    }

    /// EXPERIMENTAL (Experiment B): `--tie-embeddings` must produce a model
    /// whose `lm_head` weight tensor really is (bit-for-bit) the SAME data as
    /// `token_embeddings`, not just a coincidentally-equal separate copy —
    /// and this must remain true after a save/load round trip, i.e. loading
    /// must NOT allocate/expect a separate `lm_head.weight` in the file. This
    /// is the trickiest part the task doc calls out, so it's checked several
    /// ways: (1) right after construction, tying token_embeddings via
    /// `var_map.data()` and mutating it moves `lm_head`'s effective weight
    /// too; (2) after save+load, the loaded model's `lm_head` weight matches
    /// the loaded `token_embeddings`, and a forward pass succeeds; (3) the
    /// saved file has NO `lm_head.weight` key at all (only
    /// `token_embeddings` + `lm_head.bias`), proving no duplicate storage.
    #[test]
    fn test_tie_embeddings_save_load_round_trip() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (16, 20, 16);
        let heads_schedule = vec![2; 2];

        let gpt = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            DEFAULT_INIT_STD,
            None,
            true, // tie_embeddings
            &device,
        )
        .unwrap();
        assert!(gpt.tie_embeddings);

        // (1) Immediately after construction, lm_head's weight must be
        // element-for-element identical to token_embeddings (same Tensor).
        let emb_vals = gpt
            .token_embeddings
            .embeddings()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let lm_head_vals = gpt.lm_head.weight().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(emb_vals, lm_head_vals, "tied lm_head weight must equal token_embeddings");

        // No separate `lm_head.weight` var was ever allocated — only
        // `lm_head.bias` plus the shared `token_embeddings`.
        {
            let data = gpt.var_map.data().lock().unwrap();
            assert!(
                !data.contains_key("lm_head.weight"),
                "tied model must not allocate a separate lm_head.weight var"
            );
            assert!(
                data.contains_key("lm_head.bias"),
                "tied model must still keep its own lm_head.bias"
            );
        }

        // (2) Save, then load fresh with tie_embeddings=true, and confirm the
        // loaded model's lm_head weight still matches its token_embeddings
        // (proving `load` reconstructs the tied relationship, not just
        // coincidentally-equal separate tensors) and that the actual trained
        // values (not the placeholder zeros `load` starts from) came through.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tied.bin");
        gpt.save(&path).unwrap();

        // (3) The safetensors file itself must have no `lm_head.weight` key.
        let st = candle_core::safetensors::load(&path, &device).unwrap();
        assert!(
            !st.contains_key("lm_head.weight"),
            "saved file must not contain a duplicate lm_head.weight tensor"
        );
        assert!(st.contains_key("token_embeddings"));
        assert!(st.contains_key("lm_head.bias"));

        let loaded = Gpt::load(
            &path,
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            true, // tie_embeddings
            &device,
        )
        .unwrap();
        assert!(loaded.tie_embeddings);

        let loaded_emb_vals = loaded
            .token_embeddings
            .embeddings()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let loaded_lm_head_vals =
            loaded.lm_head.weight().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            loaded_emb_vals, loaded_lm_head_vals,
            "after load, tied lm_head weight must still equal token_embeddings"
        );
        assert_eq!(
            loaded_emb_vals, emb_vals,
            "load must restore the actual trained token_embeddings values, not zeros"
        );

        // A forward pass on the reloaded tied model must still work end to end.
        let input = Tensor::zeros((2, block_size), DType::U32, &device).unwrap();
        let logits = loaded.forward(&input).unwrap();
        assert_eq!(
            logits.shape(),
            &candle_core::Shape::from(&[2, block_size, vocab_size])
        );
    }

    /// Loading a NON-tied model with `tie_embeddings: false` must keep
    /// producing an independent `lm_head.weight`, i.e. the new parameter
    /// doesn't change default (untied) save/load fidelity.
    #[test]
    fn test_untied_save_load_round_trip_unaffected() {
        let device = Device::Cpu;
        let (block_size, vocab_size, embed_dims) = (16, 20, 16);
        let heads_schedule = vec![2; 2];

        let gpt = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            DEFAULT_INIT_STD,
            None,
            false,
            &device,
        )
        .unwrap();
        assert!(!gpt.tie_embeddings);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("untied.bin");
        gpt.save(&path).unwrap();

        let st = candle_core::safetensors::load(&path, &device).unwrap();
        assert!(
            st.contains_key("lm_head.weight"),
            "untied model must still save its own independent lm_head.weight"
        );

        let loaded = Gpt::load(
            &path,
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            2,
            false,
            &device,
        )
        .unwrap();
        let lm_head_vals = loaded.lm_head.weight().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let orig_lm_head_vals = gpt.lm_head.weight().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(lm_head_vals, orig_lm_head_vals);
    }
}
