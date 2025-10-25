use candle_core::{DType, Device};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder, VarMap, linear_b};

use crate::error::SmolResult;

pub struct Gpt {
    pub token_embeddings: Embedding,
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
        let token_embedding = Embedding::new(
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
        // Convert embedding dimension back to vocabulary size
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder)?;

        Ok(Gpt {
            token_embeddings: token_embedding,
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
        let token_embeddings = var_builder.get_with_hints(
            (vocab_size, embed_dims),
            "token_embeddings",
            Init::Const(0.0),
        )?;
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder)?;
        var_map.load(path)?;

        Ok(Gpt {
            token_embeddings: Embedding::new(token_embeddings, embed_dims),
            lm_head,
            var_map,
            block_size,
        })
    }
}

impl Module for Gpt {
    fn forward(
        &self,
        input: &candle_core::Tensor, // (batch_size, seq_len)
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let token_embedding = self.token_embeddings.forward(input)?; // (batch_size, seq_len, embed_dims)
        let logits = self.lm_head.forward(&token_embedding)?; // (batch_size, seq_len, vocab_size)
        Ok(logits)
    }
}
