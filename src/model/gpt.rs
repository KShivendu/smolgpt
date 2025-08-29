use candle_core::{DType, Device};
use candle_nn::{Embedding, Init, Linear, Module, VarBuilder, VarMap, linear_b};

use crate::error::SmolResult;

pub struct Gpt {
    pub token_embedding: Embedding,
    pub lm_head: Linear,
    pub var_map: VarMap,
    pub vocab_size: usize,
}

impl Gpt {
    pub fn new(vocab_size: usize, embed_dims: usize, device: &Device) -> SmolResult<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (vocab_size, embed_dims),
            "embeddings",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        // Create the token embedding layer
        let token_embedding = Embedding::new(embeddings, embed_dims);
        // Convert embedding dimension back to vocabulary size
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder)?;

        Ok(Gpt {
            token_embedding,
            lm_head,
            var_map,
            vocab_size,
        })
    }

    pub fn save(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load(
        path: &std::path::PathBuf,
        vocab_size: usize,
        embed_dims: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings =
            var_builder.get_with_hints((vocab_size, embed_dims), "embeddings", Init::Const(0.0))?;
        let lm_head = linear_b(embed_dims, vocab_size, true, var_builder)?;
        var_map.load(path)?;

        Ok(Gpt {
            token_embedding: Embedding::new(embeddings, embed_dims),
            lm_head,
            var_map,
            vocab_size,
        })
    }
}

impl Module for Gpt {
    fn forward(
        &self,
        input: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let token_embeddings = self.token_embedding.forward(input)?; // (batch_size, seq_len, embed_dims)
        let logits = self.lm_head.forward(&token_embeddings)?; // (batch_size, seq_len, vocab_size)
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpt_model() {
        // Placeholder test to ensure the gpt module is included correctly.
        assert_eq!(2 + 2, 4);
    }
}
