use crate::error::SmolResult;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder, VarMap};
use std::path::PathBuf;

pub struct BigramLM {
    pub token_embedding: Embedding,
    pub var_map: VarMap,
    pub vocab_size: usize,
}

impl BigramLM {
    pub fn new(vocab_size: usize, device: &Device) -> SmolResult<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (vocab_size, vocab_size),
            "embeddings",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let token_embedding = Embedding::new(embeddings, vocab_size);

        Ok(BigramLM {
            token_embedding,
            var_map,
            vocab_size,
        })
    }

    pub fn save(&self, path: &PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load(
        path: &PathBuf,
        vocab_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (vocab_size, vocab_size),
            "embeddings",
            Init::Const(0.0),
        )?;
        var_map.load(path)?;

        let token_embedding = Embedding::new(embeddings, vocab_size);

        Ok(BigramLM {
            token_embedding,
            var_map,
            vocab_size,
        })
    }
}

impl Module for BigramLM {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.token_embedding.forward(input)
    }
}
