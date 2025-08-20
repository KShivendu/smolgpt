use crate::{
    dataset::{Dataset, DatasetType},
    error::{SmolError, SmolResult},
};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{
    AdamW, Embedding, Init, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops::softmax,
};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rngs::ThreadRng,
};
use std::path::PathBuf;

pub struct BigramLM {
    token_embedding: Embedding,
    var_map: VarMap,
    vocab_size: usize,
    rng: ThreadRng,
}

impl Module for BigramLM {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.token_embedding.forward(input)
    }
}

impl BigramLM {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> SmolResult<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (vocab_size, hidden_size),
            "embeddings",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let token_embedding = Embedding::new(embeddings, hidden_size);
        let rng = rand::rng();

        Ok(BigramLM {
            token_embedding,
            var_map,
            vocab_size,
            rng,
        })
    }

    pub fn save(&self, path: &PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load(
        path: &PathBuf,
        vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (vocab_size, hidden_size),
            "embeddings",
            Init::Const(0.0),
        )?;
        var_map.load(path)?;

        let token_embedding = Embedding::new(embeddings, hidden_size);
        let rng = rand::rng();

        Ok(BigramLM {
            token_embedding,
            var_map,
            vocab_size,
            rng,
        })
    }

    pub fn train(
        &self,
        dataset: &mut Dataset,
        num_epochs: usize,
        num_batches: usize,
    ) -> Result<(), SmolError> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (stacked_x, stacked_y) =
                dataset.get_random_batches(DatasetType::Training, self.vocab_size, num_batches)?;
            let logits = self.forward(&stacked_x)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape((batch_size * time_size, channel_size))?,
                &stacked_y.reshape((batch_size * time_size,))?,
            )?;
            // Looks like params.zero_grad()?; is not required in candle because gradients are accumulated externally
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            println!(
                "Epoch {}/{}: Loss = {}",
                epoch + 1,
                num_epochs,
                loss.to_scalar::<f32>()?
            );
        }

        Ok(())
    }

    pub fn generate(
        &mut self,
        max_new_tokens: usize,
        device: &Device,
    ) -> Result<Vec<u32>, SmolError> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated_ids.push(0); // Start with a token, e.g., <BOS>
        for i in 1..max_new_tokens {
            let logits = self.forward(&Tensor::from_vec(
                generated_ids.clone(),
                Shape::from(i),
                device,
            )?)?;
            let most_recent_logits = logits.i(i - 1)?;
            let probabilities = softmax(&most_recent_logits, 0)?;
            let vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(&mut self.rng, &vec)?;
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }
}

pub fn sample_multinomial(rng: &mut ThreadRng, prs: &Vec<f32>) -> SmolResult<u32> {
    let distribution = WeightedIndex::new(prs)
        .map_err(|e| SmolError::custom_error(&format!("Failed to create distribution: {}", e)))?;
    let next_token = distribution.sample(rng) as u32;
    Ok(next_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use temp_dir::TempDir;

    #[test]
    fn test_bigram_lm_save_load_preserves_weights() {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let model = BigramLM::new(vocab_size, hidden_size, &device).unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bigram_lm.pt");
        model.save(&path).unwrap();

        let loaded_model = BigramLM::load(&path, vocab_size, hidden_size, &device).unwrap();

        assert_eq!(model.vocab_size, loaded_model.vocab_size);

        let original_embeddings = model.token_embedding.embeddings().to_vec2::<f32>().unwrap();
        let loaded_embeddings = loaded_model
            .token_embedding
            .embeddings()
            .to_vec2::<f32>()
            .unwrap();

        // Check actual values as well. Tensor doesn't implement PartialEq, so we compare the data.
        assert_eq!(original_embeddings, loaded_embeddings);

        std::fs::remove_file(path).unwrap(); // Clean up
    }
}
