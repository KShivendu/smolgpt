mod bigramlm;
mod gpt;

use crate::{
    args::ModelType,
    dataset::{Dataset, DatasetType},
    error::{SmolError, SmolResult},
    model::{bigramlm::BigramLM, gpt::Gpt},
};
use candle_core::{Device, IndexOp, Shape, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, loss, ops::softmax};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rngs::ThreadRng,
};

pub enum LanguageModel {
    BigramLM(BigramLM),
    Gpt(Gpt),
}

impl LanguageModel {
    pub fn new_bigram(vocab_size: usize, hidden_size: usize, device: &Device) -> SmolResult<Self> {
        let model = BigramLM::new(vocab_size, hidden_size, device)?;
        Ok(LanguageModel::BigramLM(model))
    }

    pub fn new_gpt(vocab_size: usize, embed_dims: usize, device: &Device) -> SmolResult<Self> {
        let model = Gpt::new(vocab_size, embed_dims, device)?;
        Ok(LanguageModel::Gpt(model))
    }

    pub fn new(
        model_type: ModelType,
        vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        match model_type {
            ModelType::Gpt => Self::new_gpt(vocab_size, hidden_size, device),
            ModelType::Bigram => Self::new_bigram(vocab_size, hidden_size, device),
        }
    }

    pub fn train(
        &self,
        dataset: &mut Dataset,
        num_epochs: usize,
        num_batches: usize,
    ) -> SmolResult<()> {
        let model = self.get_model();
        let vocab_size = self.get_vocab_size();
        let mut optimizer = AdamW::new(self.get_var_map().all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (stacked_x, stacked_y) =
                dataset.get_random_batches(DatasetType::Training, vocab_size, num_batches)?;
            let logits = model.forward(&stacked_x)?;
            // Batch size -> Number of sequences processed in parallel
            // Time size -> Number of tokens in each sequence (context length)
            // Channel size -> The dimension of each token's representation (here vocab size)
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

    #[cfg(test)]
    /// Returns the embeddings as a 2D vector for testing purposes
    fn get_embeddings_vec2d(&self) -> SmolResult<Vec<Vec<f32>>> {
        let embedding_table = match self {
            LanguageModel::BigramLM(model) => model.token_embedding.clone(),
            LanguageModel::Gpt(model) => model.token_embedding.clone(),
        };
        let flattened = embedding_table.embeddings().to_vec2::<f32>()?;
        Ok(flattened)
    }

    pub fn generate(
        &self,
        max_new_tokens: usize,
        rng: &mut ThreadRng,
        device: &candle_core::Device,
    ) -> SmolResult<Vec<u32>> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated_ids.push(0); // Start with a token, e.g., <BOS>
        let model = self.get_model();
        for i in 1..max_new_tokens {
            let logits = model.forward(&Tensor::from_vec(
                generated_ids.clone(),
                Shape::from(i),
                device,
            )?)?;
            let most_recent_logits = logits.i(i - 1)?;
            let probabilities = softmax(&most_recent_logits, 0)?;
            let vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(rng, &vec)?;
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }

    pub fn save(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        match self {
            LanguageModel::BigramLM(model) => model.save(path),
            LanguageModel::Gpt(model) => model.save(path),
        }
    }

    pub fn load(
        model_type: ModelType,
        path: &std::path::PathBuf,
        vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        match model_type {
            ModelType::Gpt => Self::load_gpt(path, vocab_size, hidden_size, device),
            ModelType::Bigram => Self::load_bigram(path, vocab_size, hidden_size, device),
        }
    }

    pub fn load_bigram(
        path: &std::path::PathBuf,
        vocab_size: usize,
        hidden_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let model = BigramLM::load(path, vocab_size, hidden_size, device)?;
        Ok(LanguageModel::BigramLM(model))
    }

    pub fn load_gpt(
        path: &std::path::PathBuf,
        vocab_size: usize,
        embed_dims: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let model = Gpt::load(path, vocab_size, embed_dims, device)?;
        Ok(LanguageModel::Gpt(model))
    }

    fn get_model(&self) -> &dyn candle_nn::Module {
        match self {
            LanguageModel::BigramLM(model) => model,
            LanguageModel::Gpt(model) => model,
        }
    }

    fn get_var_map(&self) -> &candle_nn::VarMap {
        match self {
            LanguageModel::BigramLM(model) => &model.var_map,
            LanguageModel::Gpt(model) => &model.var_map,
        }
    }

    fn get_vocab_size(&self) -> usize {
        match self {
            LanguageModel::BigramLM(model) => model.vocab_size,
            LanguageModel::Gpt(model) => model.vocab_size,
        }
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
    use rstest::rstest;
    use temp_dir::TempDir;

    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    fn test_lm_save_load_preserves_weights(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, hidden_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(vocab_size, hidden_size, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.pt");
        model.save(&path).unwrap();

        let loaded_model = match model_type {
            "bigramlm" => {
                LanguageModel::load_bigram(&path, vocab_size, hidden_size, &device).unwrap()
            }
            "gpt" => LanguageModel::load_gpt(&path, vocab_size, hidden_size, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        assert_eq!(model.get_vocab_size(), loaded_model.get_vocab_size());

        let original_embeddings = model.get_embeddings_vec2d().unwrap();
        let loaded_embeddings = loaded_model.get_embeddings_vec2d().unwrap();

        assert_eq!(original_embeddings, loaded_embeddings);
    }
}
