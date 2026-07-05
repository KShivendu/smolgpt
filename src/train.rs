use std::time::Instant;

use crate::{
    args::{Args, TokenizerType},
    dataset::{self, Dataset},
    error::SmolError,
    model::LanguageModel,
    tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer},
};
use candle_core::{Device, Shape, Tensor};

pub fn do_training(args: Args) -> Result<(), SmolError> {
    let Args {
        dataset_path,
        model_path,
        epochs,
        train,
        generate,
        model_type,
        tokenizer: tokenizer_type,
        vocab_size: target_vocab_size,
    } = args;
    let corpus = dataset::load_corpus(&dataset_path, false);
    let device = Device::Cpu;

    let tokenizer: Box<dyn Tokenizer<u32>> = match tokenizer_type {
        TokenizerType::Char => Box::new(SimpleTokenizer::new(&corpus)),
        TokenizerType::Bpe => Box::new(BpeTokenizer::train(&corpus, target_vocab_size)),
    };
    println!(
        "Tokenizer: {:?}, vocab size: {}",
        tokenizer_type,
        tokenizer.vocab_size()
    );

    // Keep char- and BPE-trained models in separate files: their vocabularies
    // (and therefore embedding tables) are incompatible.
    let model_path = model_path.unwrap_or_else(|| {
        let suffix = match tokenizer_type {
            TokenizerType::Char => "char",
            TokenizerType::Bpe => "bpe",
        };
        match model_type {
            crate::args::ModelType::Gpt => format!("gpt-{suffix}.bin").into(),
            crate::args::ModelType::Bigram => format!("bigram-{suffix}.bin").into(),
        }
    });

    let encoded_corpus = tokenizer.encode(&corpus);
    let encoded_corpus_len = encoded_corpus.len();
    let data = Tensor::from_vec(encoded_corpus, Shape::from(encoded_corpus_len), &device)?;

    println!(
        "Encoded text tensor shape: {:?}; dtype {:?}",
        data.shape(),
        data.dtype()
    );

    let mut dataset = Dataset::new(data, 0.9)?;
    // debug_dataset(&mut dataset)?;

    let num_batches = 64;
    let block_size = 32;
    let hidden_size = 32;
    let vocab_size = tokenizer.vocab_size();

    if !args.train && !args.generate {
        return Err(SmolError::invalid_argument(
            "Either --train or --generate must be specified",
        ));
    }

    let model = if model_path.exists() {
        println!("Loading {model_type:?} model from {}", model_path.display());
        LanguageModel::load(
            model_type,
            &model_path,
            block_size,
            vocab_size,
            hidden_size,
            &device,
        )?
    } else {
        println!("Creating new {model_type:?} model");
        LanguageModel::new(model_type, block_size, vocab_size, hidden_size, &device)?
    };

    if train {
        let now = Instant::now();
        model.train(&mut dataset, &model_path, epochs, num_batches)?;
        println!("Training completed in {:.2?}", now.elapsed());
    }

    if generate {
        let rng = &mut rand::rng();
        let output = model.generate(500, rng, &device)?;
        let decoded_output = tokenizer.decode(&output);
        println!("Generated text: {decoded_output}");
    }

    Ok(())
}
