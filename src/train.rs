use crate::{
    args::Args,
    dataset::{self, Dataset},
    error::SmolError,
    model::BigramLM,
    tokenizer::{SimpleTokenizer, Tokenizer},
};
use candle_core::{Device, Shape, Tensor};

pub fn do_training(args: Args) -> Result<(), SmolError> {
    let Args {
        dataset_path,
        model_path,
        epochs,
        train,
        generate,
    } = args;
    let corpus = dataset::load_corpus(&dataset_path, false);
    let tokenizer = SimpleTokenizer::new(&corpus);
    let device = Device::Cpu;

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
    let vocab_size = tokenizer.vocab_size();

    if !args.train && !args.generate {
        return Err(SmolError::invalid_argument(
            "Either --train or --generate must be specified",
        ));
    }

    let mut model = if model_path.exists() {
        println!("Loading model from {}", model_path.display());
        BigramLM::load(&model_path, vocab_size, vocab_size, &device)?
    } else {
        BigramLM::new(vocab_size, vocab_size, &device)?
    };

    if train {
        model.train(&mut dataset, epochs, num_batches)?;
        model.save(&model_path)?;
    }

    if generate {
        let output = model.generate(500, &device)?;
        let decoded_output = tokenizer.decode(&output);
        println!("Generated text: {decoded_output}");
    }

    Ok(())
}
