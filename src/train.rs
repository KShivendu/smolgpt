use std::time::Instant;

use crate::{
    args::{Args, TokenizerType},
    dataset::{self, Dataset},
    error::SmolError,
    model::LanguageModel,
    tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer},
};
use candle_core::{Device, Shape, Tensor};
use rand::{rngs::StdRng, SeedableRng};

pub fn do_training(args: Args) -> Result<(), SmolError> {
    let Args {
        dataset_path,
        model_path,
        epochs,
        train,
        generate,
        eval,
        eval_samples,
        eval_min,
        eval_max,
        rft,
        rft_rounds,
        rft_prompts,
        rft_samples,
        rft_temperature,
        rft_epochs,
        rft_min,
        rft_max,
        patience,
        min_delta,
        model_type,
        tokenizer: tokenizer_type,
        vocab_size: target_vocab_size,
        seed,
        block_size,
        hidden_size,
        num_heads,
        num_blocks,
        serve,
        port,
        host,
    } = args;

    if !train && !generate && !eval && !rft && !serve {
        return Err(SmolError::invalid_argument(
            "Either --train, --generate, --eval, --rft, or --serve must be specified",
        ));
    }

    // --serve manages its own per-request model/tokenizer loading from
    // models.toml, so it neither needs nor benefits from the corpus/dataset/
    // model preamble below. Bail out early before the corpus read.
    if serve {
        crate::serve::run_serve(&host, port)?;
        return Ok(());
    }

    let corpus = dataset::load_corpus(&dataset_path, false);
    let device = Device::Cpu;

    let mut rng: StdRng = match seed {
        Some(s) => {
            println!("Using seeded RNG (seed = {s})");
            // NOTE: this seeds batch sampling (Dataset) and token sampling
            // (generate). Fresh model init still uses candle's CPU RNG, which
            // cannot be seeded in candle-core 0.9.1. For fully reproducible
            // runs, train once to save the model, then `--generate` loads it
            // from disk under the seed.
            StdRng::seed_from_u64(s)
        }
        None => StdRng::from_os_rng(),
    };

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

    let vocab_size = tokenizer.vocab_size();

    // Eval-only / RFT-only need just the tokenizer (built above) + a loaded
    // model. Skip the encoded-corpus tensor and Dataset construction in those
    // cases to keep them fast and avoid requiring the corpus to be encoded.
    // (The tokenizer still needs the corpus *string* for vocab scanning, which
    // is why `corpus` is loaded unconditionally above.) `--serve` is handled
    // earlier and never reaches this point, so it isn't mentioned here.
    let only_eval = eval && !train && !generate && !rft;
    let only_rft = rft && !train && !generate && !eval;

    let mut dataset: Option<Dataset> = None;
    if !only_eval && !only_rft {
        let encoded_corpus = tokenizer.encode(&corpus);
        let encoded_corpus_len = encoded_corpus.len();
        let data = Tensor::from_vec(encoded_corpus, Shape::from(encoded_corpus_len), &device)?;
        println!(
            "Encoded text tensor shape: {:?}; dtype {:?}",
            data.shape(),
            data.dtype()
        );
        dataset = Some(Dataset::with_rng(data, 0.9, rng.clone())?);
    }

    let num_batches = 64;

    // --eval / --rft never train from scratch, so the model file must already
    // exist on disk. (RFT does SFT *on the winners*, but it must start from a
    // pretrained model — there's no point sampling completions from a freshly
    // initialized model.)
    if (eval || rft) && !model_path.exists() {
        return Err(SmolError::invalid_argument(&format!(
            "{} requires an existing model file at {}; train first",
            if rft { "--rft" } else { "--eval" },
            model_path.display()
        )));
    }

    let model = if model_path.exists() {
        println!("Loading {model_type:?} model from {}", model_path.display());
        LanguageModel::load(
            model_type,
            &model_path,
            block_size,
            vocab_size,
            hidden_size,
            num_heads,
            num_blocks,
            &device,
        )?
    } else {
        println!("Creating new {model_type:?} model");
        LanguageModel::new(
            model_type,
            block_size,
            vocab_size,
            hidden_size,
            num_heads,
            num_blocks,
            &device,
        )?
    };

    if train {
        let dataset = dataset
            .as_mut()
            .expect("dataset must be built when --train is set");
        let now = Instant::now();
        // Dropout on for regular SFT; pass CLI patience/min_delta so `--train`
        // gets early stopping by default (patience=200, min_delta=0.001). Users
        // can disable with `--patience 0`.
        model.train_with_dropout(dataset, &model_path, epochs, num_batches, true, patience, min_delta)?;
        println!("Training completed in {:.2?}", now.elapsed());
    }

    if generate {
        println!("Generating from {model_type:?} model ({})", model_path.display());
        let output = model.generate(500, &mut rng, &device)?;
        let decoded_output = tokenizer.decode(&output);
        println!("Generated text: {decoded_output}");
    }

    if eval {
        println!(
            "Evaluating {model_type:?} model ({}) on {eval_samples} held-out arithmetic problems \
             (operands in [{eval_min}, {eval_max}])",
            model_path.display()
        );
        crate::eval::run_eval(
            &model,
            tokenizer.as_ref(),
            &device,
            eval_samples,
            eval_min,
            eval_max,
            block_size,
            seed,
        )?;
    }

    if rft {
        println!(
            "Running RFT on {model_type:?} model ({}) for {rft_rounds} rounds \
             (operands in [{rft_min}, {rft_max}])",
            model_path.display()
        );
        crate::rft::run_rft(
            &model,
            tokenizer.as_ref(),
            &device,
            &model_path,
            block_size,
            rft_rounds,
            rft_prompts,
            rft_samples,
            rft_temperature,
            rft_epochs,
            rft_min,
            rft_max,
            seed,
        )?;
    }

    Ok(())
}
