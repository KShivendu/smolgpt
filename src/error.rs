use thiserror::Error;

#[derive(Debug, Error)]
pub enum SmolError {
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("Dataset error: {0}")]
    DatasetError(String),
    // #[error("Tokenizer error: {0}")]
    // TokenizerError(String),
}

impl SmolError {
    pub fn dataset_error(msg: &str) -> Self {
        SmolError::DatasetError(msg.to_string())
    }
}
