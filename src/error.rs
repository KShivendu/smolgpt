use thiserror::Error;

#[derive(Debug, Error)]
pub enum SmolError {
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("Dataset error: {0}")]
    DatasetError(String),
    #[error("Invalid argument error: {0}")]
    InvalidArgument(String),
    #[error("Got error: {0}")]
    CustomError(String),
}

impl SmolError {
    pub fn dataset_error(msg: &str) -> Self {
        SmolError::DatasetError(msg.to_string())
    }

    pub fn invalid_argument(msg: &str) -> Self {
        SmolError::InvalidArgument(msg.to_string())
    }

    pub fn custom_error(msg: &str) -> Self {
        SmolError::CustomError(msg.to_string())
    }
}

pub type SmolResult<T> = Result<T, SmolError>;
