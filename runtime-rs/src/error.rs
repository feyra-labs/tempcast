use std::fmt;

/// Ошибка рантайма: текст с контекстом. Рантайм устройства не различает виды ошибок -
/// он либо откатывается к климатологии, либо начинает с чистого состояния.
#[derive(Debug)]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(format!("ввод-вывод: {e}"))
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error(format!("JSON: {e}"))
    }
}

impl<T> From<ort::Error<T>> for Error {
    fn from(e: ort::Error<T>) -> Self {
        Error(format!("ONNX Runtime: {e}"))
    }
}
