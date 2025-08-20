use std::collections::HashSet;

type DefaultTokenIdType = u32;

pub trait Tokenizer<TokenId> {
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
}

pub struct SimpleTokenizer {
    charset: Vec<char>,
}

impl SimpleTokenizer {
    pub fn new(corpus: &str) -> Self {
        let charset: HashSet<char> = corpus.chars().collect();
        let mut charset: Vec<char> = charset.into_iter().collect();
        charset.sort(); // Sort to ensure deterministic order
        SimpleTokenizer { charset }
    }

    pub fn vocab_size(&self) -> usize {
        self.charset.len()
    }
}

impl Tokenizer<DefaultTokenIdType> for SimpleTokenizer {
    fn encode(&self, text: &str) -> Vec<DefaultTokenIdType> {
        text.chars()
            .map(|c| self.charset.iter().position(|&x| x == c).unwrap_or(0) as DefaultTokenIdType)
            .collect()
    }

    fn decode(&self, tokens: &[DefaultTokenIdType]) -> String {
        tokens
            .iter()
            .map(|&token| self.charset.get(token as usize).cloned().unwrap_or(' '))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let corpus = "Hello, world!";
        let tokenizer = SimpleTokenizer::new(corpus);
        let query = "Hee";
        let encoded = tokenizer.encode(query);
        dbg!(&encoded);
        assert_eq!(tokenizer.decode(&encoded), query);
    }

    #[test]
    fn test_encode_empty_string() {
        let tokenizer = SimpleTokenizer::new("");
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_decode_empty_tokens() {
        let tokenizer = SimpleTokenizer::new("abc");
        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "");
    }
}
