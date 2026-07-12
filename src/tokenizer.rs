use std::collections::HashMap;
use std::collections::HashSet;

type DefaultTokenIdType = u32;

pub trait Tokenizer<TokenId> {
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn vocab_size(&self) -> usize;
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

    fn vocab_size(&self) -> usize {
        self.charset.len()
    }
}

/// A byte-level Byte-Pair Encoding (BPE) tokenizer, in the same family as the
/// one used by GPT-2/GPT-3.
///
/// Training starts from the 256 raw bytes and greedily merges the most frequent
/// adjacent pair into a new token, repeating until the target vocabulary size is
/// reached (or no pair occurs more than once). Because the base vocabulary is the
/// full byte range, *any* UTF-8 input is representable — there is no `<unk>`.
///
/// Merges are confined to pre-tokenized "chunks" (maximal runs of whitespace vs.
/// non-whitespace) so the tokenizer never merges across word boundaries, which
/// both improves quality and keeps encoding fast.
pub struct BpeTokenizer {
    /// Adjacent pair -> the token id it merges into. The id doubles as the merge
    /// rank (lower id = learned earlier = higher priority).
    ranks: HashMap<(u32, u32), u32>,
    /// token id -> the raw bytes it expands to (used for decoding).
    vocab: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer on `corpus`, learning merges until the vocabulary
    /// reaches `target_vocab_size` tokens (must be >= 256).
    pub fn train(corpus: &str, target_vocab_size: usize) -> Self {
        // Base vocabulary: one token per byte value.
        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();

        // Build a {word -> frequency} table over pre-tokenized chunks. Working on
        // unique words (weighted by count) instead of the raw byte stream makes
        // training proportional to the vocabulary of the corpus, not its length.
        let mut word_freqs: HashMap<Vec<u32>, usize> = HashMap::new();
        for chunk in pretokenize(corpus) {
            let ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
            *word_freqs.entry(ids).or_default() += 1;
        }
        let mut words: Vec<(Vec<u32>, usize)> = word_freqs.into_iter().collect();

        let num_merges = target_vocab_size.saturating_sub(256);
        for i in 0..num_merges {
            // Count every adjacent pair across all words, weighted by frequency.
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for (word, freq) in &words {
                for pair in word.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += freq;
                }
            }

            // Pick the most frequent pair, breaking ties by the pair value so
            // training is fully deterministic.
            let best = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&pair, &count)| (pair, count));

            let Some((pair, count)) = best else { break };
            if count < 2 {
                break; // nothing left worth merging
            }

            let new_id = 256 + i as u32;
            ranks.insert(pair, new_id);
            let mut merged = vocab[pair.0 as usize].clone();
            merged.extend_from_slice(&vocab[pair.1 as usize]);
            vocab.push(merged);

            // Apply the merge inside every word.
            for (word, _) in words.iter_mut() {
                *word = merge_pair(word, pair, new_id);
            }
        }

        BpeTokenizer { ranks, vocab }
    }

    /// Encode a single pre-tokenized chunk by repeatedly applying the
    /// highest-priority (lowest-rank) merge that is currently present.
    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
        loop {
            let mut best_pair: Option<(u32, u32)> = None;
            let mut best_rank = u32::MAX;
            for w in ids.windows(2) {
                let pair = (w[0], w[1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }
            match best_pair {
                // `best_rank` is exactly the merged token's id.
                Some(pair) => ids = merge_pair(&ids, pair, best_rank),
                None => break,
            }
        }
        ids
    }
}

impl Tokenizer<DefaultTokenIdType> for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<DefaultTokenIdType> {
        pretokenize(text)
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    fn decode(&self, tokens: &[DefaultTokenIdType]) -> String {
        let mut bytes = Vec::new();
        for &token in tokens {
            if let Some(b) = self.vocab.get(token as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Replace every occurrence of `pair` in `ids` with the single token `new_id`.
fn merge_pair(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

/// Split text into chunks of maximal runs of whitespace vs. non-whitespace.
/// Merges are confined to within a chunk so the tokenizer never crosses word
/// boundaries (e.g. "Hello, world" -> ["Hello,", " ", "world"]).
fn pretokenize(text: &str) -> impl Iterator<Item = &str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut prev_ws: Option<bool> = None;
    for (i, c) in text.char_indices() {
        let ws = c.is_whitespace();
        if let Some(prev) = prev_ws {
            if prev != ws {
                chunks.push(&text[start..i]);
                start = i;
            }
        }
        prev_ws = Some(ws);
    }
    if start < text.len() {
        chunks.push(&text[start..]);
    }
    chunks.into_iter()
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

    #[test]
    fn test_bpe_roundtrip() {
        let corpus = "the cat sat on the mat. the cat ran.";
        let tokenizer = BpeTokenizer::train(corpus, 300);
        for query in ["the cat", "hello world!", "", "the the the"] {
            let encoded = tokenizer.encode(query);
            assert_eq!(tokenizer.decode(&encoded), query);
        }
    }

    #[test]
    fn test_bpe_roundtrip_unseen_bytes() {
        // Byte-level BPE must round-trip text it never saw during training,
        // including multi-byte UTF-8, since the base vocab covers all 256 bytes.
        let tokenizer = BpeTokenizer::train("aaaa bbbb aaaa", 270);
        let query = "totally unseen — café 🚀";
        let encoded = tokenizer.encode(query);
        assert_eq!(tokenizer.decode(&encoded), query);
    }

    #[test]
    fn test_bpe_merges_compress() {
        // A repetitive corpus should compress: BPE must use fewer tokens than
        // the raw byte count once merges are learned.
        let corpus = "abcabcabcabcabcabc";
        let tokenizer = BpeTokenizer::train(corpus, 300);
        let encoded = tokenizer.encode(corpus);
        assert!(
            encoded.len() < corpus.len(),
            "expected compression: {} tokens vs {} bytes",
            encoded.len(),
            corpus.len()
        );
    }

    #[test]
    fn test_bpe_vocab_size_grows() {
        let tokenizer = BpeTokenizer::train("the cat sat on the mat", 300);
        // 256 base bytes plus however many merges were learnable.
        assert!(tokenizer.vocab_size() > 256);
        assert!(tokenizer.vocab_size() <= 300);
    }
}
