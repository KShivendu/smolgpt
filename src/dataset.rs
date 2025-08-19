use std::path::PathBuf;

pub fn load_corpus(path: &PathBuf, show_sample: bool) -> String {
    let text = std::fs::read_to_string(path).expect("Failed to read dataset file");

    println!("Length of the dataset: {}", text.len());

    if show_sample {
        println!("First 1000 characters of the dataset:");
        println!("{}", &text[..1000]);
    }

    text
}
