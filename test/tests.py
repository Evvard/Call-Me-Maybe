from llm_sdk import Small_LLM_Model
import json


def speak_with_llm(prompt: str) -> None:
    """Explore the SDK with one prompt when run explicitly."""
    llm = Small_LLM_Model()

    # 1. Load the vocabulary and extract the target IDs
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)

    json_tokens = ["{", "}", "\"", ":", ","]
    json_token_ids = [vocab[t] for t in json_tokens if t in vocab]
    print(json_token_ids)


if __name__ == '__main__':
    speak_with_llm("what is the sum of 2 and 3")
