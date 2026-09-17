from llm_sdk.llm_sdk import Small_LLM_Model
import json


def speak_with_llm(prompt):
    llm = Small_LLM_Model()

    # 1. Chargement du vocabulaire et extraction des IDs cibles
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)

    json_tokens = ["{", "}", "\"", ":", ","]
    json_token_ids = [vocab[t] for t in json_tokens if t in vocab]
    print(json_token_ids)


speak_with_llm("what is the sum of 2 and 3")