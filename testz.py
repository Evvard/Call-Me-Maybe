from llm_sdk.llm_sdk import Small_LLM_Model
import json


def tchat_llm(prompt):
    print("\nDemarage\n")
    qwen = Small_LLM_Model()
    vocab_path = qwen.get_path_to_vocab_file()
    json_voc = ["{", "}", ",", "\"", ":"]

    with open(vocab_path) as f:
        vocab = json.load(f)
    print()
    for i in json_voc:
        print(f"Char shearch {i} -> id {vocab[i]}")

    input_id = qwen.encode(prompt).tolist()[0]
    print(f"\nInput id de [Prompt] : {input_id}\n")

    logit = qwen.get_logits_from_input_ids(input_id)
    next_token = logit.index(max(logit))
    print(f"Next Token logit : {next_token} = {qwen.decode(next_token)}")

    for _ in range(100):
        logit = qwen.get_logits_from_input_ids(input_id)
        next_token = logit.index(max(logit))
        input_id.append(next_token)

    response = qwen.decode(input_id)
    print("---\nReponse du model\n---\n", response)


tchat_llm("What is the sum of 2 and 3?")
