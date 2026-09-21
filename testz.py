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

    '''Token de fin + enlever le None de eos_token'''
    eos_tokens = {vocab.get("<|im_end|>"), vocab.get("<|endoftext|>")}
    eos_tokens.discard(None)

    '''Pour que le modele reponde a ma question'''
    formated = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_id = qwen.encode(formated).tolist()[0]
    print(f"\nInput id de [Prompt] : {input_id}\n")

    #logit = qwen.get_logits_from_input_ids(input_id)
    #next_token = logit.index(max(logit))
    #print(f"Next Token logit : {next_token} = {qwen.decode(next_token)}")

    for _ in range(120):
        logit = qwen.get_logits_from_input_ids(input_id)
        next_token = logit.index(max(logit))

        if next_token in eos_tokens:
            break

        input_id.append(next_token)

    response = qwen.decode(input_id)
    print("---\nReponse du model\n---\n", response)


tchat_llm("What is the sum of 2 and 3?")
