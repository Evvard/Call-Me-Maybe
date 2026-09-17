from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json


def speak_with_llm(prompt):
    llm = Small_LLM_Model()
    input_ids = llm.encode(prompt).tolist()[0]
    print(input_ids)
    print()

    print("Input IDs:", input_ids)
    print()

    logits = llm.get_logits_from_input_ids(input_ids)
    print("nombre de logit possible", len(logits))
    print()

    vocab_path = llm.get_path_to_vocab_file()

    with open(vocab_path) as f:
        vocab = json.load(f)
    print("Token shearch :", vocab["{"])

    json_tokens = ["{", "}", "\"", ":", ","]
    for token in json_tokens:
        if token in vocab:
            token_id = vocab[token]
            print(f"Token '{token}' → ID {token_id}")
        else:
            print(f"Token '{token}' non trouvé dans le vocabulaire.")
    print()

    prompt = "Reverse the string 'hello'"
    inputs_ids = llm.encode(prompt).tolist()[0]
    logit = llm.get_logits_from_input_ids(inputs_ids)

    forbidden_token_id = logit.index(max(logit))
    print(f"Token banni temporairement: '{llm.decode(forbidden_token_id)}' (ID: {forbidden_token_id})")

    logit[forbidden_token_id] = float('-inf')
    logit["{"] = float('+inf')

    next_token_id = logit.index(max(logit))
    print("Nouveau Pro Token après masquage: ", llm.decode(next_token_id))
    print()


""" 
    next_token_id = logit.index(max(logit))
    print("Pro Token: ", llm.decode(next_token_id))
    print()

    first = vocab["{"]
    input_ids = llm.encode(prompt).tolist()[0]
    input_ids.append(first)
    print(input_ids)


    next_logits = llm.get_logits_from_input_ids(input_ids)
    print(llm.decode((max(next_logits))))

"""







speak_with_llm("What the sum of 2 and 3")
