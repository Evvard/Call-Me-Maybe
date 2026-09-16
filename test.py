from llm_sdk.llm_sdk import Small_LLM_Model


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
    import json
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
    next_token_id = logit.index(max(logit))
    print("Pro Token: ", llm.decode(next_token_id))
    if next_token_id in json_tokens:
        print(next_token_id)
    else:



https://numpy.org/learn/













speak_with_llm("What the sum of 2 and 3")
