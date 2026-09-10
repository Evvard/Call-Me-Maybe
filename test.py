from llm_sdk.llm_sdk import Small_LLM_Model


def speak_with_llm(prompt):
    llm = Small_LLM_Model()
    input_ids = llm.encode(prompt)
    print()
    print("Input IDs:", input_ids)
    print()


speak_with_llm("What the sum of 2 and 3")