from typing import List, Set, Dict
import numpy

from llm_sdk.llm_sdk import Small_LLM_Model
from parser import FonctionDefinition


'''Le Llm choisis la bonne fonction en masquant les logit + generation'''


def select_valid_function(prompt: str, function:
                          List[FonctionDefinition],
                          model: Small_LLM_Model) -> str:
    function_names: List[str] = [f.name for f in function]
    token_name: Dict[str, List[int]] = {
        name: model.encode(name).tolist()[0] for name in function_names}

    system_prompt = (
        "You are an API dispatcher. "
        "Select the function that matches the user request.\n"
        f"Available functions: {', '.join(function_names)}"
    )

    formatted_context = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f'<|im_start|>assistant\n{{"name": "'
    )

    input_id: List[int] = model.encode(formatted_context).tolist()[0]
    generated_name_tokens: List[int] = []

    while True:
        nb_token_choice = len(generated_name_tokens)
        matching_branches = [seq for seq in token_name.values()
                             if seq[:nb_token_choice] == generated_name_tokens]

        valid_next_token: Set[int] = {
            seq[nb_token_choice] for seq in matching_branches
            if len(seq) > nb_token_choice}

        if not valid_next_token:
            break

        rw_logit = model.get_logits_from_input_ids(input_id)
        logit = numpy.array(rw_logit, dtype=numpy.float64)

        mask = numpy.full_like(logit, -numpy.inf)
        mask[list(valid_next_token)] = 0.0
        masked_logit = logit + mask

        choosen_token = int(numpy.argmax(masked_logit))

        input_id.append(choosen_token)
        generated_name_tokens.append(choosen_token)

    for name, seq in token_name.items():
        if seq == generated_name_tokens:
            return name
    raise ValueError("Aucune fonction correspondante n'a été identifiée.")
