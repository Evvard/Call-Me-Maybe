"""Select a function using only LLM scores over valid name continuations."""

import json

from llm_sdk import Small_LLM_Model
from src.fn_add_nb import choose_token
from src.parser import FonctionDefinition


def select_valid_function(
    prompt: str, functions: list[FonctionDefinition], model: Small_LLM_Model,
) -> tuple[str, list[int]]:
    """Constrain generation to complete, JSON-quoted available names.

    Args:
        prompt: Natural-language request to dispatch without heuristics.
        functions: Validated definitions supplied by the input file.
        model: SDK instance providing encoding and next-token logits.

    Returns:
        The selected function name and the context including its final quote.
    """
    if not functions:
        raise ValueError('Cannot select from an empty function list.')
    descriptions = [item.model_dump() for item in functions]
    system = (
        'You are a function-calling API dispatcher. '
        'Return a JSON object with name and parameters. '
        'Copy argument values from the request; do not execute functions. '
        'For replacements, infer the regex from the request. '
        'Available functions: ' + json.dumps(descriptions, ensure_ascii=False)
    )
    context = (
        f'<|im_start|>system\n{system}<|im_end|>\n'
        f'<|im_start|>user\n{prompt}<|im_end|>\n'
        '<|im_start|>assistant\n{"name": '
    )
    current: list[int] = model.encode(context).tolist()[0]
    branches: dict[str, list[int]] = {
        function.name: model.encode(json.dumps(function.name)).tolist()[0]
        for function in functions
    }
    generated: list[int] = []
    while True:
        active = {name: sequence for name, sequence in branches.items()
                  if sequence[:len(generated)] == generated}
        finished = [name for name, sequence in active.items()
                    if sequence == generated]
        if finished:
            return finished[0], current
        allowed = [sequence[len(generated)] for sequence in active.values()
                   if len(sequence) > len(generated)]
        chosen = choose_token(current, allowed, model)
        current.append(chosen)
        generated.append(chosen)
