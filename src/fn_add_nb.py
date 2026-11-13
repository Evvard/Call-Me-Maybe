"""Generate scalar arguments with the SDK and simple JSON prefix rules."""

import json
import math
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model
from src.parser import FonctionDefinition, ParameterType, ParameterValue


def build_vocabulary(
    vocab: dict[str, int], model: Small_LLM_Model,
) -> dict[int, str]:
    """Read token text with the SDK instead of reimplementing a tokenizer."""
    pieces: dict[int, str] = {}
    for spelling, token_id in vocab.items():
        if spelling.startswith('<|'):
            continue
        text = model.decode([token_id])
        if text and '\ufffd' not in text:
            pieces[token_id] = text
    return pieces


def complete_number(text: str) -> bool:
    """Return whether text is a complete, finite JSON number."""
    try:
        value = json.loads(text)
        return type(value) in (int, float) and math.isfinite(value)
    except (ValueError, OverflowError):
        return False


def string_prefix(text: str) -> bool:
    """Check an open JSON string, allowing unfinished escape sequences."""
    if not text.startswith('"'):
        return False
    escaped = False
    missing_digits = 0
    for char in text[1:]:
        if missing_digits:
            if char not in '0123456789abcdefABCDEF':
                return False
            missing_digits -= 1
        elif escaped:
            if char == 'u':
                missing_digits = 4
            elif char not in '"\\/bfnrt':
                return False
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"' or ord(char) < 32:
            return False
    return True


def valid_prefix(text: str, kind: ParameterType) -> bool:
    """Return whether text can still become a value of the required type."""
    if kind == 'string':
        return string_prefix(text)
    if kind == 'number':
        if any(char not in '0123456789.-+eE \t\r\n' for char in text):
            return False
        # A missing digit after -, a decimal point, or e can be completed by 0.
        return complete_number(text) or complete_number(text + '0')
    text = text.lstrip(' \t\r\n')
    if text.rstrip(' \t\r\n') in ('true', 'false'):
        return True
    return 'true'.startswith(text) or 'false'.startswith(text)


def choose_token(
    current_ids: list[int], allowed: list[int], model: Small_LLM_Model,
) -> int:
    """Choose the best finite LLM score among allowed token IDs."""
    # A single allowed token is imposed by the schema, not chosen by the LLM.
    if len(set(allowed)) == 1 and allowed[0] >= 0:
        return allowed[0]
    logits = np.asarray(model.get_logits_from_input_ids(current_ids))
    if not allowed or min(allowed) < 0 or max(allowed) >= logits.size:
        raise ValueError('No valid token is available for this grammar.')
    scores = logits[allowed]
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.isfinite(scores).any():
        raise ValueError('The model supplied no finite valid logits.')
    return allowed[int(np.argmax(scores))]


def string_completion(text: str, suffix: str) -> tuple[str, str] | None:
    """Accept a JSON string followed only by the expected schema syntax."""
    try:
        value, end = json.JSONDecoder().raw_decode(text)
        if not isinstance(value, str):
            return None
        value.encode('utf-8')
    except (ValueError, UnicodeError):
        return None
    tail = text[end:]
    for expected in (suffix, suffix.replace(', ', ',', 1)):
        if expected.startswith(tail):
            return text[:end], expected[len(tail):]
    return None


def parse_scalar(text: str, kind: ParameterType) -> ParameterValue:
    """Convert a complete literal to its expected scalar Python type."""
    value: Any = json.loads(text)
    if kind == 'number':
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('The generated number must be finite.')
        return number
    if kind == 'string' and isinstance(value, str):
        return str(value)
    if kind == 'boolean' and isinstance(value, bool):
        return bool(value)
    raise ValueError(f'Generated value does not match {kind}.')


def decode_parameter(
    current_ids: list[int], kind: ParameterType, pieces: dict[int, str],
    model: Small_LLM_Model, suffix: str, max_tokens: int = 256,
) -> ParameterValue:
    """Generate one scalar while rejecting illegal next tokens."""
    text = '"' if kind == 'string' else ''
    stop = model.encode(suffix[0]).tolist()[0]
    if len(stop) != 1:
        raise ValueError('Expected a single-token JSON delimiter.')
    remaining = suffix
    for _ in range(max_tokens):
        allowed: list[int] = []
        endings: dict[int, tuple[str, str]] = {}
        for token_id, piece in pieces.items():
            candidate = text + piece
            if valid_prefix(candidate, kind):
                allowed.append(token_id)
            elif kind == 'string':
                ending = string_completion(candidate, suffix)
                if ending is not None:
                    allowed.append(token_id)
                    endings[token_id] = ending
        if kind == 'number' and complete_number(text):
            allowed.append(stop[0])
        chosen = choose_token(current_ids, allowed, model)
        if kind == 'number' and chosen == stop[0]:
            break
        current_ids.append(chosen)
        if chosen in endings:
            text, remaining = endings[chosen]
            break
        text += pieces[chosen]
        if kind == 'boolean' and text.strip() in ('true', 'false'):
            break
    else:
        raise ValueError(f'{kind} generation exceeded {max_tokens} tokens.')
    value = parse_scalar(text, kind)
    current_ids.extend(model.encode(remaining).tolist()[0])
    return value


def extract_all_parameters(
    selected_func: FonctionDefinition, current_ids: list[int],
    pieces: dict[int, str], model: Small_LLM_Model,
) -> dict[str, ParameterValue]:
    """Inject required keys and generate values according to their types."""
    parameters: dict[str, ParameterValue] = {}
    items = list(selected_func.parameters.items())
    if not items:
        current_ids.extend(model.encode('}').tolist()[0])
        return parameters
    first_key = json.dumps(items[0][0]) + ':'
    if items[0][1].type == 'string':
        first_key += ' "'
    current_ids.extend(model.encode(first_key).tolist()[0])
    for index, (name, detail) in enumerate(items):
        suffix = '}'
        if index + 1 < len(items):
            next_name, next_detail = items[index + 1]
            suffix = ', ' + json.dumps(next_name) + ':'
            if next_detail.type == 'string':
                suffix += ' "'
        parameters[name] = decode_parameter(
            current_ids, detail.type, pieces, model, suffix
        )
    return parameters
