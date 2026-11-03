"""Decode scalar arguments using JSON prefix grammars and masked logits."""

import codecs
import json
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model
from src.parser import FonctionDefinition, ParameterType, ParameterValue


class TokenVocabulary(BaseModel):
    """Cache decoded byte pieces and fast-path string token groups."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pieces: dict[int, bytes]
    ordinary: list[int]
    special: list[int]
    numeric: list[int]


def build_vocabulary(vocab: dict[str, int]) -> TokenVocabulary:
    """Convert Qwen ByteLevel vocabulary symbols back to original bytes.

    Args:
        vocab: Public vocabulary JSON mapping token spellings to IDs.

    Returns:
        Cached byte pieces and candidate groups, excluding special tokens.
    """
    visible = list(range(33, 127)) + list(range(161, 173))
    visible += list(range(174, 256))
    mapping = {chr(value): value for value in visible}
    missing = [value for value in range(256) if value not in visible]
    mapping.update({chr(256 + i): value for i, value in enumerate(missing)})
    pieces: dict[int, bytes] = {}
    ordinary: list[int] = []
    special: list[int] = []
    numeric: list[int] = []
    for spelling, token_id in vocab.items():
        if not spelling or spelling.startswith('<|'):
            continue
        try:
            piece = bytes(mapping[char] for char in spelling)
        except KeyError:
            continue
        pieces[token_id] = piece
        if all(char in b'-+0123456789.eE' for char in piece):
            numeric.append(token_id)
        try:
            piece.decode('utf-8')
            safe_utf8 = True
        except UnicodeDecodeError:
            safe_utf8 = False
        if safe_utf8 and all(c >= 32 and c not in (34, 92) for c in piece):
            ordinary.append(token_id)
        else:
            special.append(token_id)
    return TokenVocabulary(pieces=pieces, ordinary=ordinary,
                           special=special, numeric=numeric)


def number_state(data: bytes) -> tuple[bool, bool]:
    """Return whether bytes form a valid JSON number prefix and full value.

    Args:
        data: Entire candidate numeric text, without surrounding whitespace.

    Returns:
        A pair (valid prefix, complete number), including exponent syntax.
    """
    state = 'start'
    for char in data:
        if state == 'start':
            next_state = ('sign' if char == 45 else 'zero' if char == 48
                          else 'integer' if 49 <= char <= 57 else '')
        elif state == 'sign':
            next_state = ('zero' if char == 48 else 'integer'
                          if 49 <= char <= 57 else '')
        elif state in ('zero', 'integer', 'fraction'):
            next_state = (state if 48 <= char <= 57 and state != 'zero'
                          else 'dot' if char == 46 and state != 'fraction'
                          else 'exponent' if char in (69, 101) else '')
        elif state in ('dot', 'exponent', 'exponent_sign', 'exponent_digits'):
            next_state = ('fraction' if state == 'dot' and 48 <= char <= 57
                          else 'exponent_digits' if state != 'dot'
                          and 48 <= char <= 57 else 'exponent_sign'
                          if state == 'exponent' and char in (43, 45) else '')
        else:
            return False, False
        if not next_state:
            return False, False
        state = next_state
    return True, state in ('zero', 'integer', 'fraction', 'exponent_digits')


def string_state(data: bytes) -> tuple[bool, bool]:
    """Validate a JSON string prefix, including escapes and partial UTF-8.

    Args:
        data: Candidate bytes beginning with an opening quotation mark.

    Returns:
        Whether the prefix is legal and whether a final quote completes it.
    """
    if not data:
        return True, False
    if data[0] != 34:
        return False, False
    escaped = False
    unicode_left = 0
    for index, char in enumerate(data[1:], 1):
        if unicode_left:
            if char not in b'0123456789abcdefABCDEF':
                return False, False
            unicode_left -= 1
        elif escaped:
            if char == 117:
                unicode_left = 4
            elif char not in b'"\\/bfnrt':
                return False, False
            escaped = False
        elif char == 34:
            if index != len(data) - 1:
                return False, False
            try:
                value = json.loads(data)
                value.encode('utf-8')
            except (ValueError, UnicodeError):
                return False, False
            return True, True
        elif char == 92:
            escaped = True
        elif char < 32:
            return False, False
    try:
        codecs.utf_8_decode(data, 'strict', False)
    except UnicodeError:
        return False, False
    return True, False


def primitive_state(data: bytes, kind: ParameterType) -> tuple[bool, bool]:
    """Dispatch prefix validation according to a declared scalar type.

    Args:
        data: Bytes already generated for the current scalar.
        kind: One of number, string, or boolean.

    Returns:
        Prefix validity and completion status.
    """
    if kind == 'number':
        valid, complete = number_state(data)
        if valid and complete:
            try:
                valid = math.isfinite(float(data))
            except (ValueError, OverflowError):
                valid = False
        return valid, complete and valid
    if kind == 'string':
        return string_state(data)
    return any(word.startswith(data) for word in (b'true', b'false')), \
        data in (b'true', b'false')


def choose_token(
    current_ids: list[int], allowed: list[int], model: Small_LLM_Model,
) -> int:
    """Select the highest finite logit among schema-compatible token IDs.

    Args:
        current_ids: Full conversation and tokens generated so far.
        allowed: Tokens that keep the current grammar valid.
        model: SDK instance providing the next-token logits.

    Returns:
        The chosen token ID, or raises ValueError if no choice is possible.
    """
    logits = np.asarray(model.get_logits_from_input_ids(current_ids),
                        dtype=np.float64)
    candidates = np.asarray(sorted(set(allowed)), dtype=np.int64)
    if candidates.size == 0 or candidates.min() < 0:
        raise ValueError('No valid token is available for this grammar.')
    if candidates.max() >= logits.size:
        raise ValueError('Vocabulary IDs exceed the model logits size.')
    scores = logits[candidates]
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.isfinite(scores).any():
        raise ValueError('The model supplied no finite valid logits.')
    return int(candidates[int(np.argmax(scores))])


def string_completion(
    data: bytes, suffix: str,
) -> tuple[bytes, str] | None:
    """Recognize a complete string plus a prefix of known schema syntax.

    Args:
        data: Candidate value bytes, possibly including following punctuation.
        suffix: Known delimiter and next argument key, or the closing brace.

    Returns:
        The complete literal and syntax still to inject, or None.
    """
    alternatives = [suffix, suffix.replace(', ', ',', 1)]
    for end in range(1, len(data)):
        if data[end] != 34:
            continue
        literal = data[:end + 1]
        if string_state(literal) != (True, True):
            continue
        tail = data[end + 1:]
        for expected in alternatives:
            encoded = expected.encode('utf-8')
            if encoded.startswith(tail):
                remaining = encoded[len(tail):].decode('utf-8')
                return literal, remaining
    return None


def decode_parameter(
    current_ids: list[int], kind: ParameterType, tokens: TokenVocabulary,
    model: Small_LLM_Model, delimiter: str, max_tokens: int = 256,
) -> ParameterValue:
    """Generate one scalar while allowing only legal JSON continuations.

    Args:
        current_ids: Context, extended in place with every selected token.
        kind: Required type of the current argument.
        tokens: Byte-level vocabulary cache.
        model: Public SDK interface for tokenization and logits.
        delimiter: Known syntax after this value, including the next key.
        max_tokens: Safety bound; exceeding it raises an explicit error.

    Returns:
        A correctly typed scalar, never a silently substituted default.
    """
    data = b'"' if kind == 'string' else b''
    if data:
        current_ids.extend(model.encode('"').tolist()[0])
    stop = model.encode(delimiter[0]).tolist()[0]
    if len(stop) != 1:
        raise ValueError('Expected a single-token JSON delimiter.')
    for _ in range(max_tokens):
        valid, complete = primitive_state(data, kind)
        if not valid:
            raise ValueError('Internal decoder state is invalid.')
        if complete and kind != 'number':
            current_ids.extend(model.encode(delimiter).tolist()[0])
            return parse_scalar(data, kind)
        allowed: list[int] = []
        if kind == 'string':
            # Ordinary UTF-8 text is safe only outside escape sequences.
            safe = data.endswith(b'"') and data == b'"'
            if not safe:
                safe = string_state(data + b'x')[0]
                safe = safe and not data.endswith(b'\\')
                try:
                    data.decode('utf-8')
                except UnicodeError:
                    safe = False
            candidates = tokens.special if safe else list(tokens.pieces)
            if safe:
                allowed.extend(tokens.ordinary)
        elif kind == 'number':
            candidates = tokens.numeric
        else:
            candidates = list(tokens.pieces)
        completions: dict[int, tuple[bytes, str]] = {}
        for token_id in candidates:
            candidate = data + tokens.pieces[token_id]
            if primitive_state(candidate, kind)[0]:
                allowed.append(token_id)
            elif kind == 'string':
                completion = string_completion(candidate, delimiter)
                if completion is not None:
                    completions[token_id] = completion
                    allowed.append(token_id)
        if kind == 'number' and complete:
            allowed.append(stop[0])
        chosen = choose_token(current_ids, allowed, model)
        if kind == 'number' and chosen == stop[0]:
            current_ids.extend(model.encode(delimiter).tolist()[0])
            return parse_scalar(data, kind)
        current_ids.append(chosen)
        if chosen in completions:
            literal, remaining = completions[chosen]
            current_ids.extend(model.encode(remaining).tolist()[0])
            return parse_scalar(literal, kind)
        data += tokens.pieces[chosen]
    # A final quote or boolean may have completed on the last iteration.
    if primitive_state(data, kind)[1] and kind != 'number':
        current_ids.extend(model.encode(delimiter).tolist()[0])
        return parse_scalar(data, kind)
    raise ValueError(f'{kind} generation exceeded {max_tokens} tokens.')


def parse_scalar(data: bytes, kind: ParameterType) -> ParameterValue:
    """Parse a complete scalar and reject non-finite numeric results.

    Args:
        data: Complete JSON literal produced by the constrained decoder.
        kind: Expected scalar type.

    Returns:
        A string, finite float, or boolean matching the declared type.
    """
    value: Any = json.loads(data)
    if kind == 'number':
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite numbers are not valid JSON output.')
        return number
    if kind == 'boolean' and type(value) is bool:
        return bool(value)
    if kind == 'string' and type(value) is str:
        return str(value)
    raise ValueError(f'Generated value does not match {kind}.')


def extract_all_parameters(
    selected_func: FonctionDefinition, current_ids: list[int],
    tokens: TokenVocabulary, model: Small_LLM_Model,
) -> dict[str, ParameterValue]:
    """Inject schema keys and decode each required argument in order.

    Args:
        selected_func: Function selected from the supplied definitions.
        current_ids: Context ending with the parameters object's open brace.
        tokens: Vocabulary prepared once for the complete input batch.
        model: SDK instance used for constrained token selection.

    Returns:
        All required arguments; the context ends with a closing brace.
    """
    result: dict[str, ParameterValue] = {}
    items = list(selected_func.parameters.items())
    if items:
        first_key = json.dumps(items[0][0]) + ': '
        current_ids.extend(model.encode(first_key).tolist()[0])
    for index, (name, detail) in enumerate(items):
        if index + 1 < len(items):
            suffix = ', ' + json.dumps(items[index + 1][0]) + ': '
        else:
            suffix = '}'
        result[name] = decode_parameter(
            current_ids, detail.type, tokens, model, suffix
        )
    if not items:
        current_ids.extend(model.encode('}').tolist()[0])
    return result
