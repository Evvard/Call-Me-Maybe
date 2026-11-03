"""Deterministic tests of JSON grammar safety without a model download."""

import json
import unittest
from typing import Any

from pydantic import BaseModel, Field

from src.function_select import select_valid_function
from src.parser import FonctionDefinition

from src.fn_add_nb import (
    build_vocabulary, decode_parameter, number_state, primitive_state,
    string_state,
)


class ArrayResult(BaseModel):
    """Emulate the public tensor's tolist method for test tokenization."""

    values: list[list[int]]

    def tolist(self) -> list[list[int]]:
        """Return the IDs as a batch of one sequence."""
        return self.values


class ScriptedModel(BaseModel):
    """Supply a fixed sequence of high-scoring next tokens."""

    vocabulary: dict[str, int]
    planned: list[int]
    position: int = 0
    contexts: list[list[int]] = Field(default_factory=list)

    def encode(self, text: str) -> ArrayResult:
        """Encode fixture characters, each represented by one token."""
        return ArrayResult(values=[[self.vocabulary[c] for c in text]])

    def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
        """Record context and prefer the next token in the fixture."""
        self.contexts.append(ids.copy())
        scores = [-10.0] * (max(self.vocabulary.values()) + 1)
        token = self.planned[min(self.position, len(self.planned) - 1)]
        self.position += 1
        scores[token] = 10.0
        return scores


class DecodingTests(unittest.TestCase):
    """Exercise malformed prefixes and lossless string extraction."""

    def test_invalid_number_prefixes(self) -> None:
        """Repeated signs, leading zeros, and duplicate dots are illegal."""
        for text in (b'--2', b'01', b'1.2.3', b'+2', b'1e+-2'):
            with self.subTest(text=text):
                self.assertFalse(number_state(text)[0])

    def test_complete_numbers_and_incomplete_prefixes(self) -> None:
        """Decimals and exponents require digits after their markers."""
        for text in (b'-2', b'0', b'1.25', b'-2e+3'):
            self.assertEqual(number_state(text), (True, True))
        for text in (b'', b'-', b'1.', b'1e', b'1e-'):
            self.assertEqual(number_state(text), (True, False))

    def test_quotes_and_escapes(self) -> None:
        """Escaped quotes and Unicode must survive JSON decoding."""
        text = json.dumps('  Hé "ami" \\ \n ☃  ', ensure_ascii=False)
        self.assertEqual(string_state(text.encode()), (True, True))
        for invalid in (b'"a"x', b'"a\n"', b'"\\x"', b'"\\u12x4"'):
            self.assertFalse(string_state(invalid)[0])

    def test_boolean_literals(self) -> None:
        """Only true and false can be generated as booleans."""
        self.assertEqual(primitive_state(b'tru', 'boolean'), (True, False))
        self.assertEqual(primitive_state(b'false', 'boolean'), (True, True))
        self.assertEqual(primitive_state(b'True', 'boolean'), (False, False))

    def test_multi_character_token_before_quote_is_preserved(self) -> None:
        """A token containing text and a final quote must retain its text."""
        vocab = {'"': 0, 'hello"': 1, ',': 2}
        model: Any = ScriptedModel(vocabulary=vocab, planned=[1])
        self.assertEqual(decode_parameter(
            [], 'string', build_vocabulary(vocab), model, ','
        ), 'hello')

    def test_generation_limit_raises(self) -> None:
        """Reaching a limit must not return a silently truncated string."""
        vocab = {'"': 0, 'x': 1, ',': 2}
        model: Any = ScriptedModel(vocabulary=vocab, planned=[1])
        with self.assertRaisesRegex(ValueError, 'exceeded'):
            decode_parameter([], 'string', build_vocabulary(vocab), model,
                             ',', max_tokens=2)

    def test_non_finite_numbers_are_masked(self) -> None:
        """An exponent overflowing finite float must be rejected."""
        self.assertEqual(primitive_state(b'1e309', 'number'), (False, False))

    def test_prefix_function_name_can_finish(self) -> None:
        """A short name must remain selectable beside a longer prefix name."""
        vocabulary = {chr(i): i for i in range(128)}
        functions = [FonctionDefinition.model_validate({
            'name': name, 'description': name, 'parameters': {},
            'returns': {'type': 'string'},
        }) for name in ('fn', 'fn_long')]
        model: Any = ScriptedModel(
            vocabulary=vocabulary, planned=[ord(c) for c in '"fn"']
        )
        name, context = select_valid_function('anything', functions, model)
        self.assertEqual(name, 'fn')
        self.assertEqual(context[-4:], [ord(c) for c in '"fn"'])

    def test_boolean_generation(self) -> None:
        """The decoder follows LLM scores for the complete boolean literal."""
        vocab = {char: index for index, char in enumerate('truefals,')}
        model: Any = ScriptedModel(
            vocabulary=vocab, planned=[vocab[c] for c in 'false']
        )
        self.assertIs(decode_parameter(
            [], 'boolean', build_vocabulary(vocab), model, ','
        ), False)

    def test_quote_and_comma_in_one_token(self) -> None:
        """A merged closing quote/comma must finish the value normally."""
        vocab = {'"': 0, 'hello",': 1, ',': 2}
        model: Any = ScriptedModel(vocabulary=vocab, planned=[1])
        self.assertEqual(decode_parameter(
            [], 'string', build_vocabulary(vocab), model, ',', max_tokens=1
        ), 'hello')

    def test_number_generation_with_exponent(self) -> None:
        """Negative decimal/exponent tokens produce the expected float."""
        vocab = {char: index for index, char in enumerate('-2.5e+3,')}
        model: Any = ScriptedModel(
            vocabulary=vocab, planned=[vocab[c] for c in '-2.5e+3,']
        )
        context: list[int] = []
        result = decode_parameter(
            context, 'number', build_vocabulary(vocab), model, ','
        )
        self.assertEqual(result, -2500.0)
        self.assertEqual(context[-1], vocab[','])

    def test_merged_token_can_start_next_argument_key(self) -> None:
        """Already generated comma and next-key bytes are not duplicated."""
        vocab = {chr(i): i for i in range(128)}
        vocab['hello",Ġ"b'] = 128
        model: Any = ScriptedModel(vocabulary=vocab, planned=[128])
        context: list[int] = []
        value = decode_parameter(
            context, 'string', build_vocabulary(vocab), model, ', "b": '
        )
        self.assertEqual(value, 'hello')
        self.assertEqual(context, [ord('"'), 128, ord('"'), ord(':'), 32])


if __name__ == '__main__':
    unittest.main()
