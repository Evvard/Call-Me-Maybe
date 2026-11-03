"""Regression tests for function schemas and bundled input files."""

import unittest
from typing import Any

from pydantic import ValidationError

from src.parser import (
    FonctionDefinition, validation_function, validate_call,
)


class ValidationTests(unittest.TestCase):
    """Reject unsupported schemas before loading the model."""

    def definition(self, kind: str = 'number') -> dict[str, Any]:
        """Build an independent function definition fixture."""
        return {
            'name': 'add', 'description': 'Add a number.',
            'parameters': {'a': {'type': kind}},
            'returns': {'type': 'number'},
        }

    def test_unknown_parameter_type_is_rejected(self) -> None:
        """An unsupported type must never become a string implicitly."""
        with self.assertRaises(ValidationError):
            FonctionDefinition.model_validate(self.definition('numbre'))

    def test_extra_schema_fields_are_rejected(self) -> None:
        """Misspelled schema fields must be reported."""
        data = self.definition()
        data['unexpected'] = True
        with self.assertRaises(ValidationError):
            FonctionDefinition.model_validate(data)

    def test_bundled_definitions_load(self) -> None:
        """The default invocation must receive valid definitions."""
        self.assertEqual(len(validation_function(
            'data/input/functions_definition.json'
        )), 5)

    def test_output_requires_exact_keys(self) -> None:
        """Missing and unexpected arguments must never reach the output."""
        definition = FonctionDefinition.model_validate(self.definition())
        for params in ({}, {'a': 2.0, 'extra': 3.0}):
            with self.assertRaises(ValueError):
                validate_call('Add', definition, params)

    def test_output_types_and_finite_numbers(self) -> None:
        """Booleans, strings, infinity and NaN are not valid numbers."""
        definition = FonctionDefinition.model_validate(self.definition())
        for value in ('2', True, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                validate_call('Add', definition, {'a': value})
        result = validate_call('Add', definition, {'a': 2.0})
        self.assertEqual(result.parameters, {'a': 2.0})


if __name__ == '__main__':
    unittest.main()
