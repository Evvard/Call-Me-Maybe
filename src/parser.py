"""Load input files and enforce function-call schemas with Pydantic."""

import json
import math
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

ParameterValue = Union[str, float, bool]
ParameterType = Literal['number', 'string', 'boolean']


class InputDetail(BaseModel):
    """Describe one required argument and reject unknown fields."""

    model_config = ConfigDict(extra='forbid', strict=True)
    type: ParameterType


class OutputDetail(InputDetail):
    """Describe the return type without executing the function."""


class FonctionDefinition(BaseModel):
    """Validate a function name, description, arguments, and return type."""

    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, InputDetail]
    returns: OutputDetail


class PrompTest(BaseModel):
    """Preserve one original user prompt without type coercion."""

    model_config = ConfigDict(extra='forbid', strict=True)
    prompt: str


class FunctionCallResult(BaseModel):
    """Store the three output fields after function-specific validation."""

    model_config = ConfigDict(extra='forbid', strict=True)
    prompt: str
    name: str
    parameters: dict[str, ParameterValue]


def load_json_list(path_json: str | Path) -> list[Any]:
    """Read a UTF-8 JSON array or raise a contextual input error.

    Args:
        path_json: Location of the input JSON file.

    Returns:
        The decoded array, whose items still need schema validation.
    """
    path = Path(path_json)
    try:
        with path.open(encoding='utf-8') as stream:
            data: Any = json.load(stream)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f'Cannot read {path}: {error}') from error
    if not isinstance(data, list):
        raise ValueError(f'{path}: expected a JSON array.')
    return data


def validation_function(path_json: str | Path) -> list[FonctionDefinition]:
    """Load definitions and reject an empty set or duplicate names.

    Args:
        path_json: Path to the available function definitions.

    Returns:
        Validated definitions with unique names.
    """
    functions = [FonctionDefinition.model_validate(item)
                 for item in load_json_list(path_json)]
    if not functions:
        raise ValueError('The functions definition list cannot be empty.')
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise ValueError('Function names must be unique.')
    return functions


def load_prompts(path_json: str | Path) -> list[PrompTest]:
    """Read and validate prompts, allowing an empty input array.

    Args:
        path_json: Path to the prompt array.

    Returns:
        Prompts in their original order.
    """
    return [PrompTest.model_validate(item)
            for item in load_json_list(path_json)]


def validate_call(
    prompt: str,
    definition: FonctionDefinition,
    parameters: dict[str, ParameterValue],
) -> FunctionCallResult:
    """Check exact argument keys, scalar types, and finite numbers.

    Args:
        prompt: Original user request.
        definition: Function selected by the LLM.
        parameters: Extracted values to validate against that function.

    Returns:
        A serializable call, or raises ValueError for a schema mismatch.
    """
    if parameters.keys() != definition.parameters.keys():
        raise ValueError(f'{definition.name}: argument keys do not match.')
    for name, detail in definition.parameters.items():
        value = parameters[name]
        if detail.type == 'number':
            valid = (isinstance(value, (int, float))
                     and not isinstance(value, bool)
                     and math.isfinite(value))
        elif detail.type == 'boolean':
            valid = type(value) is bool
        else:
            valid = type(value) is str
        if not valid:
            raise ValueError(f'{definition.name}.{name}: expected '
                             f'{detail.type}, got {value!r}.')
    return FunctionCallResult(prompt=prompt, name=definition.name,
                              parameters=dict(parameters))
