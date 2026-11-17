*This project has been created as part of the 42 curriculum by eolivier.*

# Call Me Maybe

## Description

Convert natural-language requests into function names and scalar arguments with
Qwen/Qwen3-0.6B and the provided LLM SDK. The program prepares calls; it does not
execute functions or return their calculated results. This version targets the
mandatory assignment and does not add bonus features.

## Instructions

Use Python 3.10 or newer and uv. From the project root:

```sh
uv sync
uv run python -m src
```

The SDK downloads the model on first use if it is not already available locally.
No manual virtual-environment activation is needed for `uv run`.

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

make install
make run
make debug
make lint
make lint-strict
make clean
```

`make lint` runs `flake8 .` and `mypy .` with all options required by the subject.
`make lint-strict` also checks `mypy . --strict`. Both tools exclude the supplied
`llm_sdk` directory; mypy also skips analysis of that package through imports.
Generated environments, caches
and output files are excluded from Git. `make clean` removes project caches and
`data/output` while preserving the environment and Git metadata.

## Algorithm explanation

1. Pydantic validates the function definitions and prompts. Definitions must have
   unique nonempty names and supported scalar argument types.
2. The LLM chooses among tokenized, JSON-quoted function names. At each step,
   incompatible name continuations are excluded from token selection.
   Once a function is chosen, argument generation uses only that definition and
   the original request, with instructions to preserve string values exactly.
3. The application reads the public vocabulary file and uses the SDK's `decode`
   method to obtain token text. It does not reimplement a tokenizer.
4. Required JSON keys and punctuation are injected from the schema. The model
   chooses argument values from tokens allowed by JSON prefix rules: numbers, integers,
   escaped strings, and the boolean literals `true` and `false`.
5. JSON whitespace around scalar values is allowed. The model generates string
   opening quotes, so tokens containing both a quote and text remain available.
   Tokens spanning a string's closing quote and following punctuation are accepted
   only when the remaining structure matches the expected next key.
6. The final call is checked against the selected function's exact argument keys
   and types, then serialized with `json.dump(..., allow_nan=False)`.

When only one token is allowed, the grammar imposes it without an LLM decision.
Otherwise the highest finite LLM score among allowed tokens is selected. This is
equivalent to masking all forbidden tokens with negative infinity. Function
selection does not use keyword heuristics or hardcoded answers.

## Design decisions

- Scalar arguments support `number`, `integer`, `string` and `boolean`. Nested arguments
  and additional model-selection features are not implemented.
- The application uses the SDK's public methods. Torch and Transformers remain
  inside the supplied SDK; application code does not access SDK private fields.
- The supplied SDK is preserved. The application adds no inference
  cache or batching. Function selection sees all definitions; argument generation
  uses a shorter context containing only the selected definition and request.
- Numbers are checked with Python JSON parsing. An explicit loop checks open
  strings and their escapes, with a direct check for ordinary printable text.
  The decoder uses no regular expressions.
- Token pieces decoded as the Unicode replacement character are excluded. Some
  rare Unicode text split across individual tokens may therefore be harder to
  generate; JSON Unicode escapes remain supported.
- `number` values are finite Python floats, with the usual precision limits.
  `integer` values remain Python ints through validation and JSON serialization;
  booleans and decimal values are rejected for integer arguments. Invalid values
  are reported rather than replaced with invented zeroes.
- A 256-token argument limit reports unfinished values instead of silently
  truncating them. Strings retain their leading/trailing whitespace.
- Output is written directly with a context manager. There is no temporary-file
  recovery mechanism. An output path cannot overwrite either input file.
- Empty prompt arrays produce `[]` without loading model weights.

## Performance analysis

The targets are at least 90% correct calls, valid schema-compliant JSON, and
processing a test batch in under five minutes on standard hardware.

On November 17, complete runs with the original Qwen/Qwen3-0.6B SDK, followed by
the supplied grader, produced:

| Set | Grader score | Valid schemas | Elapsed time |
|---|---|---|---|
| Public, 11 prompts | 10/11 (90.9%) | 11/11 | 162.62 seconds |
| Private, 11 prompts | 10/11 (90.9%) | 11/11 | 173.96 seconds |

Measurements include model loading with weights already downloaded and use no
inference cache or batching added by the application. Diagnostic progress logging
was external to the application. The private run previously took 307.54 seconds
and scored 9/11 before these changes.

One public regex contains backspace characters instead of correctly matching
digits. One private template loses its embedded quotation marks. These remaining
semantic errors are distinct from JSON/schema validity. Results on these sets do
not guarantee the same accuracy or runtime on other inputs and machines.

## Challenges faced

A numeric character whitelist can accept `--2` or `1.2.3`; prefix grammar rules
reject those continuations. String tokens can contain both content and closing
punctuation, so discarding an entire token at the first quote loses data. The
standard JSON decoder identifies the literal boundary and the expected schema
suffix controls any trailing syntax. Exact final key/type validation prevents
extra or missing arguments from reaching successful output.

## Testing strategy

Verification uses the assignment's normal CLI, complete lint targets, and temporary
checks of malformed input, scalar grammar rules, string escapes and output types.
These checks and the original manual SDK experiments are kept outside the
submitted repository. No comprehensive bonus test suite is added. There is no
`make test` target in the current Makefile.

The supplied grading tool's private definitions contain `integer` parameters.
They previously caused input validation to stop before any output was generated.
The November 14 correction adds integer support to schema loading, constrained
generation, final validation and JSON serialization, without modifying the SDK
or the function selection module.

## Example usage

For `What is the sum of 2 and 3?`, the expected object in the output array is:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

- The 42 Call Me Maybe assignment, version 1.7, defines the requirements. Its PDF
  is a development reference and is not included in the submission.
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Python JSON module](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used for comment translation, assignment review and documentation.
