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
3. The application reads the public vocabulary file and uses the SDK's `decode`
   method to obtain token text. It does not reimplement a tokenizer.
4. Required JSON keys and punctuation are injected from the schema. The model
   chooses argument values from tokens allowed by JSON prefix rules: numbers,
   escaped strings, and the boolean literals `true` and `false`.
5. JSON whitespace around numbers and booleans is allowed. String opening quotes
   are encoded together with their schema keys rather than as separate fragments.
   Tokens spanning a string's closing quote and following punctuation are accepted
   only when the remaining structure matches the expected next key.
6. The final call is checked against the selected function's exact argument keys
   and types, then serialized with `json.dump(..., allow_nan=False)`.

When only one token is allowed, the grammar imposes it without an LLM decision.
Otherwise the highest finite LLM score among allowed tokens is selected. This is
equivalent to masking all forbidden tokens with negative infinity. Function
selection does not use keyword heuristics or hardcoded answers.

## Design decisions

- Scalar arguments support `number`, `string` and `boolean`. Nested arguments
  and additional model-selection features are not implemented.
- The application uses the SDK's public methods. Torch and Transformers remain
  inside the supplied SDK; application code does not access SDK private fields.
- The SDK matches the copy in `llm_sdk.zip`. The application adds no inference
  cache or batching and sends the complete context for each logits request.
- Numbers are checked with Python JSON parsing. An explicit loop checks open
  strings and their escapes. The decoder uses no regular expressions.
- Token pieces decoded as the Unicode replacement character are excluded. Some
  rare Unicode text split across individual tokens may therefore be harder to
  generate; JSON Unicode escapes remain supported.
- Numbers are finite Python floats, with the usual precision limits. Invalid
  values are reported rather than replaced with invented zeroes.
- A 256-token argument limit reports unfinished values instead of silently
  truncating them. Strings retain their leading/trailing whitespace.
- Output is written directly with a context manager. There is no temporary-file
  recovery mechanism. An output path cannot overwrite either input file.
- Empty prompt arrays produce `[]` without loading model weights.

## Performance analysis

The mandatory targets are at least 90% correct calls, 100% valid schema-compliant
JSON, and a batch runtime under five minutes on standard hardware. Schema validity
and semantic accuracy are separate properties: a valid float can still be the
wrong number for a request.

The November 13 run after removing regular expressions, before restoring the
original SDK, processed the 11 bundled prompts in 267.29 seconds with the model
already downloaded. Runtime has not been measured again with the restored SDK.
All 11 outputs were
schema-compliant and selected the right function; 10 calls had correct arguments
(90.91%). One vowel replacement used `" *"` instead of `"*"`. This semantic error
remains unresolved. A separate three-prompt run with different function definitions
was schema-compliant in all cases but correct in only two: an explicitly empty
string became a single space. Across both runs, 12 of 14 calls were correct
(85.71%), below the 90% target. Full assignment compliance is therefore not
established. Detailed checks and limitations are recorded in the French `doc.md`.
Results do not guarantee equal accuracy on unseen prompts or the same runtime on
different hardware.

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
These checks are kept outside the submitted repository; no comprehensive bonus
test suite is added. The `test/` directory retains the original manual SDK
experiments, which run only when explicitly invoked. There is no `make test`
target in the current Makefile.

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

- The provided `en.subject.pdf`, version 1.7, defines the requirements.
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Python JSON module](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used for comment translation, assignment review, validation and
constrained-decoding corrections, dependency/lint setup, temporary verification
and documentation. It was also used to simplify the decoder and remove the added
inference cache, custom byte conversion and temporary-file output logic.
`parser.py` and `function_select.py` were left unchanged during this simplification.
The author must understand and be able to explain the submitted implementation.
