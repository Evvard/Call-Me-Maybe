*This project has been created as part of the 42 curriculum by eolivier.*

# Call Me Maybe

## Description

Translate natural-language requests into function calls with typed arguments,
using Qwen/Qwen3-0.6B through the provided LLM SDK. The program selects a function
and extracts its inputs; it does not execute that function. Constrained decoding
restricts the model's token choices to the supplied definitions and JSON grammar.

## Instructions

Use Python 3.10 or newer and install uv. From the repository root:

```sh
uv sync
uv run python -m src
```

The first run needs access to Hugging Face to download the model. Subsequent runs
can use the local cache. Model inference is performed locally; hardware and model
storage requirements are inherited from the supplied SDK.

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
make test
make clean
```

`make lint` runs `flake8 .` and the subject's required mypy options. Strict lint
runs `flake8 .` and `mypy . --strict`. Neither command suppresses SDK diagnostics.
Generated files, environments and caches are excluded from version control.

## Algorithm explanation

1. Pydantic validates both input arrays, the supported scalar types, unique function
   names and the absence of unknown schema fields.
2. The model receives the user request and available function definitions. A prefix
   tree of tokenized, JSON-quoted names restricts selection to existing functions.
   Including the closing quotation mark lets a short name finish even when another
   name starts with the same text.
3. The program injects structural punctuation and argument names from the schema.
   The LLM chooses argument values, with each candidate token checked against the
   relevant JSON prefix grammar before selection.
4. Numbers follow a finite-state grammar: optional minus, integer, optional decimal
   fraction and optional exponent. Boolean values are restricted to `true` and
   `false`. Strings track quotation marks, escapes, Unicode escapes and UTF-8 bytes.
5. Function-specific validation checks exact argument keys, scalar types and finite
   numeric values. `json.dump(..., allow_nan=False)` serializes the validated calls
   to a temporary sibling file, then atomically replaces the requested output.

The highest finite score among valid token IDs is equivalent to greedy selection
with all forbidden logits set to negative infinity. Function choice is always
based on LLM logits; there is no keyword dispatch or hardcoded answer table.

## Design decisions

- The application uses standard Python, NumPy, Pydantic and the SDK's public API.
  Torch and Transformers belong inside the supplied SDK, not the application.
- The vocabulary decoder targets Qwen's ByteLevel vocabulary. Extra model support
  would require checking its tokenizer representation first.
- Mandatory scalar arguments support `number`, `string` and `boolean`. Unsupported
  types are rejected explicitly; nested schemas are outside this implementation.
- Numbers are represented as finite Python floats. This entails ordinary float
  precision limits. Invalid numbers are never replaced with an invented zero.
- Every required argument is generated in declaration order. Strings preserve
  leading/trailing whitespace and escaped quotation marks.
- A 256-token bound per scalar reports an error instead of returning truncated data.
- Invalid input is reported before loading model weights. Empty prompt arrays produce
  `[]` without initializing the model. An output path cannot overwrite an input.
- The SDK's imports and annotations were adjusted for its installed package layout
  and current type stubs. Incremental inference now reuses its key/value cache
  for extended contexts; a new prompt resets the cache automatically. The public
  methods remain available and the application never accesses private SDK fields.

## Performance analysis

Syntax and schema validity are enforced independently of the model's semantic
accuracy. A well-typed argument can still be the wrong argument for a request.
The subject's targets are at least 90% correct complete calls and a batch runtime
below five minutes on standard hardware. Measured results are recorded in the
French `doc.md`; these targets must be evaluated rather than inferred from lint.

On the bundled 11 prompts, the measured pipeline time was 57.9 seconds with model
weights already cached locally (dependency import time is outside this timer).
All 11 outputs were valid and schema-compliant, all 11 functions were selected
correctly, and 10 complete calls were correct (90.91%). Regex spellings were
checked by their behavior on the supplied strings. One call generated `**`
instead of `*` as its replacement, so semantic accuracy was not 100%. This is a
measurement on the bundled examples, not a guarantee for unseen evaluation data.

## Challenges faced

The original number whitelist accepted malformed values such as `--2` and `1.2.3`.
A grammar now rejects these continuations before selection. The original string
stop condition discarded a whole token if it contained a quotation mark, losing
text before that mark. Byte-level prefix validation preserves such text and
supports escaped characters. Input schema typos, duplicate function names and
mismatched output types now produce explicit errors.

## Testing strategy

`make test` runs deterministic tests without model downloads or neural inference.
The suite covers schema rejection, bundled definitions, exact output keys, strict
scalar types, invalid numeric syntax, non-finite values, escaped strings, boolean
selection, shared function-name prefixes and generation limits.

The original manual SDK experiments are retained under `test/`; they execute only
when invoked directly. Model-based accuracy checks complement the deterministic
suite. Valid output files must contain exactly `prompt`, `name` and `parameters`.

## Example usage

Input prompt:

```json
{"prompt": "What is the sum of 2 and 3?"}
```

Expected call within the output array:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

- The local assignment, `en.subject.pdf`, version 1.7, defines the requirements.
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Python JSON module](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used to translate French code comments to English, compare the
implementation with the assignment, correct validation and constrained decoding,
repair dependency and lint configuration, write regression tests, run checks and
prepare the README and detailed French learning document. The author must review
and understand these changes and be able to explain them during evaluation.
