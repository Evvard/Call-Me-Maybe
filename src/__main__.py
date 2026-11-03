"""Module entry point executed via 'uv run python -m src'."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model
from src.fn_add_nb import build_vocabulary, extract_all_parameters
from src.function_select import select_valid_function
from src.parser import load_prompts, validate_call, validation_function


def parse_args() -> argparse.Namespace:
    """Parse paths to definitions, prompts, and the output JSON file.

    Returns:
        The three path arguments with the subject's default values.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to output JSON file",
    )
    return parser.parse_args()


def write_results(path: str, results: List[Dict[str, Any]]) -> None:
    """Serialize validated calls atomically, refusing NaN and infinity.

    Args:
        path: Output destination supplied by the command line.
        results: Fully validated calls in original prompt order.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=output.parent,
            prefix=output.name + '.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(results, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write('\n')
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    """Main function orchestrating the generation pipeline."""
    try:
        args = parse_args()
        if Path(args.output).resolve() in {
            Path(args.input).resolve(),
            Path(args.functions_definition).resolve(),
        }:
            raise ValueError('Output must not overwrite either input file.')
        started = time.monotonic()

        functions = validation_function(args.functions_definition)
        if not functions:
            raise ValueError("The functions definition list cannot be empty.")

        functions_map = {fn.name: fn for fn in functions}
        prompts = load_prompts(args.input)

        if not prompts:
            write_results(args.output, [])
            return

        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab: Dict[str, int] = json.load(f)

        tokens = build_vocabulary(vocab)
        results: List[Dict[str, Any]] = []

        for index, item in enumerate(prompts, 1):
            chosen_name, current_ids = select_valid_function(
                item.prompt, functions, model
            )

            transition = ', "parameters": {'
            current_ids.extend(model.encode(transition).tolist()[0])

            selected_func_def = functions_map[chosen_name]
            extracted_params = extract_all_parameters(
                selected_func_def, current_ids, tokens, model
            )

            validated_output = validate_call(
                item.prompt, selected_func_def, extracted_params
            )
            results.append(validated_output.model_dump())
            elapsed = time.monotonic() - started
            print(f'Processed {index}/{len(prompts)}: {chosen_name} '
                  f'({elapsed:.1f}s)', file=sys.stderr)

        write_results(args.output, results)

    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
