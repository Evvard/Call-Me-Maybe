"""Exercise file handling and the CLI without neural inference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.__main__ import write_results
from src.parser import load_prompts, validation_function


class FileAndCliTests(unittest.TestCase):
    """Check clear input failures and atomic, strict JSON output."""

    def test_missing_and_malformed_input(self) -> None:
        """Report an absent file, invalid JSON, and wrong root type."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prompts.json'
            with self.assertRaisesRegex(ValueError, 'Cannot read'):
                load_prompts(path)
            for content in ('[', '{}'):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_prompts(path)

    def test_empty_and_duplicate_definitions(self) -> None:
        """An ambiguous or empty dispatcher schema cannot be loaded."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'functions.json'
            definition = {'name': 'f', 'description': 'f', 'parameters': {},
                          'returns': {'type': 'string'}}
            for items in ([], [definition, definition]):
                path.write_text(json.dumps(items), encoding='utf-8')
                with self.assertRaises(ValueError):
                    validation_function(path)

    def test_empty_prompts_write_empty_array(self) -> None:
        """The documented invocation supports an empty batch offline."""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'empty.json'
            source.write_text('[]', encoding='utf-8')
            output = Path(folder) / 'nested' / 'output.json'
            completed = subprocess.run(
                [sys.executable, '-m', 'src', '--input', str(source),
                 '--output', str(output)], capture_output=True, text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text()), [])

    def test_failure_preserves_previous_output(self) -> None:
        """A serialization error must not truncate the previous result."""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output.json'
            output.write_text('[42]', encoding='utf-8')
            with self.assertRaises(ValueError):
                write_results(str(output), [{'number': float('nan')}])
            self.assertEqual(output.read_text(), '[42]')
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_cli_errors_are_clear(self) -> None:
        """Invalid input exits nonzero without an unhandled traceback."""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'bad.json'
            source.write_text('{', encoding='utf-8')
            result = subprocess.run(
                [sys.executable, '-m', 'src', '--input', str(source)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Error:', result.stderr)
            self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
