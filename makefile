PACKAGE_NAME = call_me_maybe

install:
	uv pip install flake8 mypy
	uv pip install numpy pydantic
run:
	python3 

debug:
	python3 -m pdb 

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict