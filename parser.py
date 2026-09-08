from pydantic import BaseModel
from typing import List, Dict
import json


def parsing_prompt() -> str | None:
    try:
        file = "data/input/function_calling_tests.json"
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data)
    except FileNotFoundError | PermissionError:
        print(f"File {file} not exist")
        return None


def parsing_function() -> str | None:
    try:
        file = "data/input/functions_definition.json"
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data)
    except FileNotFoundError | PermissionError:
        print(f"File {file} not exist")
        return None


def dictionary_creation(obj: str) -> Dict[str, str] | None:
    word = ""
    prompt = {}
    start = 0
    count = 0
    for i in obj:
        if i == "{":
            start += 1
        if start == 1 and i == ("\"" or "\'"):
            count += 1
        if start == 1 and count == 3 and i != "}":
            word += i
        if i == "}":
            start = 0
            count = 0
            prompt.add({'prompt': word})
            word = ""
    return dict(prompt)




if __name__ == "__main__":
    prompte = parsing_prompt()
    prompt = {}

    function = parsing_function()

    prompt.update(dictionary_creation(prompte))

    print(prompt)
