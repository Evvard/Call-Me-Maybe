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


def dictionary_creation(object: str) -> Dict[str, str] | None:
    try:
        prompt = ""
        sep = 0
        for i in object:
            if i in ("\'" or "\""):
                print("test")
                sep += 1
            if sep == 3:
                prompt += i
        prompt.strip("}")
        return {'prompt': prompt}
    except Exception as m:
        print(m)
        return None


if __name__ == "__main__":
    prompte = parsing_prompt()
    prompt = {}

    function = parsing_function()

    for i in prompte:
        print(i)
        prompt.update(dictionary_creation(i))
        
    print(prompt)
