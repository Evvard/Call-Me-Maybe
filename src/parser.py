import json
from pydantic import BaseModel, Field
from typing import Dict, List
from pathlib import Path


'''function def'''


class InputDetail(BaseModel):
    """Parametre en entree"""
    type: str = Field(description="Type entrer ('str, int, flot')")


class OutputDetail(BaseModel):
    """Parametre output"""
    type: str = Field(description="Type entrer ('str, int, flot')")


class FonctionDefinition(BaseModel):
    """Shema de la fonvction"""
    name: str = Field(description="Name of function")
    description: str = Field(description="Description")
    parameters: Dict[str, InputDetail] = Field(
        default_factory=dict, description="Dict of Parameters of function")
    returns: OutputDetail = Field(description="Type of return")


'''Function calling'''


class PrompTest(BaseModel):
    """Requete User"""
    prompt: str = Field(description="Prompte")


'''Validation of functions_definition.json et creation des dict'''


def validation_function(path_json: str | Path) -> List[FonctionDefinition]:
    path = Path(path_json)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier spécifié est introuvable: {path}")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Le fichier doit contenir une liste format JSON.")
        return [FonctionDefinition(**item) for item in data]
    except Exception as err:
        raise ValueError(f"Probbleme : {err}") from err


def load_prompts(path_json: str) -> List[PrompTest]:
    path = Path(path_json)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier spécifié est introuvable: {path}")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Le fichier doit contenir une liste format JSON.")
        return [PrompTest(**item) for item in data]
    except Exception as err:
        raise ValueError(f"Probbleme : {err}") from err


"""
if __name__ == "__main__":
    try:
        funcs = validation_function("data/input/functions_definition.json")
        print(f"OK {len(funcs)} fonctions chargées avec succès :")
        for fn in funcs:
params = [f"{name} ({detail.type})" for name, detail in fn.parameters.items()]
print(f"  - {fn.name}: {', '.join(params) if params else 'aucun paramètre'}")

        prompts = load_prompts("data/input/function_calling_tests.json")
        print(f"\nOK {len(prompts)} prompts de test chargés avec succès :")
        for p in prompts[:3]:  # On affiche les 3 premiers
            print(f"  - \"{p.prompt}\"")

    except Exception as e:
        print(f"NO Erreur : {e}")"""
