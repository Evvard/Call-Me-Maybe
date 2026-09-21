import json
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List
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
    paramaters: Dict[str, InputDetail] = Field(
        default_factory=dict, description="Dict of Parameters of function")
    returns: OutputDetail = Field(description="Type of return")


'''Function calling'''


class PrompTest(BaseModel):
    """Requete User"""
    prompt: str = Field(description="Prompte")
