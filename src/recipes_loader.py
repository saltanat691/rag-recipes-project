import json
from pathlib import Path
from typing import List
from .models import Recipe

def load_recipes(path: str) -> List[Recipe]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"recipes.json not found at: {p.resolve()}")

    data = json.loads(p.read_text(encoding="utf-8"))
    recipes: List[Recipe] = []
    for obj in data:
        recipes.append(Recipe(id=obj["id"], title=obj["title"], content=obj["content"]))
    return recipes
