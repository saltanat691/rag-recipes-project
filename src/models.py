from dataclasses import dataclass

@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    content: str
