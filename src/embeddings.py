from dataclasses import dataclass
from typing import List
from openai import OpenAI

@dataclass
class OpenAIEmbeddingClient:
    model: str

    def __post_init__(self) -> None:
        self.client = OpenAI()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
