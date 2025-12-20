import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    weaviate_host: str = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_http_port: int = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    weaviate_grpc_port: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    collection_name: str = os.getenv("WEAVIATE_COLLECTION", "RecipeNote")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
    recipes_path: str = os.getenv("RECIPES_PATH", "recipes.json")
