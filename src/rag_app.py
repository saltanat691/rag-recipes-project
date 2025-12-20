from typing import Tuple, List
from .embeddings import OpenAIEmbeddingClient
from .llm import OpenAIChatClient
from .vector_db import VectorDb
from .recipes_loader import load_recipes
from .models import Recipe
from .config import Settings

class RagApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = OpenAIEmbeddingClient(settings.embedding_model)
        self.llm = OpenAIChatClient(settings.chat_model, max_context_chars=settings.max_context_chars)
        self.db = VectorDb(
            host=settings.weaviate_host,
            http_port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
            collection_name=settings.collection_name,
        )

        recipes: List[Recipe] = load_recipes(settings.recipes_path)

        if self.db.is_empty():
            texts = [f"{r.title}\n\n{r.content}" for r in recipes]
            vecs = self.embedder.embed_texts(texts)

            notes = [{"note_id": r.id, "title": r.title, "content": r.content} for r in recipes]
            self.db.insert(notes, vecs)

    @staticmethod
    def _format_context(results) -> str:
        chunks = []
        for obj in results.objects:
            p = obj.properties
            chunks.append(f"Title: {p['title']}\nContent: {p['content']}")
        return "\n\n---\n\n".join(chunks)

    def answer(self, question: str, k: int = 3) -> Tuple[str, str]:
        qvec = self.embedder.embed_query(question)
        results = self.db.search(qvec, k=k)
        if not results.objects:
            return "I do not know based on the provided context.", ""

        context = self._format_context(results)
        answer = self.llm.rag_answer(context=context, question=question)
        return answer, context
