"""
RAG Recipe Assistant using:
- Weaviate v4 (local, via Docker) as vector DB
- OpenAI API for embeddings + LLM
- Small recipe dataset
- Simple CLI UI

Pipeline:
1. User question (CLI)
2. OpenAI Embeddings -> vector
3. Weaviate near_vector search
4. OpenAI ChatCompletion with retrieved context
5. Answer printed to console
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv

import weaviate
from weaviate.classes.config import Property, DataType, Configure

from openai import OpenAI

from recipes_data import RECIPES

# ---------------------------------------------------------------------------
# Load env (for OPENAI_API_KEY, optional OPENAI_BASE_URL, etc.)
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

COLLECTION_NAME = "RecipeNote"

# OpenAI models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Rough safeguard to avoid over-long prompts
MAX_CONTEXT_CHARS = 8000

# ---------------------------------------------------------------------------
# OpenAI clients
# ---------------------------------------------------------------------------

@dataclass
class OpenAIEmbeddingClient:
    model: str

    def __post_init__(self) -> None:
        self.client = OpenAI()
        print(f"🔤 OpenAI embedding model: {self.model}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embeddings API supports batching; send all at once
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


@dataclass
class OpenAIChatClient:
    model: str

    def __post_init__(self) -> None:
        self.client = OpenAI()
        print(f"🧠 OpenAI chat model: {self.model}")

    def rag_answer(self, context: str, question: str) -> str:
        # Truncate context at character level as a simple safeguard
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS:]

        system_prompt = textwrap.dedent(
            """
            You are a precise recipe assistant working with a small recipe knowledge base.
            
            - Always use the exact ingredient amounts, times, and temperatures from CONTEXT.
            - If CONTEXT includes grams, milliliters, minutes, etc., repeat them accurately.
            - Answer in clear numbered steps when appropriate.
            - If the answer is not in CONTEXT, reply exactly:
              "I do not know based on the provided context."

            """
        ).strip()

        user_content = textwrap.dedent(
            f"""
            CONTEXT:
            {context}

            QUESTION:
            {question}
            """
        ).strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        return resp.choices[0].message.content.strip()

    def answer_without_context(self, question: str) -> str:
        """
        Baseline LLM answer WITHOUT RAG.

        We intentionally make this generic, so it does NOT give precise
        ingredient amounts or cooking times. This makes the benefit of RAG visible.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": (
                        "You are a casual home-cooking advisor.\n"
                        "- Give only high-level, generic advice.\n"
                        "- DO NOT mention exact gram weights, milliliters, temperatures or times.\n"
                        "- Use phrases like 'a bit of', 'some', 'for a few minutes', etc.\n"
                        "- Do NOT give numbered step-by-step instructions.\n"
                        "- Keep answers short: 3–5 sentences."
                    ),
                 },
                {"role": "user", "content": question},
            ],
        )

        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Weaviate wrapper
# ---------------------------------------------------------------------------

class VectorDb:
    def __init__(self) -> None:
        print(f"🔌 Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")
        self.client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )
        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready. Is Docker running?")

        self.collection = self._ensure_collection()

    def _ensure_collection(self):
        if self.client.collections.exists(COLLECTION_NAME):
            print(f"🧹 Collection '{COLLECTION_NAME}' already exists, deleting it to reset data...")
            self.client.collections.delete(COLLECTION_NAME)

        print(f"📚 Creating collection '{COLLECTION_NAME}'...")
        col = self.client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="note_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ],
            # BYO vectors, no built-in vectorizer
            vector_config=Configure.Vectors.self_provided(),
        )
        print("✅ Collection created.")
        return col

    def is_empty(self) -> bool:
        info = self.collection.aggregate.over_all(total_count=True)
        return info.total_count == 0

    def insert(self, notes: List[dict], vectors: List[List[float]]) -> None:
        print(f"⬆️ Ingesting {len(notes)} notes...")
        with self.collection.batch.fixed_size(batch_size=16) as batch:
            for rec, vec in zip(notes, vectors):
                batch.add_object(
                    properties={
                        "note_id": rec["id"],
                        "title": rec["title"],
                        "content": rec["content"],
                    },
                    vector=vec,
                )
        print("✅ Ingestion complete. Failed objects:", self.collection.batch.failed_objects)

    def search(self, vector: List[float], k: int = 4):
        return self.collection.query.near_vector(
            near_vector=vector,
            limit=k,
        )


# ---------------------------------------------------------------------------
# RAG app
# ---------------------------------------------------------------------------

class RagApp:
    def __init__(self) -> None:
        self.embedder = OpenAIEmbeddingClient(EMBEDDING_MODEL)
        self.llm = OpenAIChatClient(CHAT_MODEL)
        self.db = VectorDb()

        if self.db.is_empty():
            print("📥 Collection is empty, ingesting built-in recipes...")
            texts = [f"{r['title']}\n\n{r['content']}" for r in RECIPES]
            vecs = self.embedder.embed_texts(texts)
            self.db.insert(RECIPES, vecs)
        else:
            print("📥 Collection already has data, skipping ingestion.")

    @staticmethod
    def _format_context(results) -> str:
        chunks = []
        for obj in results.objects:
            p = obj.properties
            chunks.append(
                f"Title: {p['title']}\nContent: {p['content']}"
            )
        return "\n\n---\n\n".join(chunks)

    def answer(self, question: str) -> Tuple[str, str]:
        qvec = self.embedder.embed_query(question)
        results = self.db.search(qvec, k=3)
        if not results.objects:
            return "I do not know based on the provided context.", ""

        context = self._format_context(results)
        answer = self.llm.rag_answer(context=context, question=question)
        return answer, context


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    app = RagApp()

    print("\n✅ OpenAI RAG Recipe Assistant ready.")
    print("Ask any cooking question, or type 'exit'.\n")

    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            answer, context = app.answer(q)

            print("\n--- RAG ANSWER ---")
            print(answer)
            print("\n--- NO-RAG ANSWER (BASELINE LLM) ---")
            baseline = app.llm.answer_without_context(q)
            print(baseline)
    finally:
        app.db.client.close()
        print("🔒 Closed Weaviate client.")


if __name__ == "__main__":
    main()
