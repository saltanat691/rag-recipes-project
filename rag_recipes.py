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
# Tiny recipe dataset
# ---------------------------------------------------------------------------

RECIPES = [
    {
        "id": "r1",
        "title": "Simple Tomato Spaghetti",
        "content": (
            "Serves: 2\n"
            "Prep time: 5 minutes\n"
            "Cook time: 15 minutes\n\n"
            "Ingredients:\n"
            "- 160 g dried spaghetti\n"
            "- 2 tbsp olive oil (30 ml)\n"
            "- 2 cloves garlic, finely chopped\n"
            "- 400 g canned crushed tomatoes\n"
            "- 1/2 tsp salt (3 g)\n"
            "- 1/4 tsp black pepper (1 g)\n"
            "- 6 fresh basil leaves, torn\n"
            "- 20 g grated Parmesan (optional)\n\n"
            "Instructions:\n"
            "1. Bring 1.5 liters of water to a boil in a large pot, add 1 tbsp salt, and cook the spaghetti for 8–10 minutes until al dente.\n"
            "2. While the pasta cooks, heat 2 tbsp olive oil in a pan over medium heat for 1 minute.\n"
            "3. Add the chopped garlic and cook for 30–40 seconds until fragrant, stirring constantly so it does not burn.\n"
            "4. Add 400 g crushed tomatoes, 1/2 tsp salt, and 1/4 tsp pepper. Stir and simmer on low heat for 8 minutes.\n"
            "5. Drain the spaghetti, reserving 50 ml of the cooking water.\n"
            "6. Add the spaghetti to the pan with the sauce, pour in 2–3 tbsp (30–45 ml) of the reserved water, and toss for 1–2 minutes.\n"
            "7. Turn off the heat, add the torn basil leaves, and mix gently.\n"
            "8. Serve immediately with 10–20 g grated Parmesan on top if desired."
        ),
    },
    {
        "id": "r2",
        "title": "Creamy Mushroom Pasta",
        "content": (
            "Serves: 2\n"
            "Prep time: 10 minutes\n"
            "Cook time: 20 minutes\n\n"
            "Ingredients:\n"
            "- 160 g short pasta (penne or fusilli)\n"
            "- 1 tbsp olive oil (15 ml)\n"
            "- 20 g butter\n"
            "- 200 g mushrooms, sliced\n"
            "- 1 small onion (80 g), finely chopped\n"
            "- 1 clove garlic, minced\n"
            "- 150 ml heavy cream\n"
            "- 20 g grated Parmesan\n"
            "- 1/4 tsp salt (1–2 g)\n"
            "- 1/4 tsp black pepper (1 g)\n\n"
            "Instructions:\n"
            "1. Cook the pasta in boiling salted water for 9–11 minutes until al dente.\n"
            "2. While the pasta cooks, heat 1 tbsp olive oil and 20 g butter in a pan over medium heat for 1 minute.\n"
            "3. Add the chopped onion and cook for 3 minutes, stirring occasionally.\n"
            "4. Add the sliced mushrooms and cook for 5–6 minutes until they release their liquid and start to brown.\n"
            "5. Add the minced garlic and cook for 30 seconds.\n"
            "6. Pour in 150 ml heavy cream, add salt and pepper, and simmer on low heat for 3–4 minutes.\n"
            "7. Drain the pasta and add it to the pan with the sauce. Stir in 20 g grated Parmesan and cook for 1 minute.\n"
            "8. Serve hot. Add extra pepper or Parmesan on top if desired."
        ),
    },
    {
        "id": "r3",
        "title": "Garlic Butter Chicken Thighs",
        "content": (
            "Serves: 2\n"
            "Prep time: 10 minutes\n"
            "Cook time: 25 minutes\n\n"
            "Ingredients:\n"
            "- 4 chicken thighs (about 500 g), bone-in, skin-on\n"
            "- 1/2 tsp salt (3 g)\n"
            "- 1/2 tsp black pepper (2 g)\n"
            "- 20 g butter\n"
            "- 2 tbsp olive oil (30 ml)\n"
            "- 3 cloves garlic, minced\n"
            "- 1 tbsp lemon juice (15 ml)\n"
            "- 1 tbsp chopped fresh parsley (5 g)\n\n"
            "Instructions:\n"
            "1. Pat the chicken thighs dry and season both sides with 1/2 tsp salt and 1/2 tsp pepper.\n"
            "2. Heat 2 tbsp olive oil in a large pan over medium-high heat for 1–2 minutes.\n"
            "3. Place the chicken thighs skin-side down and sear for 6–7 minutes until the skin is golden and crisp.\n"
            "4. Flip the chicken, reduce heat to medium, and cook for another 6–8 minutes.\n"
            "5. Add 20 g butter and 3 minced garlic cloves to the pan. Cook for 1 minute, spooning the garlic butter over the chicken.\n"
            "6. Add 1 tbsp lemon juice, cover the pan with a lid, and simmer on low heat for 5 minutes, or until the internal temperature reaches 75°C.\n"
            "7. Turn off the heat, sprinkle with chopped parsley, and rest for 2 minutes before serving."
        ),
    },
    {
        "id": "r4",
        "title": "Lemon Herb Baked Salmon",
        "content": (
            "Serves: 2\n"
            "Prep time: 5 minutes\n"
            "Cook time: 15 minutes\n\n"
            "Ingredients:\n"
            "- 2 salmon fillets (about 150 g each)\n"
            "- 1 tbsp olive oil (15 ml)\n"
            "- 1/2 tsp salt (3 g)\n"
            "- 1/4 tsp black pepper (1 g)\n"
            "- 1 tbsp lemon juice (15 ml)\n"
            "- 1 tsp dried oregano (2 g)\n"
            "- 1 tsp dried dill or parsley (2 g)\n\n"
            "Instructions:\n"
            "1. Preheat the oven to 200°C (392°F).\n"
            "2. Line a baking tray with parchment paper and place the salmon fillets skin-side down.\n"
            "3. Brush each fillet with 1/2 tbsp olive oil and 1/2 tbsp lemon juice.\n"
            "4. Sprinkle salt, pepper, oregano, and dill evenly over the fillets.\n"
            "5. Bake for 12–15 minutes, depending on thickness, until the salmon flakes easily with a fork.\n"
            "6. Rest for 2 minutes before serving with lemon wedges."
        ),
    },
    {
        "id": "r5",
        "title": "Vegetable Stir-Fry with Rice",
        "content": (
            "Serves: 2\n"
            "Prep time: 10 minutes\n"
            "Cook time: 15 minutes (excluding rice)\n\n"
            "Ingredients:\n"
            "- 150 g broccoli florets\n"
            "- 1 medium carrot (80 g), sliced\n"
            "- 1 red bell pepper (120 g), sliced\n"
            "- 2 tbsp vegetable oil (30 ml)\n"
            "- 2 cloves garlic, minced\n"
            "- 1 tsp grated fresh ginger (5 g)\n"
            "- 3 tbsp soy sauce (45 ml)\n"
            "- 1 tsp sesame oil (5 ml)\n"
            "- 250 g cooked white rice (about 1 cup uncooked)\n\n"
            "Instructions:\n"
            "1. Prepare 250 g cooked rice in advance according to package instructions.\n"
            "2. Heat 2 tbsp vegetable oil in a wok or large pan over high heat for 1–2 minutes.\n"
            "3. Add minced garlic and grated ginger and stir-fry for 30 seconds.\n"
            "4. Add broccoli, carrot, and bell pepper. Stir-fry for 5–6 minutes until the vegetables are crisp-tender.\n"
            "5. Pour in 3 tbsp soy sauce and 1 tsp sesame oil and stir-fry for another 1–2 minutes.\n"
            "6. Serve the stir-fried vegetables over 125 g cooked rice per person."
        ),
    },
    {
        "id": "r6",
        "title": "Quick Beef Chili",
        "content": (
            "Serves: 3\n"
            "Prep time: 10 minutes\n"
            "Cook time: 30 minutes\n\n"
            "Ingredients:\n"
            "- 300 g ground beef\n"
            "- 1 small onion (80 g), chopped\n"
            "- 1 clove garlic, minced\n"
            "- 1 tbsp vegetable oil (15 ml)\n"
            "- 400 g canned chopped tomatoes\n"
            "- 200 g canned kidney beans, drained and rinsed\n"
            "- 1 tbsp tomato paste (15 g)\n"
            "- 1 tsp ground cumin (3 g)\n"
            "- 1 tsp smoked paprika (3 g)\n"
            "- 1/2 tsp chili powder (2 g)\n"
            "- 1/2 tsp salt (3 g)\n\n"
            "Instructions:\n"
            "1. Heat 1 tbsp oil in a pot over medium heat for 1 minute.\n"
            "2. Add chopped onion and cook for 3–4 minutes until soft.\n"
            "3. Add minced garlic and cook for 30 seconds.\n"
            "4. Add 300 g ground beef and cook for 5–6 minutes, breaking it up with a spoon, until no longer pink.\n"
            "5. Stir in tomato paste, cumin, smoked paprika, chili powder, and salt. Cook for 1 minute.\n"
            "6. Add canned tomatoes and kidney beans, stir, and bring to a simmer.\n"
            "7. Reduce heat to low and simmer uncovered for 15–20 minutes, stirring occasionally.\n"
            "8. Taste and adjust seasoning if needed before serving."
        ),
    },
    {
        "id": "r7",
        "title": "Chicken Caesar Salad",
        "content": (
            "Serves: 2\n"
            "Prep time: 15 minutes\n"
            "Cook time: 10 minutes\n\n"
            "Ingredients:\n"
            "- 200 g chicken breast\n"
            "- 1/2 tsp salt (3 g)\n"
            "- 1/4 tsp black pepper (1 g)\n"
            "- 1 tbsp olive oil (15 ml)\n"
            "- 100 g romaine lettuce, chopped\n"
            "- 30 g croutons\n"
            "- 20 g grated Parmesan\n"
            "- 2 tbsp Caesar dressing (30 ml)\n\n"
            "Instructions:\n"
            "1. Season 200 g chicken breast with 1/2 tsp salt and 1/4 tsp pepper.\n"
            "2. Heat 1 tbsp olive oil in a pan over medium heat for 1 minute.\n"
            "3. Cook the chicken breast for 4–5 minutes on each side, until cooked through. Rest for 3 minutes, then slice.\n"
            "4. In a large bowl, combine 100 g chopped romaine, 30 g croutons, and 20 g Parmesan.\n"
            "5. Add 2 tbsp Caesar dressing and toss gently.\n"
            "6. Top the salad with the sliced chicken and serve immediately."
        ),
    },
    {
        "id": "r8",
        "title": "Overnight Oats with Berries",
        "content": (
            "Serves: 1\n"
            "Prep time: 5 minutes (plus overnight chilling)\n"
            "Cook time: 0 minutes\n\n"
            "Ingredients:\n"
            "- 50 g rolled oats\n"
            "- 150 ml milk or plant-based milk\n"
            "- 50 g plain yogurt\n"
            "- 1 tsp honey or maple syrup (5 ml)\n"
            "- 40 g mixed berries (fresh or frozen)\n\n"
            "Instructions:\n"
            "1. In a jar or bowl, combine 50 g oats, 150 ml milk, 50 g yogurt, and 1 tsp honey.\n"
            "2. Stir well, cover, and refrigerate for at least 6 hours or overnight.\n"
            "3. In the morning, stir the oats, add 40 g berries on top, and serve chilled.\n"
            "4. If the mixture is too thick, add 1–2 tbsp extra milk (15–30 ml)."
        ),
    },
    {
        "id": "r9",
        "title": "Chocolate Mug Cake",
        "content": (
            "Serves: 1\n"
            "Prep time: 5 minutes\n"
            "Cook time: 1 minute\n\n"
            "Ingredients:\n"
            "- 20 g all-purpose flour (about 2 tbsp)\n"
            "- 10 g cocoa powder (1 tbsp)\n"
            "- 20 g sugar (2 tbsp)\n"
            "- 1/8 tsp baking powder (a small pinch)\n"
            "- 40 ml milk (about 3 tbsp)\n"
            "- 15 ml vegetable oil (1 tbsp)\n\n"
            "Instructions:\n"
            "1. In a microwave-safe mug (at least 250 ml), mix flour, cocoa, sugar, and baking powder.\n"
            "2. Add 40 ml milk and 15 ml oil. Stir until a smooth batter forms with no dry spots.\n"
            "3. Microwave on high (700–900 W) for 45–60 seconds, until the cake rises and the top looks set.\n"
            "4. Let cool for 2 minutes before eating. Do not overcook or the cake will become dry."
        ),
    },
    {
        "id": "r10",
        "title": "Banana Pancakes",
        "content": (
            "Serves: 2\n"
            "Prep time: 10 minutes\n"
            "Cook time: 10 minutes\n\n"
            "Ingredients:\n"
            "- 1 ripe banana (about 100 g), mashed\n"
            "- 1 egg\n"
            "- 80 ml milk (about 1/3 cup)\n"
            "- 60 g flour (about 1/2 cup)\n"
            "- 1/2 tsp baking powder (3 g)\n"
            "- 1 tbsp sugar (12 g)\n"
            "- 1 tbsp butter or oil (15 ml) for frying\n\n"
            "Instructions:\n"
            "1. In a bowl, mash the banana with a fork until smooth.\n"
            "2. Add the egg and 80 ml milk. Whisk until combined.\n"
            "3. Add flour, baking powder, and sugar. Gently whisk until a smooth batter forms.\n"
            "4. Heat a non-stick pan over medium heat and add 1 tbsp butter or oil.\n"
            "5. Pour 2–3 tbsp of batter per pancake into the pan.\n"
            "6. Cook each pancake for 2–3 minutes, until small bubbles form on the surface, then flip and cook for 1–2 more minutes.\n"
            "7. Repeat with the remaining batter. Serve warm with fruit or syrup."
        ),
    },
]



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
