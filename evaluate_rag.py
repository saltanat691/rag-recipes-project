"""
Automated evaluation for RAG Recipe Assistant.

Metrics:
- Recall@3
- MRR@3 (PRIMARY acceptance metric)

Configurations:
1) Baseline: single-document indexing (one vector per recipe)
2) Enhanced: chunked indexing (ingredients+instructions) + per-recipe aggregation + LLM reranking

Acceptance rule:
- PASS if MRR@3 mean improves by >= 30% vs baseline mean (over N runs)

Usage:
  python evaluate_rag.py
  python evaluate_rag.py --runs 3
  python evaluate_rag.py --k 1 --runs 3
"""

import argparse
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from openai import OpenAI
import json
from pathlib import Path

load_dotenv()

# ------------------ CONFIG ------------------

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051

from src.eval_config import (
    EMBED_MODEL,
    RERANK_MODEL,
    BASE_COLLECTION,
    CHUNK_COLLECTION,
    IMPROVEMENT_THRESHOLD,
)

# ------------------ DATASET ------------------
def load_recipes():
    return json.loads(Path("recipes.json").read_text(encoding="utf-8"))
RECIPES = load_recipes()

# ------------------ METRICS ------------------

def recall_at_k(ranked_ids: List[str], expected: List[str], k: int) -> float:
    topk = ranked_ids[:k]
    return 1.0 if any(e in topk for e in expected) else 0.0

def reciprocal_rank(ranked_ids: List[str], expected: List[str], k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in expected:
            return 1.0 / i
    return 0.0

def safe_improvement(enh: float, base: float) -> float:
    # Guard divide-by-zero
    if base <= 0:
        return float("inf") if enh > 0 else 0.0
    return (enh - base) / base

# ------------------ CLIENTS ------------------

@dataclass
class EvalClients:
    def __post_init__(self):
        self.client = OpenAI()

    def embed(self, text: str) -> List[float]:
        return self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[text],
        ).data[0].embedding

    def rerank(self, question: str, candidates: List[Dict]) -> List[str]:
        """
        candidates: list of {"note_id": str, "text": str}
        returns: ordered list of note_ids
        """
        system = (
            "You are a strict reranker for recipe retrieval.\n"
            "Rank candidates by relevance to the question.\n"
            "Prefer the candidate that directly contains the asked-for fact (time/temp/amount).\n"
            "Return ONLY JSON: {\"ranked_ids\": [..]}.\n"
            "Use only provided IDs."
        )
        payload = {"question": question, "candidates": candidates}

        resp = self.client.chat.completions.create(
            model=RERANK_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        obj = json.loads(txt)
        ranked = obj.get("ranked_ids", [])

        # Safety: filter to only known ids and keep all candidates in some order
        known = [c["note_id"] for c in candidates]
        ranked = [rid for rid in ranked if rid in known]
        for rid in known:
            if rid not in ranked:
                ranked.append(rid)
        return ranked

# ------------------ WEAVIATE ------------------

def connect_db():
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    if not client.is_ready():
        raise RuntimeError("Weaviate not ready. Is Docker running?")
    return client

def recreate_collection(client, name, properties):
    if client.collections.exists(name):
        client.collections.delete(name)
    return client.collections.create(
        name=name,
        properties=properties,
        vector_config=Configure.Vectors.self_provided(),
    )

# ------------------ INGESTION ------------------

def ingest_baseline(col, clients: EvalClients):
    for r in RECIPES:
        vec = clients.embed(r["title"] + "\n" + r["content"])
        col.data.insert(
            properties={
                "note_id": r["id"],
                "title": r["title"],
                "content": r["content"],
            },
            vector=vec,
        )

def chunk_recipe(recipe: Dict) -> List[Tuple[str, str]]:
    parts = recipe["content"].split("Instructions:\n", 1)
    if len(parts) == 2:
        before, after = parts
        ingredients = before.strip()
        instructions = ("Instructions:\n" + after.strip()).strip()
    else:
        ingredients = recipe["content"]
        instructions = recipe["content"]

    if "Ingredients:\n" in ingredients:
        ingredients = "Ingredients:\n" + ingredients.split("Ingredients:\n", 1)[1].strip()

    return [
        ("ingredients", ingredients),
        ("instructions", instructions),
    ]

def ingest_chunked(col, clients: EvalClients):
    for r in RECIPES:
        for section, text in chunk_recipe(r):
            vec = clients.embed(f"{r['title']}\nsection:{section}\n{text}")
            col.data.insert(
                properties={
                    "note_id": r["id"],
                    "title": r["title"],
                    "section": section,
                    "content": text,
                },
                vector=vec,
            )

# ------------------ RETRIEVAL ------------------

def retrieve_baseline(col, qvec, limit=10) -> List[Tuple[str, str]]:
    res = col.query.near_vector(
        near_vector=qvec,
        limit=limit,
        return_metadata=["distance"],
    )
    out = []
    for obj in res.objects:
        out.append((obj.properties["note_id"], obj.properties["content"]))
    return out

def retrieve_chunked(col, qvec, limit=30):
    res = col.query.near_vector(
        near_vector=qvec,
        limit=limit,
        return_metadata=["distance"],
    )
    out = []
    for obj in res.objects:
        p = obj.properties
        out.append((
            p["note_id"],
            p.get("section", "full"),
            p["content"],
            p.get("title", "")
        ))
    return out


def aggregate_chunks_by_recipe(chunks: List[Tuple[str, str, str]], max_chars_per_recipe=800) -> Dict[str, str]:
    """
    Aggregate multiple chunks per recipe into a compact rerank text per recipe.
    Ensures we don't lose chunk evidence due to dict overwrites.
    """
    agg: Dict[str, List[str]] = {}
    for note_id, section, content in chunks:
        agg.setdefault(note_id, [])
        # keep section label so reranker understands context
        agg[note_id].append(f"{section}:\n{content}")

    joined: Dict[str, str] = {}
    for note_id, parts in agg.items():
        text = "\n\n".join(parts)
        joined[note_id] = text[:max_chars_per_recipe]
    return joined

def select_top_recipes_from_chunks(
        chunks: List[Tuple[str, str, str, str]],
        top_recipes: int = 12,
        max_chars_per_recipe: int = 900,
) -> List[Dict]:
    """
    Pick top recipes based on first occurrence in chunk retrieval list
    (proxy for nearest distance rank), then build rerank candidates.
    chunks items: (note_id, section, content, title)
    returns: list of {"note_id": str, "text": str}
    """
    order: List[str] = []
    per_recipe_parts: Dict[str, List[str]] = {}

    for note_id, section, content, title in chunks:
        if note_id not in per_recipe_parts:
            order.append(note_id)
            per_recipe_parts[note_id] = []
        per_recipe_parts[note_id].append(f"TITLE: {title}\nSECTION: {section}\n{content}")

    candidates = []
    for rid in order[:top_recipes]:
        txt = "\n\n".join(per_recipe_parts[rid])[:max_chars_per_recipe]
        candidates.append({"note_id": rid, "text": txt})
    return candidates

# ------------------ EVALUATION ------------------

def run_once(queries, k: int) -> Dict[str, float]:
    clients = EvalClients()
    db = connect_db()

    try:
        # Baseline collection
        base_col = recreate_collection(
            db,
            BASE_COLLECTION,
            [
                Property(name="note_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ],
        )
        ingest_baseline(base_col, clients)

        # Enhanced collection (chunked)
        chunk_col = recreate_collection(
            db,
            CHUNK_COLLECTION,
            [
                Property(name="note_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ],
        )
        ingest_chunked(chunk_col, clients)

        base_rrs, base_recalls = [], []
        enh_rrs, enh_recalls = [], []

        for q in queries:
            qvec = clients.embed(q["query"])
            expected = q["expected"]

            # --- Baseline ---
            base_hits = retrieve_baseline(base_col, qvec, limit=k)
            base_ranked_ids = [rid for rid, _ in base_hits]  # ordered
            base_recalls.append(recall_at_k(base_ranked_ids, expected, k))
            base_rrs.append(reciprocal_rank(base_ranked_ids, expected, k))

            # --- Enhanced: chunked retrieve + per-recipe aggregation + rerank ---
            chunk_hits = retrieve_chunked(chunk_col, qvec, limit=80)

            # Build a smaller, higher-quality candidate set for reranking
            candidates = select_top_recipes_from_chunks(
                chunk_hits,
                top_recipes=12,
                max_chars_per_recipe=1200,
            )

            ranked_ids = clients.rerank(q["query"], candidates)

            enh_recalls.append(recall_at_k(ranked_ids, expected, k))
            enh_rrs.append(reciprocal_rank(ranked_ids, expected, k))

        b_recall = sum(base_recalls) / len(base_recalls)
        b_mrr = sum(base_rrs) / len(base_rrs)
        e_recall = sum(enh_recalls) / len(enh_recalls)
        e_mrr = sum(enh_rrs) / len(enh_rrs)

        return {
            "baseline_recall": b_recall,
            "baseline_mrr": b_mrr,
            "enhanced_recall": e_recall,
            "enhanced_mrr": e_mrr,
        }

    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    # FIXED EVAL SET (expand this! keep constant across all runs)
    QUERIES_RETRIEVAL = [
        # Retrieval Benchmark
        # Ambiguous pasta queries (should separate r1 vs r2)
        {"query": "I have pasta and garlic. What should I cook for two?", "expected": ["r1"]},
        {"query": "Creamy pasta with mushrooms—how do I make it?", "expected": ["r2"]},
        {"query": "What pasta recipe uses butter and onion?", "expected": ["r2"]},
        {"query": "Tomato pasta—what do I add after draining?", "expected": ["r1"]},

        # Similar protein cooking queries (force right dish)
        {"query": "How do I know chicken is fully cooked? What temperature?", "expected": ["r3"]},
        {"query": "Bake fish with lemon—what temperature and time?", "expected": ["r4"]},

        # Short keyword-y queries
        {"query": "soy sauce sesame oil stir fry", "expected": ["r5"]},
        {"query": "kidney beans ground beef spices", "expected": ["r6"]},
        {"query": "romaine croutons parmesan chicken", "expected": ["r7"]},

        # Dessert ambiguity
        {"query": "quick chocolate cake in a cup microwave time", "expected": ["r9"]},
        {"query": "banana breakfast batter flip when bubbles", "expected": ["r10"]},

        # Edge phrasing / incomplete
        {"query": "overnight oats: how many hours?", "expected": ["r8"]},

        # Noisy queries
        {"query": "tomato pasta garlic time simmer", "expected": ["r1"]},
        {"query": "tommato spageti garlic simer how long", "expected": ["r1"]},
        {"query": "сколько минут тушить томаты для пасты?", "expected": ["r1"]},

        {"query": "cream mushroom pasta how much cream", "expected": ["r2"]},
        {"query": "cremy mushrom pasta cream ml?", "expected": ["r2"]},
        {"query": "сколько сливок в пасте с грибами?", "expected": ["r2"]},

        {"query": "mug cake microwave seconds", "expected": ["r9"]},
        {"query": "choc mug cake microwvave 45 or 60 sec", "expected": ["r9"]},
        {"query": "сколько секунд готовить кекс в кружке?", "expected": ["r9"]},
        {"query": "Quick beef chili: after adding tomatoes and beans, exactly how many minutes should it simmer?", "expected": ["r6"]},
        {"query": "Overnight oats: what is the minimum chill time in hours?", "expected": ["r8"]},

    ]

    QUERIES_PROJECT = [
        # Planning Benchmark
        {"query": "For a healthy family breakfast, how long should overnight oats chill?", "expected": ["r8"]},
        {"query": "Low-sugar breakfast: what are the exact amounts for overnight oats?", "expected": ["r8"]},
        {"query": "Breakfast for kids: how long do I cook banana pancakes on each side?", "expected": ["r10"]},
        {"query": "What ingredients do I need for banana pancakes, with quantities?", "expected": ["r10"]},

        {"query": "Chicken night (only once this week): what internal temperature should chicken thighs reach?", "expected": ["r3"]},
        {"query": "Chicken thighs: how long do I sear skin-side down?", "expected": ["r3"]},

        {"query": "Fish night (only once this week): what oven temperature do I bake salmon at?", "expected": ["r4"]},
        {"query": "How many minutes do I bake salmon fillets in the oven?", "expected": ["r4"]},

        {"query": "Meat-based dinner: what spices are in the quick beef chili?", "expected": ["r6"]},
        {"query": "Quick beef chili: how long should it simmer after adding beans?", "expected": ["r6"]},

        {"query": "Weeknight pasta: how long should I simmer tomatoes for simple tomato spaghetti?", "expected": ["r1"]},
        {"query": "Creamy mushroom pasta: how many ml of cream does it use?", "expected": ["r2"]},
        {"query": "Vegetable stir-fry: how long do I stir-fry the vegetables before adding soy sauce?", "expected": ["r5"]},

        {"query": "Tomato garlic pasta: how many minutes do I simmer the tomatoes?", "expected": ["r1", "r12", "r35", "r36", "r37"]},
        {"query": "Chocolate mug cake: should I microwave 45 seconds or 60 seconds?", "expected": ["r9", "r46", "r47", "r48"]},

    ]


    results = []
    for i in range(args.runs):
        r = run_once(QUERIES_RETRIEVAL, k=args.k)
        results.append(r)
        print(f"\n=== RUN {i+1}/{args.runs} (K={args.k}) ===")
        print(f"Baseline  Recall@{args.k}: {r['baseline_recall']:.3f}   MRR@{args.k}: {r['baseline_mrr']:.3f}")
        print(f"Enhanced  Recall@{args.k}: {r['enhanced_recall']:.3f}   MRR@{args.k}: {r['enhanced_mrr']:.3f}")

    # Means across runs
    b_mrrs = [r["baseline_mrr"] for r in results]
    e_mrrs = [r["enhanced_mrr"] for r in results]
    b_recalls = [r["baseline_recall"] for r in results]
    e_recalls = [r["enhanced_recall"] for r in results]

    b_mrr_mean = statistics.mean(b_mrrs)
    e_mrr_mean = statistics.mean(e_mrrs)
    b_recall_mean = statistics.mean(b_recalls)
    e_recall_mean = statistics.mean(e_recalls)

    mrr_improvement = safe_improvement(e_mrr_mean, b_mrr_mean)
    recall_improvement = safe_improvement(e_recall_mean, b_recall_mean)

    pass_mrr = (mrr_improvement >= IMPROVEMENT_THRESHOLD) and (b_mrr_mean > 0)
    pass_overall = pass_mrr  # primary metric = MRR

    print("\n=== SUMMARY (MEANS OVER RUNS) ===")
    print(f"Baseline mean  Recall@{args.k}: {b_recall_mean:.3f}   MRR@{args.k}: {b_mrr_mean:.3f}")
    print(f"Enhanced mean  Recall@{args.k}: {e_recall_mean:.3f}   MRR@{args.k}: {e_mrr_mean:.3f}")
    print(f"MRR improvement:   {mrr_improvement*100:.1f}%")
    print(f"Recall improvement:{recall_improvement*100:.1f}%")
    print("ACCEPTANCE:", "✅ PASS (>=30% MRR improvement)" if pass_overall else "❌ NOT YET")

    # Markdown block for your report
    print("\n=== REPORT SNIPPET (MARKDOWN) ===")
    print(f"""
### Evaluation summary (K={args.k}, runs={args.runs})

**Baseline runs (MRR@{args.k}):** {", ".join(f"{x:.3f}" for x in b_mrrs)}  
**Enhanced runs (MRR@{args.k}):** {", ".join(f"{x:.3f}" for x in e_mrrs)}

**Baseline mean MRR@{args.k}:** {b_mrr_mean:.3f}  
**Enhanced mean MRR@{args.k}:** {e_mrr_mean:.3f}

**Improvement% (using means):** {mrr_improvement*100:.1f}%  
**Threshold:** 30%

**Decision:** {"PASS" if pass_overall else "NOT YET"}
""".strip())

if __name__ == "__main__":
    main()
