
# Healthy Family Meal Planner — RAG Project Description

## 1. Business Context & Motivation

Families with children who want to eat healthier at home face a recurring problem: daily meal planning creates decision fatigue, and many AI or recipe-based tools fail due to unrealistic assumptions, overly complex recipes, or ungrounded instructions.

This project addresses that gap by building a trust-first, recipe-grounded meal planning assistant that helps families maintain balanced, home-cooked meals with minimal daily effort.

The system is explicitly designed for:
- Beginner-to-intermediate home cooks
- Limited cooking time (30–60 minutes per meal)
- Basic kitchen equipment
- Familiar, locally available ingredients, especially Central Asian–adjacent cuisine

The system prioritizes precision, realism, and consistency over creativity or novelty.

---

## 2. Project Goal

The goal of this project is to build a Retrieval-Augmented Generation (RAG) system that generates coherent weekly meal plans grounded entirely in explicit recipe data.

Unlike generic recipe chatbots, the system:
- Does not invent recipes
- Does not give isolated suggestions
- Produces a 7-day plan covering breakfast, lunch, dinner, and optional snacks

All outputs must respect real household constraints and strict variety rules.

---

## 3. Target Users

- Families with children
- Beginner-to-intermediate home cooks
- Time-constrained households
- Preference for familiar, non-exotic ingredients

Cooking is treated as a routine necessity, not a hobby.

---

## 4. Key User Constraints

### Time & Effort
- 30–60 minutes per meal
- Preference for one-pot meals, batch cooking, and leftover reuse

### Skill Level
- Beginner-friendly recipes only
- No advanced culinary techniques

### Equipment
- Standard home kitchen
- No specialized appliances

---

## 5. Nutritional Principles

The system follows practical nutrition guidelines, not medical standards.

Primary focus:
- Higher protein
- Lower added sugar

Nutrition values are approximate and intended as guidance only.

---

## 6. Weekly Variety Rules (Hard Constraints)

Each weekly plan must enforce:
- Chicken: max 1 meal per week
- Fish/seafood: max 1 meal per week
- Remaining days: meat-based meals

These rules are non-negotiable.

---

## 7. Expected Output

- 7-day weekly meal plan (breakfast, lunch, dinner, optional snacks)
- Cooking notes (time, difficulty, leftovers)
- Aggregated grocery list grouped by category

---

## 8. RAG System — Main Idea

The system uses Retrieval-Augmented Generation (RAG) to ensure all outputs are grounded in explicit recipe data stored in a vector database (Weaviate).

The LLM is restricted to retrieved recipes only.

---

## 9. Dataset Concept

The dataset is intentionally small but richly annotated.

### Recipe Schema

Each recipe is stored as structured JSON with the following fields:
- id
- title
- content
- meal_type
- diet_tags
- primary_protein
- cook_time_min
- total_time_min
- difficulty
- leftovers_ok
- cuisine
- grocery_items

### Example Recipe

```json
{
  "id": "r12",
  "title": "Tomato Penne with Garlic",
  "content": "...",
  "meal_type": ["lunch", "dinner"],
  "diet_tags": ["low_sugar"],
  "primary_protein": "none",
  "cook_time_min": 15,
  "total_time_min": 20,
  "difficulty": "easy",
  "leftovers_ok": true,
  "cuisine": "central_asian_adjacent",
  "grocery_items": [
    { "name": "penne pasta", "qty": 160, "unit": "g", "category": "pantry" },
    { "name": "canned tomatoes", "qty": 400, "unit": "g", "category": "pantry" }
  ]
}
```

---

## 10. System Architecture

User Query → Embedding → Vector Search (Weaviate) → Retrieved Recipes → Constraint-Aware Prompt → LLM Output

---

## 11. Evaluation Criteria

- Constraint compliance
- Groundedness (no hallucinations)
- Practical feasibility
- Internal consistency

Creativity is explicitly not a primary metric.

---

## 12. Limitations

- Small dataset
- Approximate nutrition values
- No personalization
- CLI-only interface

---

## 13. Conclusion

This project demonstrates how a constraint-driven, trust-first RAG system can generate realistic, usable weekly meal plans by prioritizing grounded data over creative freedom.
