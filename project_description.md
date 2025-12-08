
# RAG Recipe Assistant — Project Description

## 1. Main Idea

The goal of this project is to build a Retrieval-Augmented Generation (RAG) system that answers cooking-related questions with precise, factual, recipe-grounded information. The system uses a vector database (Weaviate) to store recipe embeddings and retrieves the most relevant recipe based on a user query. An LLM (OpenAI GPT model) then uses this retrieved context to produce an accurate, recipe-specific answer.

This project demonstrates how RAG improves accuracy, reduces hallucinations, and provides domain-specific, grounded responses.

## 2. Key Concepts

### Retrieval-Augmented Generation (RAG)
A technique that combines:
- Retrieval: find relevant documents using vector similarity search
- Generation: feed retrieved data into an LLM to guide the response  

### Vector Embeddings
Text converted into numeric vectors enabling semantic similarity search.

### Vector Database
Stores vectors and metadata. Implemented using Weaviate v4.

### LLM with Context
The GPT model receives both the user query and the retrieved recipe and generates grounded answers.

## 3. Dataset Concept

A custom dataset of 10 structured recipes, each containing:
- Ingredient quantities (grams, ml, tsp)
- Prep and cooking times
- Servings
- Numbered instruction steps

Examples:
- Tomato Spaghetti  
- Creamy Mushroom Pasta  
- Garlic Butter Chicken  
- Vegetable Stir-Fry  
- Overnight Oats  
- Chocolate Mug Cake  
- Banana Pancakes  
(and more)

Stored as:

```
{
  "id": "r1",
  "title": "Simple Tomato Spaghetti",
  "content": "... structured recipe ..."
}
```

## 4. System Architecture & Design

### High-Level Flow
```
User Query → Embed Query → Vector Search in Weaviate → Retrieve Best Recipe → LLM Generates Final Answer
```

### Components
- **Embedding Module:** OpenAI embeddings (1536-d vectors)
- **Vector DB:** Weaviate (Docker), with BYO vectors  
- **Retrieval:** k=1 or k=4 nearest neighbors  
- **Generator:** GPT model using strict, grounding prompt  
- **Baseline LLM:** Generic no-context version for comparison  
- **CLI Interface**  

### Prompt Engineering
RAG is constrained to use:
- A single best-matching recipe
- Only information from the dataset
- No invented ingredients or steps

Baseline LLM intentionally outputs general cooking advice.

## 5. Technical Details

### Tools
- Python 3.10
- OpenAI API
- Weaviate Client v4
- python-dotenv

### Embedding Model
- `text-embedding-3-small`

### LLM
- `gpt-4o-mini` (or similar OpenAI model)

### Vector Store
- Weaviate running locally in Docker
- Schema recreated on startup to ensure consistency

### Dataset ingestion
- Runs automatically on script startup
- Ensures all recipes are embedded and inserted

## 6. Requirements

### Functional
- Retrieve recipes semantically
- Answer using only retrieved recipe
- Provide baseline model answer
- Reject queries outside dataset scope via fallback message

### Non-Functional
- Deterministic RAG output
- Simple, readable CLI UX
- Modular code structure

### Deployment
- Docker installed
- Valid OpenAI API key
- Python environment with dependencies installed

## 7. Limitations

- Dataset limited to 10 recipes
- System cannot synthesize multi-recipe answers
- CLI only—no UI
- Baseline suppression is prompt-based, not capability-based

## 8. Video Demonstration (to be added)

Link to demo video will be inserted here after recording.

## 9. Conclusion

This project demonstrates a clean, modular RAG system using OpenAI embeddings + LLM and Weaviate for vector search. It highlights a clear difference between general LLM answers and grounded RAG answers, making it suitable for academic demonstration or further extension.
