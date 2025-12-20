from .config import Settings
from .rag_app import RagApp

def main() -> None:
    settings = Settings()
    app = RagApp(settings)

    print("\n✅ OpenAI RAG Recipe Assistant ready.")
    print("Ask any cooking question, or type 'exit'.\n")

    try:
        while True:
            q = input("Q: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            answer, _ = app.answer(q, k=3)

            print("\n--- RAG ANSWER ---")
            print(answer)
            print("\n--- NO-RAG ANSWER (BASELINE LLM) ---")
            print(app.llm.answer_without_context(q))
    finally:
        app.db.close()
        print("🔒 Closed Weaviate client.")

if __name__ == "__main__":
    main()
