import textwrap
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class OpenAIChatClient:
    model: str
    max_context_chars: int = 8000

    def __post_init__(self) -> None:
        self.client = OpenAI()

    def rag_answer(self, context: str, question: str) -> str:
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]

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

        user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

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
                )},
                {"role": "user", "content": question},
            ],
        )
        return resp.choices[0].message.content.strip()
