# src/generator.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def call_gemini_text(prompt, model="gemini-1.5-flash", temperature=0.0):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model)
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature
        }
    )
    return resp.text if hasattr(resp, "text") else str(resp)


def build_rag_prompt(question, retrieved_docs):
    instruction = "Answer the question using ONLY the provided context."
    context = "\n\n---\n\n".join(retrieved_docs)
    prompt = f"""{instruction}

CONTEXT:
{context}

QUESTION:
{question}

If the answer is not contained in the context, say "I don't know"."""
    return prompt
