import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def generate(query: str, chunks: list[str]) -> str:
    if not chunks:
        return "I could not find relevant information in the uploaded documents."

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()