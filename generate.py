import subprocess

def call_ollama(context, question, model="llama3"):
    prompt = f"""Use the context to answer the question below.
    
Context:
{context}

Question: {question}
Answer:"""

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode()
