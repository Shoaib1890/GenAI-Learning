import os
from dotenv import load_dotenv
from groq import Groq
import chromadb

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="company_knowledge"
)



with open("knowledge.txt", "r") as file:
    document = file.read()

sentences = document.replace("\n", " ").split(". ") 

chunks = []

for sentence in sentences:
    sentence = sentence.strip()

    if sentence:
        chunks.append(sentence)

collection.add(
    documents=chunks,
    ids=[str(i) for i in range(len(chunks))]
)

print("Chunks added to ChromaDB:", len(chunks))

    
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\nNumber of chunks:", len(chunks))
print("Embedding dimensions:", len(embeddings[0]))



question = input("\nAsk a question: ")

results = collection.query(
    query_texts=[question],
    n_results=3
)
 

print("\nRelevant chunks:")

retrieved_chunks = results["documents"][0]

for i, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)

print("\nQuestion:", question)

context = "\n\n".join(retrieved_chunks)


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": """
            Answer the user's question using only the provided context.
            If the answer is not present in the context, say you don't know.
            """
        },
        {
            "role": "user",
            "content": f"""
            CONTEXT:
            {context}

            QUESTION:
            {question}
            """
        }
    ]
)

answer = response.choices[0].message.content

print("\nLLM Answer:")
print(answer)