import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)



with open("knowledge.txt", "r") as file:
    document = file.read()

sentences = document.replace("\n", " ").split(". ") 

chunks = []

for sentence in sentences:
    sentence = sentence.strip()

    if sentence:
        chunks.append(sentence)

    
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\nNumber of chunks:", len(chunks))
print("Embedding dimensions:", len(embeddings[0]))


from sentence_transformers.util import cos_sim

question = input("\nAsk a question: ")
question_embedding = model.encode(question)

similarities = cos_sim(question_embedding, embeddings)[0]

top_k = 3

top_indices = similarities.argsort(descending=True)[:top_k]

best_score = similarities[top_indices[0]].item()

threshold = 0.50

if best_score < threshold:
    print("\nNo relevant information found in the knowledge base.")
    exit()

print("\nRelevant chunks:")

for index in top_indices:
    print(f"\nSimilarity: {similarities[index].item():.4f}")
    print(chunks[index])

print("\nQuestion:", question)

context = "\n\n".join(
    chunks[index] for index in top_indices
)


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