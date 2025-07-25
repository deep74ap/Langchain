from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
import numpy as np
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07" , dimension = 32)

queries = [
    "The quick brown fox jumps over the lazy dog.",
    "Data structures and algorithms are essential for problem solving.",
    "Artificial Intelligence is shaping the future of technology.",
    "Python is a versatile language used in web, data, and AI.",
    "Always test your code with edge cases."
]

doc_emb = embeddings.embed_documents(queries)

user_text = "How AI is shaping the future"

user_emb = embeddings.embed_query(user_text);

score = cosine_similarity([user_emb] , doc_emb)

index, scores =  sorted(list(enumerate(score[0])),key=lambda x : x[1])[-1]
print(f'''User query : "{user_text}" 
      has max similarity with : "{queries[index]}"
      having cosine similarity : {scores}''')
