from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

def get_embedding(text, is_query=False):
    text = str(text)

    if is_query:
        text = "Represent this sentence for searching relevant passages: " + text

    embedding = model.encode(   # convert text to vector 
        text, 
        normalize_embeddings=True     # normalize vector length for cosine similarity
    )
    return embedding.tolist()