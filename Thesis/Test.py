from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]
sentences = ["car rental"]
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
embeddings = model.encode(sentences)
print(embeddings)