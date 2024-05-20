import chromadb

chroma_client = chromadb.PersistentClient(path="chroma")
import chromadb.utils.embedding_functions as embedding_functions

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(
    name="my_collection", embedding_function=ollama_ef
)

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
    ids=["id1", "id2"],
)

results = collection.query(
    query_texts=[
        "This is a query document about florida"
    ],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)

print(results)
