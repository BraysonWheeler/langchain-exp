from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def create_vector(texts):
    vector = FAISS.from_texts(texts, embeddings)
    return vector


def sentence_embeddings():
    texts = [
        "Charlie ran up the hill",
        "Charlie ran down the hill"
    ]
    vector = create_vector(texts)
    response = vector.similarity_search_with_relevance_scores("Charlie", score_threshold=0.1)
    return response


if __name__ == "__main__":
    response = sentence_embeddings()
    print(response)
