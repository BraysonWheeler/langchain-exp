from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from constants import (MODEL_VERSION, MODEL_BASE_URL)
import numpy

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def sentence_embeddings(texts: list[str]):
    vectors = embeddings.embed_documents(texts)
    return vectors


def dot_product(vector1: list, vector2: list):
    return numpy.dot(
        vector1,
        vector2
    )


def is_similar(val: float, threshold: float):
    _ = f"Val: {val}, Threshold: {threshold} is "
    if val > threshold:
        print(_+"similar")
    else:
        print(_+"not similar")
    return


if __name__ == "__main__":
    similar_texts = [
        "Charlie ran up the hill",
        "Charlie ran down the hill"
    ]
    vectors = sentence_embeddings(similar_texts)
    dproduct = dot_product(vectors[0], vectors[1])
    # Will be Similar
    is_similar(dproduct, 8.0)

    not_similar_texts = [
        "Charlie ran up the hill",
        "Nico Harrison shouldn't have traded luka"
    ]
    vectors = sentence_embeddings(not_similar_texts)
    dproduct = dot_product(vectors[0], vectors[1])

    # Will not be Similar
    is_similar(dproduct, 8.0)



