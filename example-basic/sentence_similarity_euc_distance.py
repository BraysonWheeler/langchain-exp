from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from constants import (MODEL_VERSION, MODEL_BASE_URL)
from math import sqrt
import numpy


model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def sentence_embeddings(texts: list[str]):
    vectors = embeddings.embed_documents(texts)
    return vectors


def euclidean_distance(vector1: list, vector2: list):
    _ = []
    for i, ival in enumerate(vector1):
        _.append((vector1[i] - vector2[i])**2)

    distance = sqrt(sum(_))
    return distance


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
    distance = euclidean_distance(vectors[0], vectors[1])
    cos = numpy.cos(distance)

    # Will be Similar
    is_similar(cos, 0.8)

    not_similar_texts = [
        "Charlie ran up the hill",
        "Nico Harrison shouldn't have traded luka"
    ]

    vectors = sentence_embeddings(not_similar_texts)
    distance = euclidean_distance(vectors[0], vectors[1])
    cos = numpy.cos(distance)

    # Will not be Similar
    is_similar(cos, 0.8)
