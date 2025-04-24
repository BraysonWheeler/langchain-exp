from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def sentence_embeddings(texts: list[str]):
    vectors = embeddings.embed_documents(texts)
    return vectors


def cos_similarity(vector1:list, vector2: list):
    dot_product = 0
    a_magnitude = 0
    b_magnitude = 0

    for i, ival in enumerate(vector1):
        dot_product += vector1[i] * vector2[i]
        a_magnitude += vector1[i] * vector1[i]
        b_magnitude += vector2[i] * vector2[i]

    if (a_magnitude == 0 or b_magnitude == 0):
        return 0

    return dot_product / (a_magnitude * b_magnitude)


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

    score = cos_similarity(vectors[0], vectors[1])
    # Will be Similar
    is_similar(score, 0.8)

    not_similar_texts = [
        "Charlie ran up the hill",
        "Nico Harrison shouldn't have traded luka"
    ]
    vectors = sentence_embeddings(not_similar_texts)
    score = cos_similarity(vectors[0], vectors[1])
    # Will not be Similar
    is_similar(score, 0.8)
