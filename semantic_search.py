from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from sklearn.decomposition import PCA
from constants import (MODEL_VERSION, MODEL_BASE_URL)
import plotly.express as px
import pandas
import numpy
model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def load_documents():
    return [
        "which airlines fly from boston to washington dc via other cities",
        "show me the airlines that fly between toronto and denver",
        "show me round trip first class tickets from new york to miami",
        "i'd like the lowest fare from denver to pittsburgh",
        "show me a list of ground transportation at boston airport",
        "show me boston ground transportation",
        "of all airlines which airline has the most arrivals in atlanta",
        "what ground transportation is available in boston",
        "i would like your rates between atlanta and boston on september third",
        "which airlines fly between boston and pittsburgh"
    ]



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
    # print(_)
    if val > threshold:
        # print(_+"similar")
        return True
    else:
        # print(_+"not similar")
        return False

def closest(documents, query_embedding, document_embeddings):
    _ = 0.0
    document = ""
    for i in range(len(document_embeddings)):
        score = cos_similarity(query_embedding, document_embeddings[i])
        if score > _:
            _ = score
            document = documents[i]
    return _, document

            





def main():
    query = "What are the bus routes in Boston?"
    query_embedding = embeddings.embed_query(query)
    documents = load_documents()
    document_embeddings = embeddings.embed_documents(documents)

    # Closest Document
    score, document = closest(documents, query_embedding, document_embeddings)
    print(f"Closest score: {score} , Document: \n {document}")

    # Create Plot
    documents.append(query)
    document_embeddings.append(query_embedding)
    pca = PCA(n_components=2)
    data = pca.fit_transform(document_embeddings)
    df = pandas.DataFrame(
        {
            "x": data[:, 0],
            "y": data[:, 1],
            "query": documents,
        }
    )

    scatter = px.scatter(df, x=df['x'], y=df['y'], text="query")
    scatter.show()

if __name__ == "__main__":
    main()
      
