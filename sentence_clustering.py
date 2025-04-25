from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from sklearn.decomposition import PCA
from constants import (MODEL_VERSION, MODEL_BASE_URL)
import plotly.express as px
import pandas
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
# embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)
embeddings = SentenceTransformer('all-MiniLM-L6-v2') # Sentence transformers embedding looked more accurate than ollama, -> boston ground transpo documents groups together.


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

            
def cluster(embeddings):
    kmeans_model = KMeans(n_clusters=2, n_init='auto', random_state=0)
    classes = kmeans_model.fit_predict(embeddings).tolist()
    return (list(map(str,classes)))


def main():
    documents = load_documents()
    document_embeddings = embeddings.encode(documents)
    pca = PCA(n_components=2)
    data = pca.fit_transform(document_embeddings)
    clusters = cluster(document_embeddings)

    df = pandas.DataFrame(
        {
            "x": data[:, 0],
            "y": data[:, 1],
            "cluster":clusters,
            "text":documents
        }
    )

    scatter = px.scatter(df, x=df['x'], y=df['y'], text="text", color="cluster")
    scatter.show()

if __name__ == "__main__":
    main()
      
