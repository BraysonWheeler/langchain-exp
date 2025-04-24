from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import (MODEL_VERSION, MODEL_BASE_URL)
from sklearn.decomposition import PCA
import plotly.express as px
import pandas

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def load_documents():
    loader = WebBaseLoader("https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html")
    documents = loader.load()
    return documents


def split_document_into_texts(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.split_text(document.page_content)
    return texts


def embedd_text(text):
    vector = embeddings.embed_documents(text)
    return vector


if __name__ == "__main__":
    documents = load_documents()
    document_texts = [split_document_into_texts(document) for document in documents]
    for document_text in document_texts:
        vector = embedd_text(document_text)
        pca = PCA(n_components=2)
        components = pca.fit_transform(vector)
        data = pca.fit_transform(vector)
        df = pandas.DataFrame(
            {
                "x": data[:, 0],
                "y": data[:, 1],
                "query": document_text,
            }
        )

        sample = df[:df.size]  # Change if you want an actual sample
        sample.to_csv("./sample.csv")
        scatter = px.scatter(sample, x=sample['x'], y=sample['y'])
        scatter.show()
        break
