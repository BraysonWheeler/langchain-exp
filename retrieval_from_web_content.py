from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_VERSION,  base_url=MODEL_BASE_URL)


def load_web_content():
    loader = WebBaseLoader("https://docs.smith.langchain.com")
    docs = loader.load()
    return docs


def create_vector(docs):
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    return vector


if __name__ == "__main__":
    docs = load_web_content()
    vector = create_vector(docs)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the following question based only on the provided context:
            <context>
                {context}
            </context>
            Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])
