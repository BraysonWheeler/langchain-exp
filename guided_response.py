from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)


def guided_response():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer named {name}"),
        ("user", "{input}")
    ])

    chain = prompt | model

    response = chain.invoke(
        {
            "name": "Jarvis",
            "input": "What is your name?"
        }
    )
    return response


if __name__ == "__main__":
    response = guided_response()
    print(response)
