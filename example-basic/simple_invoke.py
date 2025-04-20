from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)


def simple_invoke():
    template = """Question: {question} Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    response = chain.invoke("how can langsmith help with testing?")
    return response


if __name__ == "__main__":
    response = simple_invoke()
    print(response)
