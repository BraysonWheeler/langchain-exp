import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from constants import (MODEL_VERSION, MODEL_BASE_URL)

model = OllamaLLM(model=MODEL_VERSION, base_url=MODEL_BASE_URL)
def create_classification_data(text, label):
    formatted_data = {
        "text": text,
        "label": label
    }
    return formatted_data


df = pd.read_csv('./atis_subset.csv', names=['query','intent'])


df_train, df_test = train_test_split(df, test_size=200, random_state=21)

# with open("data.jsonl", 'w+') as file:
#     for row in df_train.itertuples():
#         formatted_data = create_classification_data(row.query, row.intent)
#         file.write(json.dumps(formatted_data) + '\n')
#     file.close()
#     print("Done")

with open("test.jsonl", 'w+') as file:
    for row in df_test.itertuples():
        formatted_data = create_classification_data(row.query, row.intent)
        file.write(json.dumps(formatted_data) + '\n')
    file.close()
    print("Done")


prompt = ChatPromptTemplate.from_template(
    """
        Assign the input to one of the following categories:
        'atis_flight', 'atis_airfare', 'atis_ground_service', 'atis_flight_time', 'atis_airline', 'atis_quantity', 'atis_abbreviation', 'atis_aircraft'.
        Output should just be the category you've assigned it exactly, nothing more.
        Input: {input}
    """)

chain = prompt | model
_ = []
for i, ival in enumerate(df_test["query"]):
    response = chain.invoke(ival)
    print(response)
    _.append(create_classification_data(ival, response))
    break


with open("data-test.jsonl", 'w+') as file:
    for i in _:
        file.write(json.dumps(i) + '\n')