import openai
import os
from dotenv import load_dotenv
import langchain
from langchain.document_loaders import DirectoryLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_document(document: str):
    response = openai.Completion.create(
        engine="davinci",
        prompt=document,
        max_tokens=0,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response['choices'][0]['text']

loader = DirectoryLoader('docs', glob= '**/*.txt')


document = loader.load()
embedded_document = embed_document(document)
docs = [embedded_document[i:i+len(embedded_document)//3] for i in range(0, len(embedded_document), len(embedded_document)//3)]

def get_relevant_doc(question: str):
    response = openai.Completion.create(
        engine="davinci",
        prompt=question,
        max_tokens=0,
        n=1,
        stop=None,
        temperature=0.5,
    )
    embedded_question = response['choices'][0]['text']
    min_distance = float('inf')
    relevant_doc = None
    for doc in docs:
        distance = langchain.distance(embedded_question, doc)
        if distance < min_distance:
            min_distance = distance
            relevant_doc = doc
    return relevant_doc

def answer_question(question: str):
    relevant_doc = get_relevant_doc(question)
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"{relevant_doc}\n\nQ: {question}\nA:",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response['choices'][0]['text']

question = input("Enter a question: ")
answer = answer_question(question)
print(answer)


