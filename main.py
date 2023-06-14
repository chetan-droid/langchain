import openai
from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


loader = DirectoryLoader('docs', glob='**/*.txt')
docs = loader.load()

char_text_split = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

doc_texts = char_text_split.split_documents(docs)
# print("L1", doc_texts)

openai_embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

vec_store = Chroma.from_documents(doc_texts, openai_embeddings)

model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", return_source_documents=True)

question = input("Enter a query: ")
response = model({"query: ": question})
response["source_documents"] = vec_store.documents
print(response)
