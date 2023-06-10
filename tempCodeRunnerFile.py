
# openai_embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# vec_store = Chroma.from_documents(doc_texts, openai_embeddings)

# model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", return_source_documents=True)

# # question = input("Enter a query: ")
# # response = model({"query: ": question})
# # response["source_documents"] = vec_store.documents
# # print(response)
# model.run()