import os
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

QUERY = os.getenv("QUERY")

class RAG():
    def __init__(self):
        self.vector_db = None

    def set_vector_db(self): 
        self.vector_db = Chroma(
            collection_name="example_collection",
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="/tmp/testcontainer/chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
        return self.vector_db

    def embed_documents_and_get_vectordb(self):
        # 1️ Load & Split Text Data
        raw_text = """
        Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.
        It groups containers into logical units called pods, which can scale and be managed more efficiently.
        """

        raw_text_2 = "the cat is crazyy"
        raw_text_3 = "the car is very fast"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        docs = text_splitter.create_documents([raw_text, raw_text_2, raw_text_3])# splits the texts in documents with config defined on text_splitter

        # 2️ Generate Embeddings & Store in Vector DB (Chroma)
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma.from_documents(docs, embedding=embedding_function, persist_directory="./chroma_db")
        return self.vector_db

    def query_vector_db(self, query): #this is not used as part of the RAG implementation. Its an example of querying the VDB alone
        results = self.vector_db.similarity_search(
            query,
            k=2,
        )
        print(f"\nVECTORDB QUESTION - only as example, shows example context for:>>{query}<<")
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]")
        return results
        
    def app_main(self, query):  
        # 3️ Setup OpenAI LLM 
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

        # 4️ LangChain RAG Pipeline + Query the RAG System

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        retrieval_qa_chat_prompt = '''Answer any use questions based solely on the context below:
        <context>
        {context}
        </context>
        '''
        prompt = ChatPromptTemplate.from_messages(
            [("system", retrieval_qa_chat_prompt), ("human", "{input}")]
        )
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        chain_result = rag_chain.invoke({"input": query})
        
        print("*****************************\n\n\nCHAIN QUESTION:")
        print(chain_result['input'])

        print("\nCHAIN CONTEX:\n")
        for i, doc in enumerate(chain_result["context"]):
            print(f"CONTEX {i}  <====================")
            print(doc.page_content)

        print("\n\nCHAIN RESULT:\n")
        print(chain_result['answer'])
        return chain_result['answer']
