from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

class RAGAgent:
    def __init__(self, json_path="logs/anomalies.json", persist_dir="rag_store"):
        self.json_path = json_path
        self.persist_dir = persist_dir
        self._init_rag_pipeline()

    def _init_rag_pipeline(self):
        # Load JSON documents (assumes an array of objects)
        loader = JSONLoader(file_path=self.json_path, jq_schema=".[]")
        self.docs = loader.load()

        # Create embedding model
        self.embeddings = OpenAIEmbeddings()

        # Load into Chroma vector store
        self.vectorstore = Chroma.from_documents(
            self.docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        # RAG chain (retrieval + LLM)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
            retriever=self.vectorstore.as_retriever()
        )

    def ask(self, question):
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"RAG Error: {str(e)}"
