import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field, PrivateAttr
import requests
import os

# Load environment variables
load_dotenv()

# Custom Perplexity LLM wrapper
from pydantic import Field, PrivateAttr

class PerplexityLLM(LLM):
    api_key: Optional[str] = Field(default=None)
    model: str = "sonar"
    temperature: float = 0.0
    _endpoint: str = PrivateAttr(default="https://api.perplexity.ai/chat/completions")

    def __init__(self, **data):
        super().__init__(**data)
        if self.api_key is None:
            self.api_key = os.getenv("PPLX_API_KEY")
    @property
    def _llm_type(self) -> str:
        return "perplexity"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = requests.post(self._endpoint, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()
        # Typical response format: {'choices': [{'message': {'content': 'text here'}}]}
        return output["choices"][0]["message"]["content"] if "choices" in output else output.get("text", "")
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vectorstore):
    perplexity_llm = PerplexityLLM(api_key=os.getenv("PPLX_API_KEY"), model="sonar")
    chain = ConversationalRetrievalChain.from_llm(perplexity_llm, vectorstore.as_retriever())
    return chain

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please upload and process PDFs first.")
        return
    vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    # Use the conversational retrieval chain which will use the vectorstore's retriever internally
    conv_chain = get_conversational_chain(vectorstore)
    response = conv_chain({"question": user_question, "chat_history": []}, return_only_outputs=True)
    st.write("Reply:", response["answer"])


def main():
    st.set_page_config(page_title="Chat with Multiple PDFs - Perplexity AI")
    st.header("Chat with Multiple PDFs using Perplexity API")

    user_question = st.text_input("Ask a question about your uploaded PDFs:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs or len(pdf_docs) == 0:
                st.warning("Please upload one or more PDF files before processing.")
            else:
                with st.spinner("Processing PDFs and creating FAISS index..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip() == "":
                        st.warning("No readable text found in uploaded PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("PPLX_API_KEY")
    if api_key:
        os.environ["PPLX_API_KEY"] = api_key
        main()
    else:
        st.error("Perplexity API Key not set. Please add PPLX_API_KEY to your .env file.")
