import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

genai.configure(api_key=os.getenv("Google_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store_gemini(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_gemini")

def get_vector_store_groq(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_groq")

def get_conversational_chain_gemini():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_groq_response(prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

def user_input_gemini(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index_gemini", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain_gemini()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Gemini Response: ", response["output_text"])

def user_input_groq(user_question):
    embeddings = HuggingFaceEmbeddings()
    new_db = FAISS.load_local("faiss_index_groq", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:
    {context}

    Question:
    {user_question}

    Answer:
    """
    response = get_groq_response(prompt)
    st.write("Groq Response: ", response)

def main():
    st.set_page_config("Chat PDF", page_icon='public/pdf.png')
    st.header("Chat with Multiple PDFs using  GeminiLlama")

    tab1, tab2 = st.tabs(["Gemini", "Groq"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store_gemini(text_chunks)
                get_vector_store_groq(text_chunks)
                st.success("Done")

    with tab1:
        st.subheader("Gemini")
        user_question_gemini = st.text_input("Ask a Question from the PDF Files (Gemini)")
        if user_question_gemini:
            user_input_gemini(user_question_gemini)

    with tab2:
        st.subheader("Groq")
        user_question_groq = st.text_input("Ask a Question from the PDF Files (Groq)")
        if user_question_groq:
            user_input_groq(user_question_groq)

if __name__ == "__main__":
    main()