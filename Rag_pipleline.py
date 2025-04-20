import os
import json
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(page_title="📝 File Q&A with RAG", layout="centered")
st.title("📝 File Q&A with RAG (Retrieval-Augmented Generation)")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF file to chat with", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("🔄 Processing your PDF..."):

        # Save uploaded file to disk temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split documents
        loader = PyPDFLoader(file_path=temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents=docs)

        # Create embeddings
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=openai_api_key
        )

        # Ingest into Qdrant
        vector_store = QdrantVectorStore.from_documents(
            documents=[],
            url="http://localhost:6333",
            collection_name="learning_langchain",
            embedding=embedder
        )
        vector_store.add_documents(documents=split_docs)

        # Set retriever
        retriever = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="learning_langchain",
            embedding=embedder
        )

        st.success("✅ PDF processed and stored in vector database!")

        # Chat UI
        user_query = st.text_input("💬 Ask a question based on the PDF")
        if user_query:
            with st.spinner("🤖 Generating answer..."):

                # Fetch relevant chunks
                relevant_chunks = retriever.similarity_search(query=user_query)
                context = "\n".join([doc.page_content for doc in relevant_chunks])

                # System prompt
                system_prompt = f"""
                You are a helpful assistant. Use the context below to answer the question.

                Context:
                {context}
                """

                # Chat with OpenAI
                result = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ]
                )

                # Display answer
                st.markdown("### 🧠 Answer:")
                st.write(result.choices[0].message.content)

        # Clean up temporary file
        os.remove(temp_file_path)
