import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from openai import OpenAI
from langchain.schema import Document

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name
        
        # Now use the path with PyPDFLoader
        pdf_reader = PyPDFLoader(tmp_file_path)
        pages = pdf_reader.load()
        for page in pages:
            text += page.page_content + "\n"  # Use actual newline, not "/n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorise(text_chunks):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",api_key = os.environ['OPENAI_API_KEY'])
    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url = "http://localhost:6333",
        collection_name="pdf_data_langchain",
        embedding=embeddings
    )

    vector_store.add_documents(documents=documents)
    return vector_store

def get_conversation_chain(vectorstore, query: str, history: list = []):
    retriever = vectorstore.as_retriever()
    client = OpenAI()

    # Step 1: Retrieve relevant documents
    relevant_chunks = retriever.get_relevant_documents(query)

    # Step 2: Format retrieved chunks into context
    context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])

    # Step 3: Prepare the system prompt
    system_prompt = f"""
    You are a helpful AI Assistant who responds based on the available context.
    
    Context:
    {context_text}
    """

    # Step 4: Prepare message history with chat memory
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous chat history (if any)
    messages.extend(history)
    
    # Add current user query
    messages.append({"role": "user", "content": query})

    # Step 5: Generate response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    answer = response.choices[0].message.content

    # Step 6: Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Extract, chunk, and vectorize the text
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorise(text_chunks)

                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.success("Documents processed successfully!")

    # Handle user question
    if user_question and st.session_state.vectorstore:
        response, updated_history = get_conversation_chain(
            st.session_state.vectorstore, user_question, st.session_state.chat_history
        )
        st.session_state.chat_history = updated_history  # update memory

        # Display chat
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")
if __name__ == '__main__':
    main()