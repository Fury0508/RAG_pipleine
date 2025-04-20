
# 📚 Chat with Multiple PDFs using LangChain + OpenAI

An intelligent chatbot that allows you to **upload multiple PDFs** and ask questions about their content. Powered by **LangChain**, **OpenAI (GPT-4 + Embeddings)**, **Qdrant vector store**, and served via a sleek **Streamlit UI**, it retrieves the most relevant document chunks using semantic search and generates accurate responses using GPT-4.

---

## 📸 Overview

This app uses the following architecture:

1. ⬆️ Upload one or more PDF files  
2. ✂️ Text is split into chunks  
3. 🧠 Each chunk is embedded using OpenAI’s `text-embedding-3-large` model  
4. 💾 Chunks are stored in a **Qdrant** vector database  
5. ❓ A question is embedded and used to perform **semantic search**  
6. 📈 The top **ranked** chunks are sent as context to **GPT-4**  
7. 🤖 GPT-4 answers your question based only on the relevant context

---

## 🧰 Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI GPT-4 + Embeddings](https://platform.openai.com/)
- [Qdrant Vector Store](https://qdrant.tech/)
- [Streamlit](https://streamlit.io/)
- Docker (for containerized deployment)

---

## 📦 Requirements

- Python 3.10+
- OpenAI API Key
- Docker (optional, for containerized setup)

---

## 🔐 Environment Variables

Create a `.env` file in the root of the project:

```env
OPENAI_API_KEY=your_openai_api_key


## 📁 Project Structure

.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker setup
├── .env                   # OpenAI API Key (excluded from git)
├── README.md              # Project documentation
└── /screenshots           # (Optional) App UI images


