# RAG Web Loader: Website QA Bot using LangChain + Chroma

This is a Retrieval-Augmented Generation (RAG) project that allows you to load **text content from any website**, split it into chunks, embed it using OpenAI, store it in ChromaDB, and ask questions over the content using LangChain's RAG chain.

## 🧠 Features

- Loads content from **any publicly accessible website**
- Splits the web content into overlapping chunks
- Creates vector embeddings using OpenAI
- Stores chunks in Chroma vector DB
- Enables RAG-based question-answering using LangChain and OpenAI

## 🔧 Tech Stack

- **LangChain** for orchestration
- **ChromaDB** for vector storage
- **OpenAI** for embeddings + LLM
- **LangChainHub** for RAG prompt
- **Python-dotenv** for managing secrets

## 🚀 Setup Instructions

1. Clone the repo or download the files.
2. Create a `.env` file (or copy from `.env.example`) and add your OpenAI key:
   ```env
   OPENAI_API_KEY=your-key-here
