import os
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load documents from the website
loader = WebBaseLoader(web_paths=["https://www.educosys.com/course/genai"])
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector DB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def print_prompt(prompt_text):
    print("Prompt -", prompt_text)
    return prompt_text

# Load RAG prompt
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI()

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(print_prompt)
    | llm
    | StrOutputParser()
)

# Ask a question
print(rag_chain.invoke("What all projects are covered in the course?"))
