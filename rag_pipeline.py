import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

from config import DOC_PATH

# Global QA chain - initialized on first use
qa_chain = None


def load_documents():
    """Load documents from the configured DOC_PATH"""
    # Ensure we're using absolute path
    doc_path = os.path.abspath(DOC_PATH)
    
    loader = DirectoryLoader(
        path=doc_path,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    return loader.load()


def create_vectorstore(docs):
    """Create FAISS vectorstore from documents"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # LOCAL EMBEDDINGS (replaces OpenAI)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def create_qa_chain(vectorstore):
    """Create retrieval QA chain"""
    # LOCAL LLM (Ollama)
    llm = Ollama(model="gemma:2b")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


def initialize_qa():
    """Initialize the QA chain - loads documents and creates vectorstore"""
    global qa_chain
    
    if qa_chain is not None:
        return qa_chain
    
    print("Initializing QA chain...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    
    vectorstore = create_vectorstore(docs)
    print("Vectorstore created")
    
    qa_chain = create_qa_chain(vectorstore)
    print("QA chain ready!")
    
    return qa_chain


def ask_question(query):
    """Ask a question using the RAG pipeline"""
    global qa_chain
    
    if qa_chain is None:
        initialize_qa()
    
    return qa_chain.invoke({"query": query})