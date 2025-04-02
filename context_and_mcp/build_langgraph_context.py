

import re, os
import tiktoken

from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import SKLearnVectorStore

def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)
        
    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    
    # Target the main article content for LangGraph documentation 
    main_content = soup.find("article", class_="md-content__inner")
    
    # If found, use that, otherwise fall back to the whole document
    content = main_content.get_text() if main_content else soup.text
    
    # Clean up whitespace
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    
    return content

def load_langgraph_docs():
    """
    Load LangGraph documentation from the official website.
    
    This function:
    1. Uses RecursiveUrlLoader to fetch pages from the LangGraph website
    2. Counts the total documents and tokens loaded
    
    Returns:
        list: A list of Document objects containing the loaded content
        list: A list of tokens per document
    """
    print("Loading LangGraph documentation...")

    # Load the documentation 
    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",  
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
    ] 

    docs = []
    for url in urls:

        loader = RecursiveUrlLoader(
            url,
            max_depth=5,
            extractor=bs4_extractor,
        )

        # Load documents using lazy loading (memory efficient)
        docs_lazy = loader.lazy_load()

        # Load documents and track URLs
        for d in docs_lazy:
            docs.append(d)

    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    print("\nLoaded URLs:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")
    
    # Count total tokens in documents
    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        total_tokens += count_tokens(doc.page_content)
        tokens_per_doc.append(count_tokens(doc.page_content))
    print(f"Total tokens in loaded documents: {total_tokens}")
    
    return docs, tokens_per_doc

def split_documents(documents):
    """
    Split documents into smaller chunks for improved retrieval.
    
    This function:
    1. Uses RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
    2. Ensures chunks are appropriately sized for embedding and retrieval
    3. Counts the resulting chunks and their total tokens
    
    Args:
        documents (list): List of Document objects to split
        
    Returns:
        list: A list of split Document objects
    """
    print("Splitting documents...")
    
    # Initialize text splitter using tiktoken for accurate token counting
    # chunk_size=8,000 creates relatively large chunks for comprehensive context
    # chunk_overlap=500 ensures continuity between chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,  
        chunk_overlap=500  
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Created {len(split_docs)} chunks from documents.")
    
    # Count total tokens in split documents
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    
    print(f"Total tokens in split documents: {total_tokens}")
    
    return split_docs

def create_vectorstore(splits):
    """
    Create a vector store from document chunks using SKLearnVectorStore.
    
    This function:
    1. Initializes an embedding model to convert text into vector representations
    2. Creates a vector store from the document chunks
    
    Args:
        splits (list): List of split Document objects to embed
        
    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    print("Creating SKLearnVectorStore...")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create vector store from documents using SKLearn
    persist_path = os.getcwd()+"/sklearn_vectorstore.parquet"
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path   ,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")
    
    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)

    return vectorstore

# Load the documents
documents, tokens_per_doc = load_langgraph_docs()

# Save the documents to a file
save_llms_full(documents)

# Split the documents
split_docs = split_documents(documents)

# Create the vector store
vectorstore = create_vectorstore(split_docs)