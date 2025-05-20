import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

try:
    with open("llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: llm_config.yaml not found. Please create this file with your configuration.")
    exit()
except yaml.YAMLError as e:
    print(f"Error parsing llm_config.yaml: {e}")
    exit()


google_api_key = llm_config.get("google_api_key")
embedding_model_name = llm_config.get("embedding_model_name", "models/embedding-001")
chromadb_persist_directory = llm_config.get("chromadb_path", "chroma_db_gemini")
data_directory = llm_config.get("data_dir", "data")


if not google_api_key:
    print("Error: Google API key not found. Please set GOOGLE_API_KEY environment variable or add 'google_api_key' to llm_config.yaml")
    exit()

try:
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model=embedding_model_name)
except Exception as e:
    print(f"Error initializing GoogleGenerativeAIEmbeddings: {e}")
    print("Please check your Google API key and model name in llm_config.yaml.")
    exit()


def load_and_chunk_documents(data_path=data_directory):
    """Loads documents from a directory and splits them into chunks."""
    all_documents = []
    
    # Add specific loaders for different file types
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.json': JSONLoader,
    }

    if not os.path.exists(data_path):
        print(f"Data directory not found at {data_path}. Please create it and add your documents.")
        return []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in loaders:
                try:
                    loader_cls = loaders[file_extension]
                    # Use a jq schema to extract content and metadata
                    # This schema attempts to get content from 'description' or 'section_desc'
                    jq_schema = '.[] | {page_content: (.description // .section_desc), metadata: .}'
                    loader = loader_cls(file_path, jq_schema=jq_schema , text_content=False)
                    all_documents.extend(loader.load())
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Skipping unsupported file type: {file_path}")


    if not all_documents:
        print("No documents loaded. Cannot create chunks.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def create_vector_db(chunks, persist_directory=chromadb_persist_directory):
    """Creates a Chroma vector database from the document chunks using Gemini embeddings."""
    if not chunks:
        print("No chunks to process. Vector database not created.")
        return None

    print(f"Creating vector database at: {persist_directory}")
    try:
        # Use Chroma.from_documents which handles embedding internally
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        print(f"Vector database created and persisted.")
        return vectordb
    except Exception as e:
        print(f"Error creating or persisting vector database: {e}")
        return None


if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created an empty '{data_directory}' directory. Please add your documents there.")
    else:
        chunks = load_and_chunk_documents()
        if chunks:
            create_vector_db(chunks)