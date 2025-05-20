from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import yaml

CHROMA_PATH = "chroma_db"

def load_config(config_path="llm_config.yaml"):
    """Loads the LLM configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_rag_chain():
    """Creates and returns a RAG chain instance."""
    config = load_config()
    api_key = config.get("google_api_key")
    if not api_key:
        raise ValueError("google_api_key not found in llm_config.yaml")

    embedding_model_name = config.get("embedding_model", "models/embedding-001")
    llm_model_name = config.get("llm_model", "gemini-1.5-flash-latest")

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=api_key)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever_k = config.get("retriever", {}).get("k", 4)
    retriever = db.as_retriever(search_kwargs={"k": retriever_k})

    template = """Answer the question based only on the following context:
    {context}
    Never mention base on the input or based on provided context.
    If the context does not contain the answer, say "I don't know".Be sure about the answer.The answer should be short and concise, yet informative.
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    rag_chain = get_rag_chain()
    print("RAG application is ready. Type 'quit' to exit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'quit':
            break
        if query:
            response = rag_chain.invoke(query)
            print("\nResponse:")
            print(response)
            print("-" * 20)

if __name__ == "__main__":
    main()
