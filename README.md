# Smart Lawyer RAG Project

**Smart Lawyer** is an intelligent legal assistant powered by Retrieval-Augmented Generation (RAG). It is designed to provide factually grounded, context-rich answers based on authoritative Indian legal documents, including the **Indian Penal Code (IPC)**, **Code of Criminal Procedure (CRPC)**, and **Code of Civil Procedure (CPC)**. Using vector-based retrieval through **ChromaDB** and advanced language models, Smart Lawyer bridges the gap between complex legal text and accessible legal assistance.


## Key Features

* **Legal Domain Focused**: Tailored to Indian laws (IPC, CRPC, CPC)
* **RAG Pipeline**: Combines the strength of semantic search and language generation
* **ChromaDB Integration**: Fast, efficient vector search on legal texts
* **Flexible Input Support**: Supports `.json`, `.pdf`, and `.txt` legal document inputs
* **Configurable**: All parameters managed via `llm_config.yaml`
* **Embeddings-based Retrieval**: Reduces hallucination and ensures factual grounding
* **Modular Scripts**: Clean architecture for data processing, embedding, querying, and application logic



## 🔹 How It Works

1. **Document Ingestion**:

   * Legal documents are extracted using the `pdf_text_extract.py` script (optional if JSON is used).

2. **Text Chunking**:

   * Raw text is split into overlapping chunks (default: 1000 tokens with 200 overlap) to respect LLM context window limits.

3. **Embedding Generation**:

   * Each chunk is embedded using an embedding model.

4. **Vector Database Storage**:

   * Embeddings and corresponding text chunks are stored in **ChromaDB** for efficient similarity search.

5. **Query Processing**:

   * A user query is embedded and compared against stored vectors.

6. **Contextual Retrieval**:

   * Most relevant legal text chunks are retrieved.

7. **LLM Augmentation & Generation**:

   * The retrieved chunks and the user query are fed to a language model (e.g., Gemini, GPT) to generate an answer grounded in legal context.


## Project Setup

### 1.Environment Setup

```bash
python -m ensurepip --upgrade
python -m venv venv
```

#### Activate the Virtual Environment:

* **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```
* **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```


## Data Requirements

Ensure the following legal datasets are present in the `data/` directory:

* `ipc.json`
* `crpc.json`
* `cpc.json`

> If you have `.pdf` or `.txt` files instead, use the `pdf_text_extract.py` script and update `create_vector_db.py` accordingly.



## Configuration

Edit the `llm_config.yaml` to match your environment:

```yaml
data_paths:
  - data/cpc.json
  - data/crpc.json
  - data/ipc.json
chroma_db_path: ./chroma_db
chunk_size: 1000
chunk_overlap: 200
google_api_key: "your-gemini-api-key"
# Add other model-specific configs here
```


## Preprocessing (Create Vector DB)

To create the ChromaDB with embeddings:

```bash
python data/python_script/create_vector_db.py
```

Ensure:

* Virtual environment is active
* `llm_config.yaml` is updated
* JSON legal files are in `data/`

This will read, chunk, embed, and store all legal text into ChromaDB.

## Running the Application

Depending on your preferred interface:

### Option 1: Web App

```bash
python app.py
```

### Option 2: Command Line

```bash
python query_rag.py
```

Follow prompts to enter legal questions and receive RAG-powered responses backed by legal precedent.

## Future Enhancements (Ideas)

* Add PDF UI Upload Support
* Integrate with LangChain for Tooling
* Create a mobile interface (Flutter recommended)
* Add alert systems for legal updates
* Improve summarization & explainability features for legal text

## Credits

Built with love by a team passionate about making law more accessible using AI. Special thanks to open-source contributors of ChromaDB, Gemini/GPT APIs, and legal text parsers.

## Quick Troubleshooting

* **Import errors?** Re-check `requirements.txt` installation.
* **Empty vector DB?** Ensure correct file paths and valid JSON structure.
* **No response or hallucinations?** Check if the embedding model and LLM are correctly integrated and API keys are valid.

## Disclaimer

This tool is meant for **educational and research purposes only**. It does not constitute legal advice. Always consult a qualified legal professional for real-world cases.

Stay curious, build responsibly, and empower access to justice.