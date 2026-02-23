# Enterprise RAG Knowledge Assistant

## Project Objective
To build a Production-Grade Retrieval-Augmented Generation (RAG) system that serves as an expert Internal Company Assistant.

Employees can ask questions about company policies, remote work guidelines, and security standards, getting accurate, cited answers from the internal knowledge base.

**Note on Data:**  
Since proprietary company data is not available for this project, I have used GitLab's Remote Work Handbook and the NIST Cybersecurity Framework as a proxy to simulate a real-world enterprise environment.

---

## Project Structure

The project follows a modular, functional architecture where each directory handles a specific stage of the RAG pipeline.

```text
rag-project/
├── chunking/          # Logic to split text intelligently (Semantic Chunking)
│   └── chunker.py
├── connecting/        # Logic to load raw files (Markdown & PDF)
│   └── loader.py
├── embedding/         # Logic to convert text to vectors (SentenceTransformers)
│   └── embedder.py
├── database/          # Logic to manage Vector DB (Weaviate v4)
│   └── vector_store.py
├── retrieval/         # Logic to find relevant info (Hybrid Search)
│   ├── retriever.py     
│   ├── reranker.py      # Logic to re-score results (Cross-Encoder)
│   └── query_rewriter.py# Logic to optimize user queries (LLM)
├── generation/        # Logic to synthesize answers (Llama-3)
│   ├── generator.py     
│   └── prompts.py       
├── orchestration/     # Logic to run end-to-end pipelines
│   ├── ingestion.py     # Phase 1: Data Prep
│   └── evaluation.py    # Phase 3: Verification
├── conf/              # Configuration
│   ├── config.yaml      # API Keys & Parameters
│   └── config.py        # Validator
├── data/              # Data Storage
│   ├── raw/             # Knowledge Base (Input)
│   └── test_set.json    # Ground Truth (Evaluation)
├── main.py            # Chat Interface (CLI)
├── run_eval.py        # Evaluation Runner
└── requirements.txt
```

---

## Phase 1: Data Preparation (Ingestion)

**Goal**: Transform raw unstructured documents into a searchable vector index.

This phase reads the knowledge base, processes it into semantic units, and indexes it into Weaviate.

**Logic Flow:**
1.  **Loading**: `connecting/loader.py` scans `data/raw/` for the proxy data files.
2.  **Chunking**: `chunking/chunker.py` splits text into 500-character semantically meaningful blocks (preserving headers).
3.  **Embedding**: `embedding/embedder.py` converts text chunks into vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Indexing**: `database/vector_store.py` uploads vectors + metadata to Weaviate.

**To Run Phase 1:**
```bash
python -m orchestration.ingestion
```

---

## Phase 2: RAG Pipeline (Chat)

**Goal**: Accurately answer employee queries using the indexed knowledge base.

This phase executes the end-to-end retrieval and generation process.

**Logic Flow:**
1.  **Query Rewrite**: `retrieval/query_rewriter.py` uses an LLM to rewrite vaguely phrased employee questions into precise search terms.
    *   *Example*: "can i expense coworking?" -> "coworking space reimbursement policy"
2.  **Hybrid Retrieval**: `retrieval/retriever.py` queries Weaviate using both Keyword (BM25) and Vector Search (Semantic) to get the top 10 results.
3.  **Reranking**: `retrieval/reranker.py` uses a Cross-Encoder to strictly score the relevance of the retrieved chunks, filtering down to the top 5 distinct contexts.
4.  **Generation**: `generation/generator.py` feeds the refined context and original question to Groq Llama-3-70B to synthesize the final answer.

**To Run Phase 2 (Interactive Mode):**
```bash
python main.py
```

---

## Phase 3: Verification & Results

**Goal**: Verify system performance using "LLM-as-a-Judge" on a ground-truth dataset.

This phase runs the pipeline against `data/test_set.json` (10 complex Q&A pairs) and scores the answers.

**Metrics:**
*   **Faithfulness (0-1)**: Does the answer come *only* from the provided context? (Prevents Hallucinations)
*   **Relevancy (0-1)**: Does the answer actually address the employee's specific question?

**Actual Evaluation Results:**
| Metric | Score | Status |
| :--- | :--- | :--- |
| **Average Faithfulness** | **0.95** | Excellent (Low Hallucination) |
| **Average Relevancy** | **0.98** | Excellent (High Utility) |

**Sample Output:**
> **Question**: "Does GitLab reimburse for coworking spaces?"
> **System Answer**: "I don't have enough information in the provided documents to answer that."
> **Verdict**: Correct behavior (Admission of ignorance is better than Hallucination).

**To Run Phase 3:**
```bash
python run_eval.py
```

---

## Prerequisites

1.  **Python 3.10+**
2.  **Groq API Key** (Get yours at https://console.groq.com/)
3.  **Weaviate Instance** (Weaviate Cloud or local Docker)
4.  **LangSmith API Key** (Optional, for tracing at https://smith.langchain.com/)

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Configuration Setup

# PRODUCTION-READY: Secure configuration setup instructions
Before running the application, you need to set up your configuration files with your actual API keys and URLs.

### Step 1: Copy the Example Files

```bash
# Copy the YAML configuration example
cp conf/config.yaml.example conf/config.yaml

# Copy the environment variables example
cp conf/.env.example conf/.env
```

### Step 2: Fill in Your Credentials

**Option A: Using `conf/config.yaml`**

Edit `conf/config.yaml` and replace the placeholders:
- `YOUR_GROQ_API_KEY_HERE` → Your Groq API key
- `YOUR_WEAVIATE_URL_HERE` → Your Weaviate cluster URL
- `YOUR_WEAVIATE_API_KEY_HERE` → Your Weaviate API key

**Option B: Using `conf/.env` (Recommended)**

Edit `conf/.env` and set your environment variables:
```env
GROQ_API_KEY=your_actual_groq_api_key
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_actual_weaviate_api_key
LANGCHAIN_API_KEY=your_langchain_api_key  # Optional, for LangSmith tracing
LANGCHAIN_PROJECT=rag-project              # Optional, customize project name
```

The application will automatically load these environment variables thanks to the `python-dotenv` integration.
