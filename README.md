

# Production-Ready RAG System (Personal Knowledge Assistant)

## Overview

This project is a complete Retrieval-Augmented Generation (RAG) system built to help organize and query my personal documents. The system integrates multiple data types including:

* Health records (PDF)
* Agriculture notes that I created for maize cultivation (DOCX)
* Restaurant information (JSON)
* Other mixed text formats

The goal is to enable natural-language querying across all these documents using a modern RAG pipeline built with LangChain, Weaviate, HuggingFace embeddings, and the Groq LLaMA model.

The codebase is structured in a modular and production-ready format that can be extended to real-world applications such as enterprise search, knowledge assistants, document question-answering tools, and personal information retrieval.

---

## Data Sources

### Health Record (PDF)

Stored in:

```
data/pdf/Whole spine mri report.pdf
```

Used for querying medical findings, summarizing reports, and extracting abnormalities.

### Agriculture Documents (DOCX)

Stored in:

```
data/docs/Essential Nutrients for Maize Cultivation.docx
data/docs/Final Fertilizers.docx
```

These files contain notes I wrote for maize cultivation. The system can answer domain-specific queries using these documents.

### Restaurant Dataset (JSON)

Stored in:

```
data/json/restaraunt.json
```

Used to test retrieval on structured text and metadata.

These diverse data formats demonstrate the flexibility of the RAG pipeline across different domains.

---

## What the System Can Do

Examples of the types of questions the system can handle:

* Summaries and insights from health reports
* Nutrient and fertilizer recommendations from agriculture documents
* Information extraction from the JSON restaurant dataset
* Combined retrieval across multiple document types

---

## Technology Stack

| Component              | Technology Used        |
| ---------------------- | ---------------------- |
| Vector Database        | Weaviate               |
| Embedding Model        | BAAI/bge-small-en-v1.5 |
| Language Model         | Groq LLaMA 3.1 Instant |
| Framework              | LangChain              |
| File Formats Supported | PDF, DOCX, JSON        |
| Programming Language   | Python                 |

---

## RAG Pipeline

1. Document Loading
   PDFs, DOCX files, and JSON are loaded and converted into LangChain Document objects.

2. Text Chunking
   Documents are split into manageable chunks using RecursiveCharacterTextSplitter to ensure quality retrieval.

3. Embedding
   Each chunk is embedded using the BGE model.

4. Vector Storage
   Embeddings are stored in Weaviate for efficient similarity search.

5. Retrieval
   The system retrieves the most relevant chunks based on semantic similarity.

6. Generation
   Groq LLaMA 3.1 uses the retrieved context and the user’s question to generate the final answer.

---

## Project Structure

```
rag_personal/
│
├── src/
│   ├── main.py
│   ├── rag_pipeline.py
│   └── __init__.py
│
├── config/
│   ├── settings.py
│   └── __init__.py
│
├── data/
│   ├── pdf/
│   ├── docs/
│   └── json/
│
├── requirements.txt
└── .gitignore
```

---

## How to Run

Run the system using:

```
python -m src.main
```

Example queries:

```
What are the essential nutrients required for maize cultivation?
```

```
Summarize the findings in the spine MRI report.
```

```
What information is provided in the restaurant JSON file?
```

---

## Why This Project Is Useful

* Demonstrates a complete RAG implementation using real personal data
* Works with multiple document types
* Shows practical usage of LangChain, Weaviate, and Groq API
* Can be extended into a UI or deployed as a service
* Valuable for demonstrating RAG and real-world document understanding in interviews and portfolios

---

## Author

Bharath Reddy
Data Science and Machine Learning
Hyderabad, India

]
