import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import weaviate

from config.settings import GROQ_API_KEY



# LOAD DOCUMENTS

def load_documents():
    base = "data"

    pdf_path = os.path.join(base, "pdf", "Whole spine mri report.pdf")
    doc1_path = os.path.join(base, "docs", "Essential Nutrients for Maize Cultivation.docx")
    doc2_path = os.path.join(base, "docs", "Final Fertilizers.docx")
    json_path = os.path.join(base, "json", "restaraunt.json")

    all_docs = []

    if os.path.exists(pdf_path):
        all_docs.extend(PyPDFLoader(pdf_path).load())

    if os.path.exists(doc1_path):
        all_docs.extend(Docx2txtLoader(doc1_path).load())

    if os.path.exists(doc2_path):
        all_docs.extend(Docx2txtLoader(doc2_path).load())

    if os.path.exists(json_path):
        all_docs.extend(TextLoader(json_path).load())

    return all_docs



# Chunk or split

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)




# Embeddings + weviate

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    client = weaviate.connect_to_embedded()

    vectorstore = WeaviateVectorStore.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        index_name="RAG_Demo"
    )

    return vectorstore



#  Retrieval
def retrieve_context(vectorstore, query, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)




# LLM Generation (GROQ)

template = """
You are a helpful AI assistant. Answer ONLY using the information from the context.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=GROQ_API_KEY)



# Full Rag Pipeline

def run_rag(question):
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents into chunks...")
    chunks = split_docs(docs)

    print("Creating vector database...")
    vectorstore = create_vector_db(chunks)

    print("Retrieving relevant context...")
    retrieved_docs = retrieve_context(vectorstore, question)

    context = "\n\n".join([d.page_content for d in retrieved_docs])

    print("Generating answer using Groq LLM...")
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return response.content
