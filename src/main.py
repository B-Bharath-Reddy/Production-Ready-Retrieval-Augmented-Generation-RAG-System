from .rag_pipeline import run_rag  

if __name__ == "__main__":
    print("RAG System Ready!")
    user_q = input("Enter your question: ")

    answer = run_rag(user_q)
    print(answer)
