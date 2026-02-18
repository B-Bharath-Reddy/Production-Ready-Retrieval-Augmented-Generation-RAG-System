
"""
Evaluation Orchestrator
-----------------------
Phase 3: Verification.
Runs the RAG pipeline against a test set and evaluates the results using Groq as a judge.
Metrics:
1. Faithfulness: Is the answer derived *only* from the context?
2. Relevancy: Does the answer actually address the user's question?
"""

import json
import os
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from conf.config import cfg
from retrieval.query_rewriter import QueryRewriter
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import Generator
import pandas as pd

# Evaluation Prompts
# "LLM-as-a-Judge" prompts to score output
FAITHFULNESS_PROMPT = """
You are a judge. Evaluate if the provided ANSWER is based ONLY on the provided CONTEXT.
Ignore outside knowledge.

Context: {context}
Answer: {answer}

Score from 0 (Hallucination) to 1 (Faithful). Return ONLY the number.
"""

RELEVANCY_PROMPT = """
You are a judge. Evaluate if the provided ANSWER actually answers the QUESTION.
Ignore if the answer is factually correct, focus on relevance.

Question: {question}
Answer: {answer}

Score from 0 (Irrelevant) to 1 (Relevant). Return ONLY the number.
"""

class Evaluator:
    """
    Orchestrates the LLM-as-a-Judge evaluation.
    
    Why this is useful:
      - Manual testing is slow and subjective.
      - Automated metrics allow us to benchmark improvements (e.g. changing chunk size or LLM).
      - Faithfulness measures hallucinations; Relevancy measures utility.
    """

    def __init__(self, test_set_path: str):
        """
        Initialize the Evaluator and the RAG components.

        Args:
            test_set_path (str): Path to the JSON test set.

        Why this is useful:
          - Sets up the same pipeline components used in production to ensure the test is valid.
          - Initializes a separate LLM instance to act as the "Judge".
        """
        self.test_set_path = test_set_path
        
        # Initialize RAG Pipeline components
        self.rewriter = QueryRewriter()
        self.retriever = Retriever()
        self.reranker = Reranker(model_name=cfg.reranking.model_name)
        self.generator = Generator()
        
        # Initialize Judge LLM (using same Groq model for simplicity)
        # Temp=0.0 to ensure consistent judging
        self.judge_llm = ChatGroq(
            temperature=0,
            model_name=cfg.generation.model_name,
            groq_api_key=cfg.groq_api_key
        )

    def load_test_set(self) -> List[Dict]:
        """
        Loads the test questions.

        Returns:
            List[Dict]: List of {"question": "...", "ground_truth": "..."} pairs.

        Why this is useful:
          - Reads the ground-truth dataset from `test_set.json`.
        """
        with open(self.test_set_path, "r") as f:
            return json.load(f)

    def measure_faithfulness(self, context: str, answer: str) -> float:
        """
        Scores how well the answer sticks to the context.

        Args:
            context (str): The retrieved context text.
            answer (str): The generated answer.

        Returns:
            float: Score between 0.0 and 1.0.

        Why this is useful:
          - Prevents hallucinations. We want the bot to admit ignorance rather than make things up.
        """
        chain = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT) | self.judge_llm | StrOutputParser()
        try:
            score = chain.invoke({"context": context, "answer": answer})
            return float(score.strip())
        except:
            return 0.0

    def measure_relevancy(self, question: str, answer: str) -> float:
        """
        Scores how well the answer addresses the question.

        Args:
            question (str): The user's original question.
            answer (str): The generated answer.

        Returns:
            float: Score between 0.0 and 1.0.

        Why this is useful:
          - Prevents vague or evasive answers. The bot should answer the user's specific intent.
        """
        chain = ChatPromptTemplate.from_template(RELEVANCY_PROMPT) | self.judge_llm | StrOutputParser()
        try:
            score = chain.invoke({"question": question, "answer": answer})
            return float(score.strip())
        except:
            return 0.0

    def run(self):
        """
        Executes the full evaluation loop.

        Why this is useful:
          - Iterates through every test case, runs the pipeline, scores it, and aggregates results.
          - Provides a final report card for the system.
        """
        print("--- Starting Evaluation ---")
        data = self.load_test_set()
        results = []

        for item in data:
            question = item['question']
            print(f"Evaluating: {question}")
            
            # --- 1. Run RAG Pipeline ---
            # Rewrite
            search_query = self.rewriter.rewrite(question)
            
            # Retrieve
            docs = self.retriever.get_relevant_documents(search_query, top_k=cfg.retrieval.top_k)
            
            # Rerank
            if cfg.reranking.enabled:
                docs = self.reranker.rerank(search_query, docs, top_n=cfg.reranking.top_n)
            else:
                docs = docs[:cfg.reranking.top_n]
            
            # Generate
            context_text = "\n\n".join([d.page_content for d in docs])
            answer = self.generator.generate_answer(question, context_text)
            
            # --- 2. Judge Results ---
            faithfulness = self.measure_faithfulness(context_text, answer)
            relevancy = self.measure_relevancy(question, answer)
            
            results.append({
                "question": question,
                "answer": answer,
                "context_size": len(context_text),
                "faithfulness": faithfulness,
                "relevancy": relevancy
            })

        # --- 3. Report Results ---
        df = pd.DataFrame(results)
        print("\n--- Evaluation Results ---")
        print(df[["question", "faithfulness", "relevancy"]])
        
        # Calculate Averages
        print(f"\nAverage Faithfulness: {df['faithfulness'].mean():.2f}")
        print(f"Average Relevancy: {df['relevancy'].mean():.2f}")
        
        # Save to CSV
        df.to_csv("evaluation_report.csv", index=False)
        print("Saved detailed report to evaluation_report.csv")

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_set.json")
    evaluator = Evaluator(test_path)
    evaluator.run()

