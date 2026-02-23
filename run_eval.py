# PRODUCTION-READY: LangSmith tracing environment variables
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "YOUR_LANGCHAIN_API_KEY_HERE")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "rag-project")

"""
Evaluation Runner Entry Point
-----------------------------
Phase 3: Verification.
Helper script to launch the evaluation suite.
"""

# PRODUCTION-READY: Standard logging setup
import logging
from orchestration.evaluation import Evaluator

# PRODUCTION-READY: Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Checks for test data and launches the Evaluator.
    """
    # Define path to test set relative to this script
    test_path = os.path.join("data", "test_set.json")
    
    if not os.path.exists(test_path):
        logger.error("Test set not found at data/test_set.json. Please generate it first.")
        return

    # Initialize and run
    evaluator = Evaluator(test_path)
    evaluator.run()

if __name__ == "__main__":
    main()