
"""
Evaluation Runner Entry Point
-----------------------------
Phase 3: Verification.
Helper script to launch the evaluation suite.
"""

from orchestration.evaluation import Evaluator
import os

def main():
    """
    Checks for test data and launches the Evaluator.
    """
    # Define path to test set relative to this script
    test_path = os.path.join("data", "test_set.json")
    
    if not os.path.exists(test_path):
        print("Test set not found at data/test_set.json. Please generate it first.")
        return

    # Initialize and run
    evaluator = Evaluator(test_path)
    evaluator.run()

if __name__ == "__main__":
    main()
