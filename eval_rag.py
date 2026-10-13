import os
from dotenv import load_dotenv

load_dotenv()

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from rag import retriever, run_rag


def run_rag_for_eval(question: str):
    """Helper to collect tokens and retrieved docs from the RAG pipeline."""
    tokens = []
    sources = []
    for event in run_rag(question, []):
        if event.get("type") == "token":
            tokens.append(event.get("data", ""))
        elif event.get("type") == "sources":
            sources = event.get("data", [])

    # Fetch context chunks directly from the vector store for the evaluator to grade
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    return "".join(tokens), contexts


def test_rag_subsidies():
    question = "What subsidies are available for small and marginal farmers?"
    actual_output, retrieval_context = run_rag_for_eval(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    # Threshold 0.7 means a passing grade requires 70% or higher score
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

    assert_test(test_case, [faithfulness_metric, relevancy_metric])