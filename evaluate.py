import json
import sys
import uuid
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv()

PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
EVAL_SESSION_ID     = f"evaluation_session_{uuid4()}"
MAX_CONTEXTS        = 10
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7


def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": f"eval_{uuid.uuid4().hex}"}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": EVAL_SESSION_ID,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()

    docs = load_document(PDF_PATH)
    add_paper(docs, EVAL_SESSION_ID)

    graph = build_graph(db_path="eval_checkpoints.db")

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
    ]

    test_cases = []
    for pair in pairs:
        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query)
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    results = evaluate(test_cases, metrics)

    summary = []
    for test_result in results.test_results:
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_path}.")


if __name__ == "__main__":
    main()
