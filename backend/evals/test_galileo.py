"""Galileo AI evaluation tests for AIRAS.

Tests hallucination detection, groundedness, PII detection, and context relevance.

Run with:
    pytest evals/test_galileo.py -v
    pytest evals/test_galileo.py -v -k "groundedness"  # Run only groundedness tests
    pytest evals/test_galileo.py -v --tb=short  # Shorter traceback
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.metrics.galileo_metrics import GalileoEvaluator, run_galileo_evals
from src.utils.galileo_setup import is_galileo_available


# Skip all tests if Galileo is not available
pytestmark = pytest.mark.skipif(
    not is_galileo_available(),
    reason="Galileo API key not configured"
)


@pytest.fixture
def galileo_eval_set():
    """Load Galileo evaluation dataset."""
    eval_path = Path(__file__).parent / "datasets" / "galileo_eval_set.json"
    with open(eval_path) as f:
        return json.load(f)


@pytest.fixture
def evaluator():
    """Create Galileo evaluator instance."""
    return GalileoEvaluator(groundedness_threshold=0.7)


class TestGroundedness:
    """Test groundedness/hallucination detection."""

    def test_grounded_answer_scores_high(self, evaluator, galileo_eval_set):
        """Verify that answers grounded in context score high."""
        query = galileo_eval_set["queries"][0]  # AAPL revenue query

        result = evaluator.evaluate_groundedness(
            query_id=query["id"],
            query=query["query"],
            answer=query["expected_answer"],
            context=query["context"],
            expected_groundedness=query["eval_criteria"]["expected_groundedness"],
        )

        print(f"\nQuery: {query['query']}")
        print(f"Groundedness score: {result.groundedness_score:.2f}")
        print(f"Expected: >= {result.expected_groundedness}")
        print(f"Passed: {result.passed}")

        # Grounded answer should score reasonably high
        assert result.groundedness_score >= 0.5, (
            f"Expected groundedness >= 0.5, got {result.groundedness_score}"
        )

    def test_hallucinated_answer_scores_low(self, evaluator, galileo_eval_set):
        """Verify that hallucinated answers score low."""
        hal_test = galileo_eval_set["hallucination_tests"][0]

        result = evaluator.evaluate_groundedness(
            query_id=hal_test["id"],
            query=hal_test["query"],
            answer=hal_test["bad_answer"],
            context=hal_test["context"],
            expected_groundedness=0.9,
        )

        print(f"\nHallucination test: {hal_test['description']}")
        print(f"Bad answer: {hal_test['bad_answer'][:100]}...")
        print(f"Groundedness score: {result.groundedness_score:.2f}")
        print(f"Flagged claims: {result.flagged_claims}")

        # Hallucinated answer should score lower
        assert result.groundedness_score < 0.7, (
            f"Expected hallucinated answer to score < 0.7, got {result.groundedness_score}"
        )

    def test_groundedness_batch(self, evaluator, galileo_eval_set):
        """Test groundedness across all queries."""
        results = []

        for query in galileo_eval_set["queries"]:
            if "context" not in query:
                continue

            result = evaluator.evaluate_groundedness(
                query_id=query["id"],
                query=query["query"],
                answer=query["expected_answer"],
                context=query["context"],
                expected_groundedness=query.get("eval_criteria", {}).get(
                    "expected_groundedness", 0.8
                ),
            )
            results.append(result)

        # Calculate stats
        scores = [r.groundedness_score for r in results]
        passed = [r for r in results if r.passed]

        print(f"\n{'='*60}")
        print("GROUNDEDNESS BATCH RESULTS")
        print(f"{'='*60}")
        print(f"Total queries: {len(results)}")
        print(f"Mean score: {sum(scores)/len(scores):.3f}")
        print(f"Pass rate: {len(passed)/len(results):.1%}")
        print(f"{'='*60}")

        # At least 50% should pass
        assert len(passed) / len(results) >= 0.5, (
            f"Expected >= 50% pass rate, got {len(passed)/len(results):.1%}"
        )


class TestHallucinationDetection:
    """Test ability to distinguish hallucinated vs grounded answers."""

    def test_detects_fabricated_numbers(self, evaluator, galileo_eval_set):
        """Test detection of fabricated financial numbers."""
        hal_test = galileo_eval_set["hallucination_tests"][0]  # Revenue test

        result = evaluator.evaluate_hallucination_detection(
            bad_answer=hal_test["bad_answer"],
            good_answer=hal_test["good_answer"],
            context=hal_test["context"],
        )

        print(f"\nTest: {hal_test['description']}")
        print(f"Bad answer score: {result['bad_answer_score']:.2f}")
        print(f"Good answer score: {result['good_answer_score']:.2f}")
        print(f"Score difference: {result['score_difference']:.2f}")
        print(f"Detection success: {result['detection_success']}")

        assert result["detection_success"], (
            "Failed to detect hallucinated numbers"
        )

    def test_detects_invented_products(self, evaluator, galileo_eval_set):
        """Test detection of invented product announcements."""
        hal_test = galileo_eval_set["hallucination_tests"][1]  # Product test

        result = evaluator.evaluate_hallucination_detection(
            bad_answer=hal_test["bad_answer"],
            good_answer=hal_test["good_answer"],
            context=hal_test["context"],
        )

        print(f"\nTest: {hal_test['description']}")
        print(f"Detection success: {result['detection_success']}")
        print(f"Flagged claims: {result['bad_flagged_claims']}")

        assert result["detection_success"], (
            "Failed to detect invented products"
        )

    def test_all_hallucination_tests(self, evaluator, galileo_eval_set):
        """Run all hallucination detection tests."""
        results = []

        for hal_test in galileo_eval_set["hallucination_tests"]:
            result = evaluator.evaluate_hallucination_detection(
                bad_answer=hal_test["bad_answer"],
                good_answer=hal_test["good_answer"],
                context=hal_test["context"],
            )
            result["test_id"] = hal_test["id"]
            results.append(result)

        detected = sum(1 for r in results if r["detection_success"])

        print(f"\n{'='*60}")
        print("HALLUCINATION DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Total tests: {len(results)}")
        print(f"Detected: {detected}")
        print(f"Detection rate: {detected/len(results):.1%}")
        print(f"{'='*60}")

        # Should detect at least 50% of hallucinations
        assert detected / len(results) >= 0.5, (
            f"Expected >= 50% detection rate, got {detected/len(results):.1%}"
        )


class TestPIIDetection:
    """Test PII detection capabilities."""

    def test_detects_email(self, evaluator):
        """Test email detection."""
        result = evaluator.evaluate_pii(
            text_id="email_test",
            text="Contact john.doe@example.com for more information.",
            expected_has_pii=True,
        )

        print(f"\nPII detected: {result.has_pii}")
        print(f"PII types: {result.pii_types}")

        # Should detect email
        assert result.has_pii, "Failed to detect email PII"

    def test_detects_phone(self, evaluator):
        """Test phone number detection."""
        result = evaluator.evaluate_pii(
            text_id="phone_test",
            text="Call us at 555-123-4567 or (800) 555-1234.",
            expected_has_pii=True,
        )

        print(f"\nPII detected: {result.has_pii}")
        print(f"PII types: {result.pii_types}")

        assert result.has_pii, "Failed to detect phone PII"

    def test_no_false_positive_on_clean_text(self, evaluator):
        """Test that clean financial text doesn't trigger false positives."""
        result = evaluator.evaluate_pii(
            text_id="clean_test",
            text="Apple reported revenue of $383.3 billion for FY2023. The company's gross margin was 44.1%.",
            expected_has_pii=False,
        )

        print(f"\nPII detected: {result.has_pii}")
        print(f"PII types: {result.pii_types}")

        # Should not detect PII in clean financial text
        assert not result.has_pii, f"False positive: detected {result.pii_types}"

    def test_pii_batch(self, evaluator, galileo_eval_set):
        """Run all PII tests from dataset."""
        results = []

        for pii_test in galileo_eval_set["pii_tests"]:
            # Test text with PII
            with_pii = evaluator.evaluate_pii(
                text_id=pii_test["id"] + "_with",
                text=pii_test["text_with_pii"],
                expected_has_pii=True,
            )
            results.append(with_pii)

            # Test text without PII
            without_pii = evaluator.evaluate_pii(
                text_id=pii_test["id"] + "_without",
                text=pii_test["text_without_pii"],
                expected_has_pii=False,
            )
            results.append(without_pii)

        passed = sum(1 for r in results if r.passed)

        print(f"\n{'='*60}")
        print("PII DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Accuracy: {passed/len(results):.1%}")
        print(f"{'='*60}")


class TestFullEvaluation:
    """Test full evaluation pipeline."""

    def test_full_eval_batch(self, galileo_eval_set):
        """Run complete evaluation on dataset."""
        evaluator = GalileoEvaluator()
        results = evaluator.evaluate_batch(galileo_eval_set)

        summary = results["summary"]

        print(f"\n{'='*60}")
        print("FULL GALILEO EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries: {summary['total_queries']}")
        print(f"Groundedness mean: {summary['groundedness']['mean']:.3f}")
        print(f"Groundedness pass rate: {summary['groundedness']['pass_rate']:.1%}")
        print(f"Hallucination detection rate: {summary['hallucination_detection']['rate']:.1%}")
        print(f"PII detection accuracy: {summary['pii_detection']['accuracy']:.1%}")
        print(f"{'='*60}")

        # Save results
        self._save_results(results)

        # Basic sanity checks
        assert summary["total_queries"] > 0

    def test_run_from_file(self):
        """Test convenience function to run evals from file."""
        eval_path = Path(__file__).parent / "datasets" / "galileo_eval_set.json"
        results = run_galileo_evals(str(eval_path))

        assert "summary" in results
        assert "results" in results

    def _save_results(self, results: dict):
        """Save evaluation results to file."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"galileo_eval_{timestamp}.json"

        # Convert dataclass results to dicts for JSON serialization
        serializable_results = {
            "timestamp": datetime.now().isoformat(),
            "summary": results["summary"],
            "groundedness_results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "score": r.groundedness_score,
                    "passed": r.passed,
                    "flagged_claims": r.flagged_claims,
                }
                for r in results["results"]["groundedness"]
            ],
            "hallucination_results": results["results"]["hallucination_detection"],
            "pii_results": [
                {
                    "text_id": r.text_id,
                    "has_pii": r.has_pii,
                    "pii_types": r.pii_types,
                    "passed": r.passed,
                }
                for r in results["results"]["pii"]
            ],
        }

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


# CLI entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
