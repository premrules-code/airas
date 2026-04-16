"""Relevance grader for Corrective RAG.

Grades retrieved documents for relevance to the query using Claude Haiku
for speed and cost efficiency.

Grades:
- RELEVANT: Document directly answers or contains key information
- PARTIAL: Document has some useful context but doesn't directly answer
- IRRELEVANT: Document is off-topic or wrong ticker/time period
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

import anthropic

from config.settings import get_settings

logger = logging.getLogger(__name__)

GradeType = Literal["relevant", "partial", "irrelevant"]


@dataclass
class GradeResult:
    """Result of grading a single document."""

    grade: GradeType
    confidence: float  # 0.0 to 1.0
    reason: str
    doc_index: int  # Position in input list

    @property
    def is_relevant(self) -> bool:
        """Consider both 'relevant' and 'partial' as usable."""
        return self.grade in ("relevant", "partial")

    @property
    def relevance_score(self) -> float:
        """Numeric score: relevant=1.0, partial=0.5, irrelevant=0.0"""
        if self.grade == "relevant":
            return 1.0
        elif self.grade == "partial":
            return 0.5
        return 0.0


class RelevanceGrader:
    """Grade retrieved documents for relevance to query.

    Uses Claude Haiku for fast, cheap grading (~$0.00025 per doc).

    Usage:
        grader = RelevanceGrader()

        # Grade single document
        result = grader.grade_document(
            query="What was Apple's revenue?",
            document="Apple reported net sales of $383 billion..."
        )
        print(result.grade)  # "relevant"

        # Grade batch (more efficient)
        results = grader.grade_batch(
            query="What was Apple's revenue?",
            documents=["doc1 text...", "doc2 text...", ...]
        )
    """

    # Default model for grading - uses settings.claude_model
    DEFAULT_MODEL = None  # Will use settings.claude_model

    # Grading prompt template - structured for reliable JSON output
    GRADING_PROMPT = """Grade this document's relevance. Output ONLY valid JSON, no markdown.

Query: {query}

Document: {document}

Grades:
- "relevant": directly answers with specific facts/numbers
- "partial": related but missing key details
- "irrelevant": off-topic or wrong company

Output format: {{"grade":"relevant","confidence":0.9,"reason":"contains the data"}}

Your response:"""

    # Batch grading prompt - structured for reliable JSON output
    BATCH_GRADING_PROMPT = """Grade each document's relevance to this query. Output ONLY valid JSON.

Query: {query}

{documents}

Grade each document as:
- "relevant": directly answers the query with specific facts/numbers
- "partial": related context but missing key details
- "irrelevant": off-topic, wrong company, or useless

Output format (JSON array, no markdown):
[{{"index":0,"grade":"relevant","confidence":0.9,"reason":"contains revenue figures"}},{{"index":1,"grade":"irrelevant","confidence":0.8,"reason":"about different company"}}]

Your response:"""

    def __init__(self, model: Optional[str] = None):
        """Initialize grader.

        Args:
            model: Claude model to use (default: uses settings.claude_model)
        """
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.claude_model

    def grade_document(
        self,
        query: str,
        document: str,
        max_doc_length: int = 1500,
    ) -> GradeResult:
        """Grade a single document for relevance.

        Args:
            query: The search query
            document: The document text to grade
            max_doc_length: Truncate document to this length (saves tokens)

        Returns:
            GradeResult with grade, confidence, and reason
        """
        # Truncate document to save tokens
        doc_text = document[:max_doc_length]
        if len(document) > max_doc_length:
            doc_text += "... [truncated]"

        prompt = self.GRADING_PROMPT.format(query=query, document=doc_text)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            result = self._parse_single_grade(text, doc_index=0)

            logger.debug(f"Graded document: {result.grade} ({result.confidence:.2f})")
            return result

        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            # Default to partial on error (don't throw away potentially useful docs)
            return GradeResult(
                grade="partial",
                confidence=0.5,
                reason=f"Grading error: {e}",
                doc_index=0,
            )

    def grade_batch(
        self,
        query: str,
        documents: List[str],
        max_doc_length: int = 800,
    ) -> List[GradeResult]:
        """Grade multiple documents in a single LLM call.

        Falls back to individual grading if batch parsing fails.

        Args:
            query: The search query
            documents: List of document texts to grade
            max_doc_length: Truncate each document to this length

        Returns:
            List of GradeResult, one per document
        """
        if not documents:
            return []

        # Format documents with indices
        formatted_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc[:max_doc_length]
            if len(doc) > max_doc_length:
                doc_text += "..."
            formatted_docs.append(f"[Document {i}]\n{doc_text}")

        documents_text = "\n\n---\n\n".join(formatted_docs)
        prompt = self.BATCH_GRADING_PROMPT.format(
            query=query, documents=documents_text
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=150 * len(documents),  # ~150 tokens per doc for safety
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            logger.debug(f"Batch grading response: {text[:500]}...")

            results = self._parse_batch_grades(text, num_docs=len(documents))

            # If parsing failed (returned None), fall back to individual grading
            if results is None:
                logger.warning("Batch parsing failed, falling back to individual grading")
                return self._grade_individually(query, documents, max_doc_length)

            relevant_count = sum(1 for r in results if r.grade == "relevant")
            partial_count = sum(1 for r in results if r.grade == "partial")
            irrelevant_count = sum(1 for r in results if r.grade == "irrelevant")
            logger.info(
                f"Batch graded {len(documents)} docs: "
                f"{relevant_count} relevant, {partial_count} partial, {irrelevant_count} irrelevant"
            )

            return results

        except Exception as e:
            logger.warning(f"Batch grading API call failed: {e}")
            # Fall back to individual grading
            return self._grade_individually(query, documents, max_doc_length)

    def _parse_single_grade(self, text: str, doc_index: int) -> GradeResult:
        """Parse single grade response."""
        try:
            # Extract JSON from response
            json_text = self._extract_json(text)
            data = json.loads(json_text)

            grade = data.get("grade", "partial").lower()
            # Normalize grade values
            if grade not in ("relevant", "partial", "irrelevant"):
                if "relev" in grade:
                    grade = "relevant"
                elif "irrelev" in grade:
                    grade = "irrelevant"
                else:
                    grade = "partial"

            return GradeResult(
                grade=grade,
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", ""),
                doc_index=doc_index,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse grade: {e}, text: {text[:100]}")
            # Try to extract grade from text directly as last resort
            text_lower = text.lower()
            if "irrelevant" in text_lower:
                return GradeResult(grade="irrelevant", confidence=0.3, reason="Parsed from text", doc_index=doc_index)
            elif "relevant" in text_lower and "partial" not in text_lower:
                return GradeResult(grade="relevant", confidence=0.3, reason="Parsed from text", doc_index=doc_index)
            return GradeResult(
                grade="partial",
                confidence=0.5,
                reason="Parse error",
                doc_index=doc_index,
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown and other wrappers."""
        # Try to find JSON array in the text
        # First, remove markdown code blocks
        if "```" in text:
            # Extract content between first set of triple backticks
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                # Remove 'json' language identifier if present
                if text.startswith("json"):
                    text = text[4:]
                elif text.startswith("\n"):
                    text = text[1:]

        # Try to find JSON array with regex
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)

        # Try to find JSON object with regex
        obj_match = re.search(r'\{[\s\S]*\}', text)
        if obj_match:
            return obj_match.group(0)

        return text.strip()

    def _parse_batch_grades(self, text: str, num_docs: int) -> List[GradeResult]:
        """Parse batch grade response."""
        try:
            # Extract JSON from response
            json_text = self._extract_json(text)
            logger.debug(f"Extracted JSON: {json_text[:200]}...")

            data = json.loads(json_text)

            # Handle case where response is a single object instead of array
            if isinstance(data, dict):
                data = [data]

            results = []
            for item in data:
                grade = item.get("grade", "partial").lower()
                # Normalize grade values
                if grade not in ("relevant", "partial", "irrelevant"):
                    if "relev" in grade:
                        grade = "relevant"
                    elif "irrelev" in grade:
                        grade = "irrelevant"
                    else:
                        grade = "partial"

                results.append(
                    GradeResult(
                        grade=grade,
                        confidence=float(item.get("confidence", 0.5)),
                        reason=item.get("reason", ""),
                        doc_index=item.get("index", len(results)),
                    )
                )

            # Ensure we have results for all docs
            while len(results) < num_docs:
                results.append(
                    GradeResult(
                        grade="partial",
                        confidence=0.5,
                        reason="Missing from response",
                        doc_index=len(results),
                    )
                )

            return results[:num_docs]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse batch grades: {e}, text: {text[:300]}...")
            # Return None to signal fallback needed
            return None

    def _grade_individually(
        self,
        query: str,
        documents: List[str],
        max_doc_length: int = 800,
    ) -> List[GradeResult]:
        """Grade documents one at a time as fallback."""
        logger.info(f"Falling back to individual grading for {len(documents)} docs")
        results = []
        for i, doc in enumerate(documents):
            result = self.grade_document(query, doc, max_doc_length)
            result.doc_index = i
            results.append(result)
        return results

    def get_usage_estimate(self, num_docs: int, avg_doc_length: int = 500) -> dict:
        """Estimate token usage and cost for grading.

        Args:
            num_docs: Number of documents to grade
            avg_doc_length: Average document length in characters

        Returns:
            Dict with estimated tokens and cost
        """
        # Rough token estimate: 1 token ≈ 4 characters
        tokens_per_doc = avg_doc_length // 4 + 100  # doc + prompt overhead
        output_tokens_per_doc = 50

        total_input = tokens_per_doc * num_docs
        total_output = output_tokens_per_doc * num_docs

        # Haiku pricing
        input_cost = total_input * 0.25 / 1_000_000
        output_cost = total_output * 1.25 / 1_000_000

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost": input_cost + output_cost,
        }
