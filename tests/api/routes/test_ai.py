import pytest
from unittest.mock import patch
from src.api.routes.ai import (
    get_user_details_for_prompt,
    _extract_code_line_evidence_from_submission,
    _normalise_objective_feedback,
)


@pytest.mark.asyncio
class TestAIFunctions:
    """Test AI route helper functions."""

    @patch("src.api.routes.ai.get_user_first_name")
    async def test_get_user_details_for_prompt_with_first_name(self, mock_get_user_first_name):
        """Test get_user_details_for_prompt when user has first name."""
        mock_get_user_first_name.return_value = "John"

        result = await get_user_details_for_prompt("1")

        assert result == "Name: John"
        mock_get_user_first_name.assert_called_once_with("1")

    @patch("src.api.routes.ai.get_user_first_name")
    async def test_get_user_details_for_prompt_no_first_name(self, mock_get_user_first_name):
        """Test get_user_details_for_prompt when user has no first name."""
        mock_get_user_first_name.return_value = None

        result = await get_user_details_for_prompt("1")

        assert result == ""
        mock_get_user_first_name.assert_called_once_with("1")

    @patch("src.api.routes.ai.get_user_first_name")
    async def test_get_user_details_for_prompt_empty_first_name(self, mock_get_user_first_name):
        """Test get_user_details_for_prompt when user has empty first name."""
        mock_get_user_first_name.return_value = ""

        result = await get_user_details_for_prompt("1")

        assert result == ""
        mock_get_user_first_name.assert_called_once_with("1")

    def test_extract_code_line_evidence_from_submission(self):
        submission = """// PYTHON
def solve(values):
    heap = []
    for value in values:
        heap.append(value)
"""

        evidence = _extract_code_line_evidence_from_submission(
            submission,
            "Use a heap and improve logic in solve function.",
        )

        assert len(evidence) > 0
        assert evidence[0]["type"] == "code_line"
        assert evidence[0]["reference"].startswith("L")

    def test_normalise_objective_feedback_incorrect(self):
        normalized = _normalise_objective_feedback(
            feedback_summary="Check the edge case for empty input.",
            is_correct=False,
            submission_text="def solve():\n    pass\n",
            external_evidence=[
                {
                    "type": "test_case",
                    "reference": "tc-1",
                    "description": "Expected [] but got error",
                }
            ],
        )

        assert normalized["overall_score"] == 0
        assert len(normalized["criteria"]) == 1
        assert normalized["criteria"][0]["criterion_name"] == "Correctness"
        assert normalized["criteria"][0]["severity"] == "high"

    def test_objective_score_is_aligned_with_correctness_true(self):
        normalized = _normalise_objective_feedback(
            feedback_summary="Correct overall logic.",
            is_correct=True,
            objective_score=1.0,
        )

        criterion = normalized["criteria"][0]
        assert criterion["score"] >= 3.0
        assert criterion["severity"] in {"low", "medium"}

    def test_objective_score_is_aligned_with_correctness_false(self):
        normalized = _normalise_objective_feedback(
            feedback_summary="Fails required checks.",
            is_correct=False,
            objective_score=4.0,
        )

        criterion = normalized["criteria"][0]
        assert criterion["score"] < 3.0
        assert criterion["severity"] == "high"
