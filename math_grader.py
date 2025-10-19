"""
Math-Verify based grading for mathematical answer verification.

This module provides deterministic, symbolic math verification as an
alternative to LLM-based judging.
"""

from typing import Tuple, Optional
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig


class MathVerifyGrader:
    """Deterministic math grader using Math-Verify library."""
    
    def __init__(self, strict: bool = False, float_rounding: int = 6):
        """Initialize the Math-Verify grader.
        
        Args:
            strict: Whether to use strict comparison (variables must match exactly)
            float_rounding: Number of decimal places for float comparisons
        """
        self.strict = strict
        self.float_rounding = float_rounding
        
        # Configure extraction for LaTeX and plain expressions
        # Prioritize boxed answers (boxed_match_priority=0 is highest priority)
        self.extraction_config = [
            LatexExtractionConfig(boxed_match_priority=0),  # Prioritize \boxed{} format
            ExprExtractionConfig()
        ]
    
    def grade_response(
        self, 
        response: str, 
        ground_truth: str,
        question: Optional[str] = None
    ) -> Tuple[float, str]:
        """Grade a response against the ground truth.
        
        Args:
            response: The model's response text
            ground_truth: The correct answer
            question: Optional question text (for context in error messages)
        
        Returns:
            Tuple of (score, reasoning) where:
            - score: 10.0 for correct, 0.0 for incorrect
            - reasoning: Explanation of the grading decision
        """
        try:
            # Ensure ground truth has LaTeX delimiters for proper parsing
            if ground_truth and not ground_truth.strip().startswith('$'):
                ground_truth_wrapped = f"${ground_truth}$"
            else:
                ground_truth_wrapped = ground_truth
            
            # Parse both response and ground truth
            pred_parsed = parse(response, extraction_config=self.extraction_config)
            gold_parsed = parse(ground_truth_wrapped, extraction_config=self.extraction_config)
            
            # Check if parsing succeeded
            if not pred_parsed:
                return 0.0, f"Could not extract answer from response: {response[:100]}"
            
            if not gold_parsed:
                return 0.0, f"Could not parse ground truth: {ground_truth}"
            
            # Try to verify - use first successfully parsed expressions
            for pred_expr in pred_parsed:
                # Skip string fallbacks if we have parsed expressions
                if isinstance(pred_expr, str):
                    continue
                    
                for gold_expr in gold_parsed:
                    if isinstance(gold_expr, str):
                        continue
                    
                    try:
                        is_correct = verify(
                            gold_expr,
                            pred_expr,
                            strict=self.strict,
                            float_rounding=self.float_rounding,
                            timeout_seconds=5,
                            raise_on_error=False
                        )
                        
                        if is_correct:
                            return 10.0, f"Correct: {pred_expr} matches {gold_expr}"
                    except Exception as e:
                        # Continue trying other combinations
                        continue
            
            # If we get here, no match was found
            pred_str = str(pred_parsed[0]) if pred_parsed else "None"
            gold_str = str(gold_parsed[0]) if gold_parsed else "None"
            return 0.0, f"Incorrect: {pred_str} does not match {gold_str}"
            
        except Exception as e:
            # Fallback for any unexpected errors
            return 0.0, f"Grading error: {str(e)}"
    
    def grade_batch(
        self,
        responses: list[str],
        ground_truths: list[str],
        questions: Optional[list[str]] = None
    ) -> list[Tuple[float, str]]:
        """Grade a batch of responses.
        
        Args:
            responses: List of model responses
            ground_truths: List of correct answers
            questions: Optional list of questions
        
        Returns:
            List of (score, reasoning) tuples
        """
        if questions is None:
            questions = [None] * len(responses)
        
        results = []
        for response, ground_truth, question in zip(responses, ground_truths, questions):
            result = self.grade_response(response, ground_truth, question)
            results.append(result)
        
        return results


def test_math_grader():
    """Test the Math-Verify grader with example cases."""
    grader = MathVerifyGrader()
    
    test_cases = [
        # (response, ground_truth, expected_score)
        (r"$(r, \theta) = (3, \frac{\pi}{2})$", r"\left( 3, \frac{\pi}{2} \right)", 10.0),
        (r"The answer is $(3, \frac{\pi}{2})$", r"\left( 3, \frac{\pi}{2} \right)", 10.0),
        (r"$(3, \pi)$", r"\left( 3, \frac{\pi}{2} \right)", 0.0),
        (r"The answer is 42", "42", 10.0),
        (r"x = 10", "10", 10.0),
        (r"$\frac{1}{3}$", "0.333333", 0.0),  # Different representations
        (r"0.333333", r"$\frac{1}{3}$", 10.0),  # Should work with tolerance
    ]
    
    print("Testing Math-Verify Grader:")
    print("=" * 70)
    
    for i, (response, ground_truth, expected) in enumerate(test_cases, 1):
        score, reasoning = grader.grade_response(response, ground_truth)
        status = "✓" if score == expected else "✗"
        print(f"\n{status} Test {i}:")
        print(f"  Response: {response[:60]}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Score: {score} (expected: {expected})")
        print(f"  Reasoning: {reasoning}")
    
    print("\n" + "=" * 70)
    print("Testing complete!")


if __name__ == "__main__":
    test_math_grader()

