"""
Best-of-N judge implementation.
"""
from typing import List, Dict, Tuple
import logging

from judges import BaseJudge, BestOfNResult
from dataset import DataSample  
from generator import GeneratedResponse


class BestOfNJudge(BaseJudge):
    """Judge that selects the best response from N candidates."""
    
    def __init__(self, config, judge_manager):
        super().__init__(config, judge_manager)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Math-Verify for verification
        self.math_grader = None
        try:
            from math_grader import MathVerifyGrader
            self.math_grader = MathVerifyGrader(strict=False, float_rounding=6)
            self.logger.info("Math-Verify grader initialized for Best-of-N verification")
        except ImportError:
            self.logger.warning("Math-Verify not available for verification")
            self.math_grader = None
    
    def evaluate(self, sample: DataSample, responses: List[GeneratedResponse]) -> BestOfNResult:
        """Evaluate all responses and select the best one using LLM judge + Math-Verify."""
        if not responses:
            raise ValueError("No responses provided for evaluation")
        
        # Cache Math-Verify results to avoid redundant calls
        response_correctness = []
        response_results: Dict[int, Tuple[float, str]] = {}
        
        if self.math_grader is not None:
            for i, response in enumerate(responses):
                try:
                    score, reasoning = self.math_grader.grade_response(
                        response=response.text,
                        ground_truth=sample.answer,
                        question=sample.question
                    )
                    response_results[i] = (score, reasoning)  # Cache for later reuse
                    response_correctness.append(score == 10.0)
                except Exception as e:
                    self.logger.warning(f"Error checking response correctness: {e}")
                    response_results[i] = (0.0, f"Error: {e}")
                    response_correctness.append(False)
        else:
            response_correctness = [False] * len(responses)
        
        pass_at_n = any(response_correctness)
        
        best_idx, reasoning, confidence = self.judge_manager.judge_best_of_n(
            question=sample.question,
            responses=responses
        )
        
        if not (0 <= best_idx < len(responses)):
            self.logger.warning(f"Judge returned invalid index {best_idx}, using 0")
            best_idx = 0
        
        best_response = responses[best_idx]
        
        is_correct = False
        verification_reasoning = ""
        if self.math_grader is not None and best_idx in response_results:
            score, verification_reasoning = response_results[best_idx]
            is_correct = (score == 10.0)
        elif self.math_grader is None:
            verification_reasoning = "Math-Verify not available"
        else:
            verification_reasoning = "Math-Verify result not cached"
        
        result = BestOfNResult(
            sample_id=sample.sample_id,
            method="best_of_n",
            best_response_idx=best_idx,
            best_response=best_response,
            judge_reasoning=reasoning,
            confidence=confidence,
            all_responses=responses,
            is_correct=is_correct,
            verification_reasoning=verification_reasoning,
            pass_at_n=pass_at_n,
            response_correctness=response_correctness,
            metadata={
                'num_responses': len(responses),
                'ground_truth': sample.answer,
                'question': sample.question,
                'judge_model': self.config.model.name,
                'judge_temperature': self.config.judge.temperature,
            }
        )
        
        if self.config.verbose:
            print(f"Best-of-N: Selected response {best_idx} with confidence {confidence:.2f}")
            print(f"  Pass@N: {'Yes ✓' if pass_at_n else 'No ✗'}")
        
        return result
