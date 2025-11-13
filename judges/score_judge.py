"""
Score-based judge implementation.
"""
from typing import List, Dict, Tuple
import logging

from judges import BaseJudge, ScoreBasedResult
from dataset import DataSample
from generator import GeneratedResponse


class ScoreBasedJudge(BaseJudge):
    """Judge that assigns individual scores to each response."""
    
    def __init__(self, config, judge_manager):
        super().__init__(config, judge_manager)
        self.logger = logging.getLogger(__name__)
        
        self.math_grader = None
        try:
            from math_grader import MathVerifyGrader
            self.math_grader = MathVerifyGrader(strict=False, float_rounding=6)
            self.logger.info("Math-Verify grader initialized for verification (LLM will do scoring)")
        except ImportError:
            self.logger.warning("Math-Verify not available for verification")
            self.math_grader = None
    
    def evaluate(self, sample: DataSample, responses: List[GeneratedResponse]) -> ScoreBasedResult:
        """Evaluate each response individually and assign LLM judge scores (0-10)."""
        if not responses:
            raise ValueError("No responses provided for evaluation")
        
        # Cache Math-Verify results to avoid redundant calls
        response_correctness = []
        response_results: Dict[int, Tuple[float, str]] = {}
        
        if self.math_grader is not None:
            for i, response in enumerate(responses):
                try:
                    verify_score, reasoning = self.math_grader.grade_response(
                        response=response.text,
                        ground_truth=sample.answer,
                        question=sample.question
                    )
                    response_results[i] = (verify_score, reasoning)  # Cache for later reuse
                    response_correctness.append(verify_score == 10.0)
                except Exception as e:
                    self.logger.warning(f"Error checking response correctness: {e}")
                    response_results[i] = (0.0, f"Error: {e}")
                    response_correctness.append(False)
        else:
            response_correctness = [False] * len(responses)
        
        pass_at_n = any(response_correctness)
        
        scores = []
        reasoning_list = []
        
        for i, response in enumerate(responses):
            try:
                score, reasoning = self.judge_manager.judge_score_based(
                    question=sample.question,
                    response=response
                )
                
                scores.append(score)
                reasoning_list.append(reasoning)
                
                if self.config.verbose:
                    print(f"Response {i} [LLM Judge]: Score {score:.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error scoring response {i} for sample {sample.sample_id}: {e}")
                scores.append(0.0)
                reasoning_list.append(f"Error during evaluation: {e}")
        
        is_correct = False
        verification_reasoning = ""
        if scores and self.math_grader is not None:
            try:
                best_idx = scores.index(max(scores))
                if best_idx in response_results:
                    verify_score, verification_reasoning = response_results[best_idx]
                    is_correct = (verify_score == 10.0)
                else:
                    verification_reasoning = "Math-Verify result not cached"
                
                if self.config.verbose:
                    print(f"Best response (idx {best_idx}) [Math-Verify]: {'Correct' if is_correct else 'Incorrect'}")
            except Exception as e:
                self.logger.error(f"Error verifying best response for sample {sample.sample_id}: {e}")
                verification_reasoning = f"Verification error: {e}"
        elif not self.math_grader:
            verification_reasoning = "Math-Verify not available"
        
        result = ScoreBasedResult(
            sample_id=sample.sample_id,
            method="score_based",
            scores=scores,
            reasoning=reasoning_list,
            responses=responses,
            average_score=0.0,
            best_score=0.0,
            worst_score=0.0,
            best_response_idx=0,
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
            print(f"Score-based: Avg {result.average_score:.2f}, Best {result.best_score:.2f}, Worst {result.worst_score:.2f}")
            print(f"  Pass@N: {'Yes ✓' if pass_at_n else 'No ✗'}")
        
        return result

