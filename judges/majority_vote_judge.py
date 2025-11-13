"""
Majority Vote Judge - No LLM judging, just answer extraction and voting.
"""
import logging
from typing import List, Dict, Tuple
from collections import Counter

from judges import BaseJudge, MajorityVoteResult
from dataset import DataSample
from generator import GeneratedResponse


class MajorityVoteJudge(BaseJudge):
    """Judge that extracts answers and picks the most common one (no LLM judging)."""
    
    def __init__(self, config, judge_manager=None):
        super().__init__(config, judge_manager)
        self.logger = logging.getLogger(__name__)
        
        self.math_grader = None
        try:
            from math_grader import MathVerifyGrader
            self.math_grader = MathVerifyGrader(strict=False, float_rounding=6)
            self.logger.info("Math-Verify initialized for majority voting")
        except ImportError:
            self.logger.error("Math-Verify required for majority voting but not available!")
            raise RuntimeError("Math-Verify is required for majority voting baseline")
    
    def _extract_answer(self, response_text: str) -> str:
        """Extract the final answer from a response using Math-Verify's parser."""
        try:
            from math_verify import parse
            from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
            
            extraction_config = [
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig()
            ]
            
            parsed = parse(response_text, extraction_config=extraction_config)
            
            if parsed and len(parsed) > 0:
                return str(parsed[0])
            else:
                return ""
        except Exception as e:
            self.logger.warning(f"Failed to extract answer: {e}")
            return ""
    
    def evaluate(self, sample: DataSample, responses: List[GeneratedResponse]) -> MajorityVoteResult:
        """Evaluate by extracting answers and finding the majority vote (no LLM judge)."""
        if not responses:
            raise ValueError("No responses provided for evaluation")
        
        extracted_answers = []
        response_texts = []
        for response in responses:
            answer = self._extract_answer(response.text)
            extracted_answers.append(answer)
            response_texts.append(response.text)
        
        # Cache Math-Verify results to avoid redundant calls
        response_correctness = []
        response_results: Dict[int, Tuple[float, str]] = {}
        
        for i, response_text in enumerate(response_texts):
            try:
                score, reasoning = self.math_grader.grade_response(
                    response=response_text,
                    ground_truth=sample.answer,
                    question=sample.question
                )
                response_results[i] = (score, reasoning)  # Cache for later reuse
                response_correctness.append(score == 10.0)
            except Exception as e:
                self.logger.warning(f"Error checking response correctness: {e}")
                response_results[i] = (0.0, f"Error: {e}")
                response_correctness.append(False)
        
        pass_at_n = any(response_correctness)
        
        non_empty_answers = [a for a in extracted_answers if a]
        
        if not non_empty_answers:
            result = MajorityVoteResult(
                sample_id=sample.sample_id,
                method="majority_vote",
                extracted_answers=extracted_answers,
                answer_counts={},
                majority_answer="",
                majority_count=0,
                responses=responses,
                is_correct=False,
                verification_reasoning="No answers could be extracted from any response",
                pass_at_n=pass_at_n,
                response_correctness=response_correctness,
                metadata={
                    'num_responses': len(responses),
                    'ground_truth': sample.answer,
                    'question': sample.question,
                }
            )
            return result
        
        answer_counter = Counter(non_empty_answers)
        majority_answer, majority_count = answer_counter.most_common(1)[0]
        
        majority_response_text = None
        for i, extracted in enumerate(extracted_answers):
            if extracted == majority_answer:
                majority_response_text = response_texts[i]
                break
        
        is_correct = False
        verification_reasoning = ""
        
        if majority_response_text:
            majority_idx = None
            for i, extracted in enumerate(extracted_answers):
                if extracted == majority_answer:
                    majority_idx = i
                    break
            
            if majority_idx is not None and majority_idx in response_results:
                score, verification_reasoning = response_results[majority_idx]
                is_correct = (score == 10.0)
            else:
                verification_reasoning = "Math-Verify result not cached for majority answer"
        else:
            verification_reasoning = "Could not find response text for majority answer"
        
        result = MajorityVoteResult(
            sample_id=sample.sample_id,
            method="majority_vote",
            extracted_answers=extracted_answers,
            answer_counts=dict(answer_counter),
            majority_answer=majority_answer,
            majority_count=majority_count,
            responses=responses,
            is_correct=is_correct,
            verification_reasoning=verification_reasoning,
            pass_at_n=pass_at_n,
            response_correctness=response_correctness,
            metadata={
                'num_responses': len(responses),
                'ground_truth': sample.answer,
                'question': sample.question,
                'num_extracted': len(non_empty_answers),
                'num_unique_answers': len(answer_counter),
            }
        )
        
        if self.config.verbose:
            print(f"Majority Vote: {majority_answer} ({majority_count}/{len(responses)} votes)")
            print(f"  Result: {'Correct ✓' if is_correct else 'Incorrect ✗'}")
            print(f"  Pass@N: {'Yes ✓' if pass_at_n else 'No ✗'}")
        
        return result

