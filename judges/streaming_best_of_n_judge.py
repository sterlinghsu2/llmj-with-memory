"""
Streaming Best-of-N judge with trajectory history.
"""
from typing import List, Dict, Any
import logging

from judges import BaseJudge, BestOfNResult
from dataset import DataSample  
from generator import GeneratedResponse
from prompt_templates import format_streaming_best_of_n_prompt


class StreamingBestOfNJudge(BaseJudge):
    """Judge that selects the best response with access to previous judgment history."""
    
    def __init__(self, config, judge_manager):
        super().__init__(config, judge_manager)
        self.logger = logging.getLogger(__name__)
        
        # Initialize trajectory history
        self.trajectory: List[Dict[str, Any]] = []
        self.max_history_tokens = config.streaming_max_history_tokens
        
        # Initialize Math-Verify for verification
        self.math_grader = None
        try:
            from math_grader import MathVerifyGrader
            self.math_grader = MathVerifyGrader(strict=False, float_rounding=6)
            self.logger.info("Math-Verify grader initialized for Streaming Best-of-N")
        except ImportError:
            self.logger.warning("Math-Verify not available for verification")
            self.math_grader = None
    
    def evaluate(self, sample: DataSample, responses: List[GeneratedResponse]) -> BestOfNResult:
        """Evaluate all responses with trajectory history and select the best one."""
        if not responses:
            raise ValueError("No responses provided for evaluation")
        
        # Cache Math-Verify results
        response_correctness = []
        response_results: Dict[int, tuple] = {}
        
        if self.math_grader is not None:
            for i, response in enumerate(responses):
                try:
                    score, reasoning = self.math_grader.grade_response(
                        response=response.text,
                        ground_truth=sample.answer,
                        question=sample.question
                    )
                    response_results[i] = (score, reasoning)
                    response_correctness.append(score == 10.0)
                except Exception as e:
                    self.logger.warning(f"Error checking response correctness: {e}")
                    response_results[i] = (0.0, f"Error: {e}")
                    response_correctness.append(False)
        else:
            response_correctness = [False] * len(responses)
        
        pass_at_n = any(response_correctness)
        
        # Get trajectory history formatted as text (also calculates num_included)
        trajectory_text, num_included, tokens_used = self._format_trajectory_for_prompt_with_stats()
        
        # Use streaming judge with history
        best_idx, reasoning, confidence = self._judge_with_history(
            sample=sample,
            responses=responses,
            trajectory_text=trajectory_text
        )
        
        if not (0 <= best_idx < len(responses)):
            self.logger.warning(f"Judge returned invalid index {best_idx}, using 0")
            best_idx = 0
        
        best_response = responses[best_idx]
        
        # Verify selected response
        is_correct = False
        verification_reasoning = ""
        if self.math_grader is not None and best_idx in response_results:
            score, verification_reasoning = response_results[best_idx]
            is_correct = (score == 10.0)
        elif self.math_grader is None:
            verification_reasoning = "Math-Verify not available"
        else:
            verification_reasoning = "Math-Verify result not cached"
        
        # Add to trajectory history with full context (all responses)
        self._add_to_trajectory(
            sample_id=sample.sample_id,
            question=sample.question,
            all_responses=[r.text for r in responses],
            best_idx=best_idx,
            reasoning=reasoning,
            confidence=confidence,
            is_correct=is_correct
        )
        
        result = BestOfNResult(
            sample_id=sample.sample_id,
            method="streaming_best_of_n",
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
                'streaming_mode': True,
                'trajectory_total_responses': len(self.trajectory) - 1,  # Before adding current
                'trajectory_included_responses': num_included,
                'history_tokens_used': tokens_used,
                'history_tokens_budget': self.max_history_tokens,
                'history_utilization_pct': round(tokens_used / self.max_history_tokens * 100, 1) if self.max_history_tokens > 0 else 0,
            }
        )
        
        if self.config.verbose:
            print(f"Streaming Best-of-N: Selected response {best_idx} with confidence {confidence:.2f}")
            print(f"  Trajectory: {num_included}/{len(self.trajectory) - 1} responses included ({tokens_used} tokens), Pass@N: {'Yes ✓' if pass_at_n else 'No ✗'}")
        
        return result
    
    def _judge_with_history(
        self, 
        sample: DataSample,
        responses: List[GeneratedResponse],
        trajectory_text: str
    ) -> tuple:
        """Call judge with trajectory history."""
        prompt = format_streaming_best_of_n_prompt(
            question=sample.question,
            responses=responses,
            trajectory_history=trajectory_text,
            tokenizer=self.judge_manager.tokenizer
        )
        
        try:
            outputs = self.judge_manager.model.generate([prompt], self.judge_manager.sampling_params)
            judgment = outputs[0].outputs[0].text.strip()
            best_idx, reasoning, confidence = self.judge_manager._parse_best_of_n_judgment(judgment)
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                # Context overflow - skip this sample
                prompt_tokens = len(self.judge_manager.tokenizer.encode(prompt))
                print(f"[CONTEXT OVERFLOW] Sample {sample.sample_id}: prompt is {prompt_tokens} tokens, exceeds max_model_len. Skipping sample.")
                raise  # Re-raise to let the pipeline handle it
            else:
                raise
        
        return best_idx, reasoning, confidence
    
    def _add_to_trajectory(
        self,
        sample_id: str,
        question: str,
        all_responses: list,
        best_idx: int,
        reasoning: str,
        confidence: float,
        is_correct: bool
    ) -> None:
        """Add a judgment to the trajectory history with full context."""
        entry = {
            'sample_id': sample_id,
            'question': question,
            'all_responses': all_responses,
            'best_response_idx': best_idx,
            'reasoning': reasoning,
            'confidence': confidence,
            'is_correct': is_correct,
        }
        self.trajectory.append(entry)
    
    def _format_trajectory_for_prompt_with_stats(self) -> tuple[str, int, int]:
        """Format trajectory history and return (text, num_included, tokens_used)."""
        if not self.trajectory:
            return "", 0, 0
        
        # Simple truncation: take most recent entries that fit in token budget
        formatted_entries = []
        total_tokens = 0
        num_included = 0
        
        # Go backwards through trajectory (most recent first)
        for entry in reversed(self.trajectory):
            entry_text = self._format_single_entry(entry)
            entry_tokens = self._count_tokens(entry_text)
            
            if total_tokens + entry_tokens <= self.max_history_tokens:
                formatted_entries.insert(0, entry_text)  # Insert at beginning to maintain order
                total_tokens += entry_tokens
                num_included += 1
            else:
                break  # Stop if we exceed budget
        
        # Log trajectory usage
        total_in_history = len(self.trajectory)
        if total_in_history > 0:
            self.logger.info(
                f"Trajectory: {num_included}/{total_in_history} responses included "
                f"({total_tokens}/{self.max_history_tokens} tokens, "
                f"{total_tokens/self.max_history_tokens*100:.1f}% of budget)"
            )
        
        trajectory_text = "\n\n".join(formatted_entries) if formatted_entries else ""
        return trajectory_text, num_included, total_tokens
    
    def _format_single_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single trajectory entry with full context (all responses)."""
        question = entry['question']
        all_responses = entry['all_responses']
        best_idx = entry['best_response_idx']
        reasoning = entry['reasoning']
        confidence = entry['confidence']
        
        # Format all responses
        responses_text = []
        for i, response in enumerate(all_responses):
            marker = " [SELECTED]" if i == best_idx else ""
            responses_text.append(f"Response {i+1}{marker}:\n{response}")
        
        return f"""Sample {entry['sample_id']}:
Question: {question}

{chr(10).join(responses_text)}

Judge's Selection: Response {best_idx + 1}
Judge's Reasoning: {reasoning}
Confidence: {confidence:.2f}"""
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not text:
            return 0
        try:
            tokens = self.judge_manager.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {e}, using character estimate")
            return len(text) // 4  # Rough estimate
    
    def reset_trajectory(self) -> None:
        """Reset the trajectory history (useful between experiments)."""
        self.trajectory = []
        self.logger.info("Trajectory history reset")

