"""
Streaming Best-of-N judge with trajectory history.
"""
from typing import List, Dict, Any, Tuple
import logging

from judges import BaseJudge, BestOfNResult
from dataset import DataSample  
from generator import GeneratedResponse
from prompt_templates import format_streaming_best_of_n_prompt, format_distillation_prompt


class StreamingBestOfNJudge(BaseJudge):
    """Judge that selects the best response with access to previous judgment history."""
    
    def __init__(self, config, judge_manager):
        super().__init__(config, judge_manager)
        self.logger = logging.getLogger(__name__)
        
        # Initialize trajectory history
        self.trajectory: List[Dict[str, Any]] = []
        self.sample_count = 0  # Track total samples processed (independent of trajectory length)
        self.max_history_tokens = config.streaming_max_history_tokens
        self.max_history_entries = config.streaming_max_history_entries  # None = use token limit
        self.trajectory_mode = config.streaming_trajectory_mode  # "full", "minimal", or "distillation"
        self.correct_only = config.streaming_correct_only  # Only include correct judgments in history
        self.enable_distillation = config.streaming_enable_distillation  # Generate distilled memory items
        
        # Get model context limit from the vLLM model configuration
        # This uses the max_model_len value set in generator.py during model initialization
        self.model_max_length = judge_manager.model.llm_engine.model_config.max_model_len
        self.logger.info(f"Model max length: {self.model_max_length} tokens")
        
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
        
        self.sample_count += 1  # Track total samples processed
        
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
        
        # Use streaming judge with history (also formats trajectory)
        best_idx, reasoning, confidence, num_included, tokens_used = self._judge_with_history(
            sample=sample,
            responses=responses
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
        
        # Save trajectory length before adding (for accurate reporting)
        trajectory_available = len(self.trajectory)
        
        # Generate distillation if enabled
        distillation = ""
        if self.enable_distillation:
            distillation = self._generate_distillation(
                sample=sample,
                responses=responses,
                reasoning=reasoning,
                selected_idx=best_idx
            )
        
        # Add to trajectory history with full context (all responses)
        self._add_to_trajectory(
            sample_id=sample.sample_id,
            question=sample.question,
            all_responses=[r.text for r in responses],
            best_idx=best_idx,
            reasoning=reasoning,
            confidence=confidence,
            is_correct=is_correct,
            distillation=distillation
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
                'trajectory_total_responses': trajectory_available,  # Entries available before this sample
                'trajectory_included_responses': num_included,
                'history_tokens_used': tokens_used,
                'history_limit_type': 'entries' if self.max_history_entries is not None else 'tokens',
                'history_entries_limit': self.max_history_entries,
                'history_tokens_budget': self.max_history_tokens,
                'history_utilization_pct': round(tokens_used / self.max_history_tokens * 100, 1) if self.max_history_tokens > 0 else 0,
                'distillation_enabled': self.enable_distillation,
                'distillation': distillation if self.enable_distillation else None,
            }
        )
        
        if self.config.verbose:
            num_correct = sum(response_correctness)
            correct_str = "Correct ✓" if is_correct else "Incorrect ✗"
            print(f"[{self.sample_count}] {sample.sample_id}: Selected {best_idx} ({correct_str}), {num_correct}/{len(responses)} correct, Trajectory: {num_included}/{trajectory_available}")
        
        return result
    
    def _judge_with_history(
        self, 
        sample: DataSample,
        responses: List[GeneratedResponse]
    ) -> Tuple[int, str, float, int, int]:
        """Call judge with trajectory history, proactively ensuring prompt fits in context.
        
        Returns:
            Tuple of (best_idx, reasoning, confidence, num_included, tokens_used)
        """
        # Calculate base prompt size (without trajectory)
        base_prompt = format_streaming_best_of_n_prompt(
            question=sample.question,
            responses=responses,
            trajectory_history="",  # Empty trajectory for base calculation
            tokenizer=self.judge_manager.tokenizer
        )
        base_tokens = len(self.judge_manager.tokenizer.encode(base_prompt))
        
        # Calculate available tokens for trajectory history
        available_tokens = self.model_max_length - base_tokens
        self.logger.debug(
            f"Sample {sample.sample_id}: Base prompt {base_tokens} tokens, "
            f"available for history: {available_tokens} tokens"
        )
        
        # Format trajectory history to fit within available tokens
        trajectory_text, num_included, tokens_used = self._format_trajectory_with_budget(
            max_tokens=available_tokens
        )
        
        # Build final prompt with truncated trajectory
        prompt = format_streaming_best_of_n_prompt(
            question=sample.question,
            responses=responses,
            trajectory_history=trajectory_text,
            tokenizer=self.judge_manager.tokenizer
        )
        
        # Verify final prompt size
        final_tokens = len(self.judge_manager.tokenizer.encode(prompt))
        if final_tokens > self.model_max_length:
            self.logger.warning(
                f"Sample {sample.sample_id}: Final prompt ({final_tokens} tokens) exceeds "
                f"limit ({self.model_max_length} tokens). Removing all trajectories."
            )
            # Fall back to no trajectories
            trajectory_text = ""
            num_included = 0
            tokens_used = 0
            prompt = format_streaming_best_of_n_prompt(
                question=sample.question,
                responses=responses,
                trajectory_history="",
                tokenizer=self.judge_manager.tokenizer
            )
            final_tokens = len(self.judge_manager.tokenizer.encode(prompt))
        
        self.logger.info(
            f"Sample {sample.sample_id}: Added {num_included}/{len(self.trajectory)} trajectories "
            f"(trajectory tokens: {tokens_used}, final prompt: {final_tokens} tokens)"
        )
        
        # Generate judgment
        try:
            outputs = self.judge_manager.model.generate([prompt], self.judge_manager.sampling_params)
            judgment = outputs[0].outputs[0].text.strip()
            best_idx, reasoning, confidence = self.judge_manager._parse_best_of_n_judgment(judgment)
            return best_idx, reasoning, confidence, num_included, tokens_used
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                self.logger.error(
                    f"Sample {sample.sample_id}: Context overflow despite proactive truncation "
                    f"(final_tokens={final_tokens}, limit={self.model_max_length}). "
                    f"This indicates a tokenization mismatch."
                )
            raise
    
    def _generate_distillation(
        self,
        sample: DataSample,
        responses: List[GeneratedResponse],
        reasoning: str,
        selected_idx: int
    ) -> str:
        """Generate distilled memory items from judge reasoning.
        
        This is the second step of two-step distillation. After the judge has made
        its selection, this method generates generalizable insights from the reasoning.
        
        Args:
            sample: The data sample being evaluated
            responses: List of candidate responses
            reasoning: The judge's reasoning from the selection step
            selected_idx: Index of the selected response (0-based)
            
        Returns:
            Distillation text containing memory items
        """
        prompt = format_distillation_prompt(
            question=sample.question,
            responses=responses,
            judge_reasoning=reasoning,
            selected_idx=selected_idx,
            tokenizer=self.judge_manager.tokenizer
        )
        
        try:
            outputs = self.judge_manager.model.generate([prompt], self.judge_manager.sampling_params)
            distillation = outputs[0].outputs[0].text.strip()
            self.logger.debug(f"Sample {sample.sample_id}: Generated distillation ({len(distillation)} chars)")
            return distillation
        except Exception as e:
            self.logger.warning(f"Sample {sample.sample_id}: Distillation generation failed: {e}")
            return ""
    
    def _add_to_trajectory(
        self,
        sample_id: str,
        question: str,
        all_responses: list,
        best_idx: int,
        reasoning: str,
        confidence: float,
        is_correct: bool,
        distillation: str = ""
    ) -> None:
        """Add a judgment to the trajectory history with full context.
        
        Args:
            sample_id: Unique identifier for the sample
            question: The math problem
            all_responses: List of all candidate response texts
            best_idx: Index of the selected response
            reasoning: The judge's reasoning for selection
            confidence: Confidence score
            is_correct: Whether the selection was correct
            distillation: Distilled memory items (if distillation enabled)
        """
        # Skip storing incorrect trajectories when correct_only is enabled
        if self.correct_only and not is_correct:
            return
        
        entry = {
            'sample_id': sample_id,
            'question': question,
            'all_responses': all_responses,
            'best_response_idx': best_idx,
            'reasoning': reasoning,
            'confidence': confidence,
            'is_correct': is_correct,
            'distillation': distillation,
        }
        self.trajectory.append(entry)
    
    def _format_trajectory_for_prompt_with_stats(self) -> Tuple[str, int, int]:
        """Format trajectory history and return (text, num_included, tokens_used)."""
        if not self.trajectory:
            return "", 0, 0
        
        # If max_history_entries is set, use simple entry-based limit
        if self.max_history_entries is not None:
            # Note: when correct_only=True, trajectory already only contains correct entries
            entries_to_include = self.trajectory[-self.max_history_entries:] if self.max_history_entries > 0 else []
            
            formatted_entries = []
            total_tokens = 0
            for entry in entries_to_include:
                entry_text = self._format_single_entry(entry)
                formatted_entries.append(entry_text)
                total_tokens += self._count_tokens(entry_text)
            
            num_included = len(entries_to_include)
            trajectory_text = "\n\n".join(formatted_entries) if formatted_entries else ""
            
            self.logger.info(
                f"Trajectory: {num_included}/{len(self.trajectory)} entries included "
                f"(last {self.max_history_entries} entries, {total_tokens} tokens)"
            )
            
            return trajectory_text, num_included, total_tokens
        
        # Otherwise use token-based truncation (legacy behavior)
        # Note: This is only used for initial stats, actual formatting uses _format_trajectory_with_budget
        formatted_entries = []
        total_tokens = 0
        num_included = 0
        
        # Go backwards through trajectory (most recent first)
        # Note: when correct_only=True, trajectory already only contains correct entries
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
            utilization_pct = (total_tokens / self.max_history_tokens * 100) if self.max_history_tokens > 0 else 0
            self.logger.info(
                f"Trajectory: {num_included}/{total_in_history} responses included "
                f"({total_tokens}/{self.max_history_tokens} tokens, "
                f"{utilization_pct:.1f}% of budget)"
            )
        
        trajectory_text = "\n\n".join(formatted_entries) if formatted_entries else ""
        return trajectory_text, num_included, total_tokens
    
    def _format_trajectory_with_budget(self, max_tokens: int) -> Tuple[str, int, int]:
        """Format trajectory history to fit within token budget (all-or-nothing per entry).
        
        Args:
            max_tokens: Maximum tokens available for trajectory history
            
        Returns:
            tuple: (trajectory_text, num_included, tokens_used)
        """
        if not self.trajectory or max_tokens <= 0:
            return "", 0, 0
        
        formatted_entries = []
        total_tokens = 0
        num_included = 0
        
        # Determine which entries to consider (apply entry limit first, then token budget)
        if self.max_history_entries is not None and self.max_history_entries > 0:
            # Take only the most recent N entries
            entries_to_consider = self.trajectory[-self.max_history_entries:]
        else:
            entries_to_consider = self.trajectory
        
        # Go backwards through trajectory (most recent first), include whole entries or none
        # Note: when correct_only=True, trajectory already only contains correct entries
        for entry in reversed(entries_to_consider):
            entry_text = self._format_single_entry(entry)
            entry_tokens = self._count_tokens(entry_text)
            
            # All-or-nothing: only include if entire entry fits
            if total_tokens + entry_tokens <= max_tokens:
                formatted_entries.insert(0, entry_text)  # Insert at beginning to maintain chronological order
                total_tokens += entry_tokens
                num_included += 1
            else:
                # Entry doesn't fit, stop adding more
                self.logger.debug(
                    f"Trajectory entry would exceed budget: {entry_tokens} tokens needed, "
                    f"{max_tokens - total_tokens} remaining. Stopping at {num_included} entries."
                )
                break
        
        trajectory_text = "\n\n".join(formatted_entries) if formatted_entries else ""
        
        self.logger.debug(
            f"Formatted {num_included}/{len(self.trajectory)} trajectories "
            f"({total_tokens}/{max_tokens} tokens, entry limit: {self.max_history_entries})"
        )
        
        return trajectory_text, num_included, total_tokens
    
    def _format_single_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single trajectory entry based on trajectory mode."""
        question = entry['question']
        best_idx = entry['best_response_idx']
        reasoning = entry['reasoning']
        confidence = entry['confidence']
        
        if self.trajectory_mode == "distillation":
            # Distillation mode: Only distilled memory items (no question)
            distillation = entry.get('distillation', '')
            if distillation:
                return f"""Sample {entry['sample_id']}:
{distillation}"""
            else:
                # Fallback to minimal if no distillation available
                return f"""Sample {entry['sample_id']}:
Judge's Reasoning: {reasoning}
Confidence: {confidence:.2f}"""
        elif self.trajectory_mode == "minimal":
            # Minimal mode: Just question and reasoning
            return f"""Sample {entry['sample_id']}:
Question: {question}

Judge's Reasoning: {reasoning}
Confidence: {confidence:.2f}"""
        else:
            # Full mode: All responses included
            all_responses = entry['all_responses']
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
        self.sample_count = 0
        self.logger.info("Trajectory history reset")

