"""
Judge model management for evaluating LLM responses.
"""
from typing import List, Tuple, Any, Optional
import re

from generator import GeneratedResponse
from prompt_templates import format_best_of_n_prompt, format_score_based_prompt


class JudgeModelManager:
    """Manages LLM for judging responses.
    
    Uses an inference backend (local vLLM or remote API) shared with the generator.
    Applies different sampling parameters (lower temperature for consistency).
    """
    
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        print(f"Judge configured (temp={self.config.judge.temperature})")
    
    def judge_best_of_n(
        self, 
        question: str, 
        responses: List[GeneratedResponse]
    ) -> Tuple[int, str, float]:
        """Judge which response is best among N responses."""
        user_content = format_best_of_n_prompt(question, responses)
        messages = [{"role": "user", "content": user_content}]
        judgment = self.backend.generate(
            messages=messages,
            temperature=self.config.judge.temperature,
            max_tokens=self.config.judge.max_tokens,
            seed=self.config.judge.seed,
        )
        best_idx, reasoning, confidence = self._parse_best_of_n_judgment(judgment)
        return best_idx, reasoning, confidence
    
    def judge_score_based(
        self, 
        question: str, 
        response: GeneratedResponse
    ) -> Tuple[float, str]:
        """Judge a single response and assign a score (0-10)."""
        user_content = format_score_based_prompt(question, response.text)
        messages = [{"role": "user", "content": user_content}]
        judgment = self.backend.generate(
            messages=messages,
            temperature=self.config.judge.temperature,
            max_tokens=self.config.judge.max_tokens,
            seed=self.config.judge.seed,
        )
        score, reasoning = self._parse_score_based_judgment(judgment)
        return score, reasoning
    
    def _parse_best_of_n_judgment(self, judgment: str) -> Tuple[int, str, float]:
        """Parse best-of-N judgment and extract index, reasoning, confidence."""
        lines = judgment.split('\n')
        best_idx = 0
        reasoning = ""
        confidence = 0.5
        
        collecting_reasoning = False
        reasoning_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("Best Response:"):
                try:
                    best_idx = int(line_stripped.split(":")[1].strip()) - 1
                except (ValueError, IndexError):
                    best_idx = 0
                collecting_reasoning = False
            elif line_stripped.startswith("Reasoning:"):
                collecting_reasoning = True
                first_part = line_stripped.split(":", 1)[1].strip()
                if first_part:
                    reasoning_lines.append(first_part)
            elif line_stripped.startswith("Confidence:"):
                collecting_reasoning = False
                try:
                    confidence = float(line_stripped.split(":")[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.5
            elif collecting_reasoning and line_stripped:
                reasoning_lines.append(line_stripped)
        
        reasoning = " ".join(reasoning_lines)
        
        return max(0, best_idx), reasoning, confidence
    
    def _parse_score_based_judgment(self, judgment: str) -> Tuple[float, str]:
        """Parse score-based judgment and extract 0-10 score with reasoning."""
        lines = judgment.split('\n')
        score = 0.0
        reasoning = ""
        extracted_answer = ""
        confidence = "100"
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("extracted_final_answer:"):
                extracted_answer = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("score:"):
                score_text = line.split(":", 1)[1].strip() if ":" in line else ""
                try:
                    match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if match:
                        parsed_score = float(match.group(1))
                        score = max(0.0, min(10.0, parsed_score))
                    else:
                        score = 0.0
                except (ValueError, AttributeError):
                    score = 0.0
            elif line.lower().startswith("confidence:"):
                confidence = line.split(":", 1)[1].strip() if ":" in line else "100"
        
        full_reasoning = f"Extracted: {extracted_answer} | Score: {score}/10 | Confidence: {confidence}"
        if reasoning:
            full_reasoning += f" | Reasoning: {reasoning}"
        
        return score, full_reasoning
