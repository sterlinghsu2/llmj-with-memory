"""
Judge model management for evaluating LLM responses.
"""
from typing import List, Tuple, Any, Optional
import re

from vllm import LLM, SamplingParams
from generator import GeneratedResponse
from prompt_templates import format_best_of_n_prompt, format_score_based_prompt


class JudgeModelManager:
    """Manages LLM for judging responses.
    
    Note: Judge always shares the same model instance as the generator,
    but uses different sampling parameters (lower temperature for consistency).
    """
    
    def __init__(self, config, shared_model: LLM):
        self.config = config
        self.model = shared_model
        self.tokenizer = self.model.get_tokenizer()
        self.sampling_params: Optional[SamplingParams] = None
        self._setup_sampling_params()
    
    def _setup_sampling_params(self) -> None:
        """Setup sampling parameters for judging (typically more deterministic)."""
        print(f"Configuring judge sampling params (temp={self.config.judge.temperature})")
        
        self.sampling_params = SamplingParams(
            temperature=self.config.judge.temperature,
            max_tokens=self.config.judge.max_tokens,
            seed=self.config.judge.seed,
        )
        
        print("Judge configured successfully")
    
    def judge_best_of_n(
        self, 
        question: str, 
        responses: List[GeneratedResponse], 
        ground_truth: str
    ) -> Tuple[int, str, float]:
        """Judge which response is best among N responses."""
        prompt = format_best_of_n_prompt(question, responses, ground_truth, self.tokenizer)
        outputs = self.model.generate([prompt], self.sampling_params)
        judgment = outputs[0].outputs[0].text.strip()
        best_idx, reasoning, confidence = self._parse_best_of_n_judgment(judgment)
        return best_idx, reasoning, confidence
    
    def judge_score_based(
        self, 
        question: str, 
        response: GeneratedResponse,
        ground_truth: str
    ) -> Tuple[float, str]:
        """Judge a single response and assign a score (0-10)."""
        prompt = format_score_based_prompt(question, response.text, ground_truth)
        outputs = self.model.generate([prompt], self.sampling_params)
        judgment = outputs[0].outputs[0].text.strip()
        score, reasoning = self._parse_score_based_judgment(judgment)
        return score, reasoning
    
    def judge_score_based_batch(
        self, 
        question: str, 
        responses: List[GeneratedResponse],
        ground_truth: str
    ) -> List[Tuple[float, str]]:
        """Judge multiple responses in a single batch using vLLM batching."""
        prompts = [
            format_score_based_prompt(question, resp.text, ground_truth) 
            for resp in responses
        ]
        
        outputs = self.model.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            judgment = output.outputs[0].text.strip()
            score, reasoning = self._parse_score_based_judgment(judgment)
            results.append((score, reasoning))
        
        return results
    
    def _parse_best_of_n_judgment(self, judgment: str) -> Tuple[int, str, float]:
        """Parse best-of-N judgment and extract index, reasoning, confidence."""
        lines = judgment.split('\n')
        best_idx = 0
        reasoning = ""
        confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith("Best Response:"):
                try:
                    best_idx = int(line.split(":")[1].strip()) - 1
                except (ValueError, IndexError):
                    best_idx = 0
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.5
        
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


def create_judge_manager(config, shared_model: LLM) -> JudgeModelManager:
    """Factory function to create judge manager."""
    return JudgeModelManager(config, shared_model)

