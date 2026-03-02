"""
Model management for LLM response generation.
"""
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

from dataset import DataSample
from prompt_templates import format_generation_prompt


@dataclass 
class GeneratedResponse:
    """Represents a generated response with metadata."""
    text: str
    sample_id: str
    response_id: int
    generation_time: float
    metadata: Optional[Dict[str, Any]] = None


class ModelManager:
    """Manages LLM for response generation via an inference backend."""
    
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
    
    def generate_responses(self, sample: DataSample) -> List[GeneratedResponse]:
        """Generate multiple responses for a single sample."""
        user_content = format_generation_prompt(sample.question)
        messages = [{"role": "user", "content": user_content}]
        
        start_time = time.time()
        texts = self.backend.generate_n(
            messages=messages,
            n=self.config.model.num_instances,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            seed=self.config.model.seed,
            top_p=self.config.model.top_p,
            top_k=self.config.model.top_k,
            stop=['<|end_of_text|>', '<|eot_id|>'],
        )
        generation_time = time.time() - start_time
        
        responses = []
        for i, text in enumerate(texts):
            response = GeneratedResponse(
                text=text,
                sample_id=sample.sample_id,
                response_id=i,
                generation_time=generation_time / len(texts),
                metadata={
                    'prompt': user_content,
                }
            )
            responses.append(response)
        
        return responses
