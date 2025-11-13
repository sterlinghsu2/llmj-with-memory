"""
Model management for LLM response generation.
"""
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

from vllm import LLM, SamplingParams
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
    """Manages LLM for response generation."""
    
    def __init__(self, config):
        self.config = config
        self.model: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Initialize the vLLM model and sampling parameters.""" 
        print(f"Loading model: {self.config.model.name}")
        
        self.model = LLM(
            model=self.config.model.name,
            seed=self.config.model.seed,
            trust_remote_code=True,
            max_model_len=16384,  # 16K context - safe for FlexAttention on CC 7.5
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.50,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        
        # Use 'n' parameter to generate multiple diverse responses per prompt
        sampling_kwargs = {
            'temperature': self.config.model.temperature,
            'max_tokens': self.config.model.max_tokens,
            'n': self.config.model.num_instances,
            'seed': self.config.model.seed,
            'stop': ['<|end_of_text|>', '<|eot_id|>'],
        }
        if self.config.model.top_p is not None:
            sampling_kwargs['top_p'] = self.config.model.top_p
        if self.config.model.top_k is not None:
            sampling_kwargs['top_k'] = self.config.model.top_k
            
        self.sampling_params = SamplingParams(**sampling_kwargs)
        
        print(f"Model loaded successfully with temperature={self.config.model.temperature}")
    
    def generate_responses(self, sample: DataSample) -> List[GeneratedResponse]:
        """Generate multiple responses for a single sample using vLLM's n parameter."""
        responses = []
        
        # Use chat template to ensure proper stop token handling
        if hasattr(self.tokenizer, 'apply_chat_template'):
            question_with_instruction = format_generation_prompt(sample.question)
            messages = [{"role": "user", "content": question_with_instruction}]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = format_generation_prompt(sample.question)
        
        start_time = time.time()
        outputs = self.model.generate([prompt], self.sampling_params)
        generation_time = time.time() - start_time
        
        # vLLM returns one RequestOutput containing multiple CompletionOutputs
        request_output = outputs[0]
        
        for i, completion in enumerate(request_output.outputs):
            response_text = completion.text.strip()
            
            response = GeneratedResponse(
                text=response_text,
                sample_id=sample.sample_id,
                response_id=i,
                generation_time=generation_time / len(request_output.outputs),
                metadata={
                    'prompt': prompt,
                    'finish_reason': completion.finish_reason,
                }
            )
            responses.append(response)
        
        return responses

