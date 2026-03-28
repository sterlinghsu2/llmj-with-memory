"""
Inference backend abstraction for LLM calls.

Provides a unified interface for both local vLLM inference and remote
API-based inference (via litellm). The rest of the codebase interacts
with this interface, making the inference method transparent.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a single completion from a list of chat messages."""
        pass

    @abstractmethod
    def generate_n(
        self,
        messages: List[Dict[str, str]],
        n: int,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate N completions from a list of chat messages."""
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        """Return the tokenizer instance."""
        pass

    @property
    @abstractmethod
    def max_model_len(self) -> int:
        """Return the maximum context length of the model."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Apply the chat template to messages and return the raw prompt string.

        Used for accurate token counting (e.g. by the streaming judge to
        compute context budget).  Both backends produce the same result here
        because they share the same tokenizer.
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return messages[-1]['content'] if messages else ""


class LocalVLLMBackend(InferenceBackend):
    """Backend that runs inference locally using vLLM."""

    def __init__(self, config):
        from vllm import LLM

        self.config = config
        print(f"Loading model: {config.model.name}")

        self._model = LLM(
            model=config.model.name,
            seed=config.model.seed,
            trust_remote_code=True,
            max_model_len=16384,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.50,
        )
        self._tokenizer = self._model.get_tokenizer()
        print(f"Model loaded successfully")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_model_len(self) -> int:
        return self._model.llm_engine.model_config.max_model_len

    @property
    def model(self):
        """Direct access to the underlying vLLM model instance."""
        return self._model

    def generate(self, messages, temperature, max_tokens, seed=None, stop=None):
        from vllm import SamplingParams

        prompt = self.format_prompt(messages)
        kwargs = {
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if seed is not None:
            kwargs['seed'] = seed
        if stop is not None:
            kwargs['stop'] = stop

        outputs = self._model.generate([prompt], SamplingParams(**kwargs))
        return outputs[0].outputs[0].text.strip()

    def generate_n(self, messages, n, temperature, max_tokens,
                   seed=None, top_p=None, top_k=None, stop=None):
        from vllm import SamplingParams

        prompt = self.format_prompt(messages)
        kwargs = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'n': n,
        }
        if seed is not None:
            kwargs['seed'] = seed
        if top_p is not None:
            kwargs['top_p'] = top_p
        if top_k is not None:
            kwargs['top_k'] = top_k
        if stop is not None:
            kwargs['stop'] = stop

        outputs = self._model.generate([prompt], SamplingParams(**kwargs))
        return [completion.text.strip() for completion in outputs[0].outputs]


class APIBackend(InferenceBackend):
    """Backend that calls a remote OpenAI-compatible API via litellm."""

    def __init__(self, config):
        import litellm
        from transformers import AutoTokenizer

        self.config = config
        self.api_base_url = config.api_base_url
        self.api_key = config.api_key or "none"
        self.model_name = config.model.name
        self._max_model_len = config.api_max_model_len or 16384

        # "openai/" prefix tells litellm to use the OpenAI-compatible protocol
        self.litellm_model = f"openai/{self.model_name}"

        print(f"Loading tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        litellm.drop_params = True
        print(f"API backend ready: {self.api_base_url} ({self.model_name})")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    def generate(self, messages, temperature, max_tokens, seed=None, stop=None):
        import litellm

        kwargs = {
            'model': self.litellm_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'api_base': self.api_base_url,
            'api_key': self.api_key,
        }
        if seed is not None:
            kwargs['seed'] = seed
        if stop is not None:
            kwargs['stop'] = stop

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content.strip()

    def generate_n(self, messages, n, temperature, max_tokens,
                   seed=None, top_p=None, top_k=None, stop=None):
        import litellm

        kwargs = {
            'model': self.litellm_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'n': n,
            'api_base': self.api_base_url,
            'api_key': self.api_key,
        }
        if seed is not None:
            kwargs['seed'] = seed
        if top_p is not None:
            kwargs['top_p'] = top_p
        if top_k is not None:
            kwargs['top_k'] = top_k
        if stop is not None:
            kwargs['stop'] = stop

        response = litellm.completion(**kwargs)
        return [choice.message.content.strip() for choice in response.choices]


def create_backend(config) -> InferenceBackend:
    """Factory function to create the appropriate backend from config."""
    if config.inference_backend == "api":
        if not config.api_base_url:
            raise ValueError("api_base_url must be set when using the API backend")
        return APIBackend(config)
    else:
        return LocalVLLMBackend(config)
