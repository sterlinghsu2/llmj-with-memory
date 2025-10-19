"""
Dataset handling for Math500 and other mathematical reasoning datasets.
"""
import json
import os
import random
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


@dataclass
class DataSample:
    """Represents a single data sample with question and answer."""
    question: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None
    sample_id: Optional[str] = None

    def __post_init__(self):
        if self.sample_id is None:
            self.sample_id = str(hash(self.question))[:8]


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.samples: List[DataSample] = []
    
    def load(self) -> List[DataSample]:
        """Load dataset and return list of samples."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[DataSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]


class Math500Loader(DatasetLoader):
    """Loader for Math500 dataset from HuggingFace.
    
    Loads MATH-500 from HuggingFace (HuggingFaceH4/MATH-500).
    This is a subset of 500 problems from the MATH benchmark.
    """
    
    # HuggingFace dataset configuration
    HF_DATASET = ('HuggingFaceH4/MATH-500', None, 'test')  # (dataset_name, config, split)
    
    def __init__(self, data_path: Optional[str] = None):
        super().__init__(data_path)
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. Install with: pip install datasets"
            )
        
        # If a local path is provided, use it; otherwise use HuggingFace
        if data_path is not None and not os.path.exists(data_path):
            raise FileNotFoundError(f"Specified dataset file not found: {data_path}")
    
    def _load_from_huggingface(self) -> List[DataSample]:
        """Load MATH-500 dataset from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        hf_name, config, split = self.HF_DATASET
        
        print(f"Loading {hf_name} from HuggingFace (split={split})...")
        
        try:
            if config:
                dataset = load_dataset(hf_name, config, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(hf_name, split=split, trust_remote_code=True)
            
            samples = []
            for i, item in enumerate(dataset):
                sample = self._parse_sample(dict(item), i)
                if sample:
                    samples.append(sample)
            
            print(f"Loaded {len(samples)} samples from HuggingFace")
            return samples
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            raise
    
    def load(self) -> List[DataSample]:
        """Load the MATH-500 dataset from HuggingFace or local file."""
        # If no local path specified, load from HuggingFace
        if self.data_path is None:
            self.samples = self._load_from_huggingface()
            return self.samples
        
        # Load from local file
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.samples = []
        
        try:
            # Try JSONL format first
            with open(self.data_path, 'r') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        sample = self._parse_sample(data, line_no)
                        if sample:
                            self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_no}: {e}")
                        continue
        except Exception:
            # Try regular JSON format
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            sample = self._parse_sample(item, i)
                            if sample:
                                self.samples.append(sample)
                    else:
                        sample = self._parse_sample(data, 0)
                        if sample:
                            self.samples.append(sample)
            except Exception as e:
                raise ValueError(f"Could not parse dataset at {self.data_path}: {e}")
        
        print(f"Loaded {len(self.samples)} samples from {self.data_path}")
        return self.samples
    
    def _parse_sample(self, data: Dict[str, Any], sample_id: int) -> Optional[DataSample]:
        """Parse a single sample from the dataset.
        
        MATH-500 format:
        - problem: The question
        - answer: The final answer (what we'll use for ground truth)
        - solution: The full solution with reasoning
        - subject: Math subject area
        - level: Difficulty level (1-5)
        - unique_id: Original problem ID
        """
        # Handle different possible formats
        question = None
        answer = None
        
        # Common field names for questions (prioritize 'problem' for MATH-500)
        question_fields = ['problem', 'question', 'input', 'query', 'prompt']
        # For MATH-500, use 'answer' (the final answer), not 'solution' (the full reasoning)
        answer_fields = ['answer', 'solution', 'output', 'target', 'ground_truth']
        
        for field in question_fields:
            if field in data:
                question = data[field]
                break
        
        for field in answer_fields:
            if field in data:
                answer = data[field]
                break
        
        if not question or not answer:
            print(f"Warning: Sample {sample_id} missing question or answer")
            return None
        
        # Clean up the text
        question = question.strip()
        answer = answer.strip()
        
        # Extract metadata (including subject, level, unique_id for MATH-500)
        metadata = {k: v for k, v in data.items() 
                   if k not in ['problem', 'question', 'answer', 'solution']}
        
        # Use unique_id from MATH-500 if available, otherwise use index
        sample_id_str = data.get('unique_id', str(sample_id))
        
        return DataSample(
            question=question,
            answer=answer,
            metadata=metadata if metadata else None,
            sample_id=sample_id_str
        )


class DatasetManager:
    """Manages dataset loading and preprocessing."""
    
    def __init__(self, config):
        self.config = config
        self.loader = None
        self.samples = []
    
    def load_dataset(self) -> List[DataSample]:
        """Load dataset based on configuration."""
        if self.config.dataset.name.lower() == "math500":
            self.loader = Math500Loader(self.config.dataset.data_path)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset.name}")
        
        self.samples = self.loader.load()
        
        # Apply dataset configuration
        if self.config.dataset.shuffle:
            random.seed(self.config.dataset.seed)
            random.shuffle(self.samples)
        
        if self.config.dataset.max_samples:
            self.samples = self.samples[:self.config.dataset.max_samples]
        
        return self.samples
    
    def get_sample(self, idx: int) -> DataSample:
        """Get a specific sample by index."""
        return self.samples[idx]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[DataSample]:
        return iter(self.samples)


# Convenience function
def load_math500(data_path: Optional[str] = None, max_samples: Optional[int] = None) -> List[DataSample]:
    """Load Math500 dataset with optional limits."""
    loader = Math500Loader(data_path)
    samples = loader.load()
    
    if max_samples:
        samples = samples[:max_samples]
    
    return samples
