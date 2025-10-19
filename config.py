"""
Configuration management for LLM-as-a-Judge pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """Configuration for the main LLM being evaluated."""
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 1.0
    max_tokens: int = 1024
    num_instances: int = 8  # Number of responses to generate per question
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    seed: Optional[int] = 42


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge.
    
    Note: Judge always uses the same model as the generator (from ModelConfig),
    but with different sampling parameters (lower temperature for consistency).
    Math-Verify is always used for deterministic ground truth verification.
    """
    temperature: float = 0.1  # Lower temperature for more consistent judging
    max_tokens: int = 1024
    seed: Optional[int] = 42


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    name: str = "math500"
    data_path: Optional[str] = None
    max_samples: Optional[int] = None  # None for all samples
    shuffle: bool = False
    seed: Optional[int] = 42


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    experiment_name: str = "llm_judge_baseline"
    output_dir: str = "experiments"
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Evaluation settings
    enable_best_of_n: bool = True
    enable_score_based: bool = True
    enable_majority_vote: bool = True  # Baseline: majority voting on extracted answers (no LLM judge)
    
    # System settings
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 1
    save_intermediate_results: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Initialize experiment directory path (directories created on-demand)."""
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)

    def save(self, filepath: Optional[str] = None) -> str:
        """Save configuration to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        if filepath is None:
            filepath = os.path.join(self.experiment_dir, "config.json")
        
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'model': {
                'name': self.model.name,
                'temperature': self.model.temperature,
                'max_tokens': self.model.max_tokens,
                'num_instances': self.model.num_instances,
                'top_p': self.model.top_p,
                'top_k': self.model.top_k,
                'seed': self.model.seed,
            },
            'judge': {
                'model_name': self.model.name,  # Judge uses same model as generator
                'temperature': self.judge.temperature,
                'max_tokens': self.judge.max_tokens,
                'seed': self.judge.seed,
            },
            'dataset': {
                'name': self.dataset.name,
                'data_path': self.dataset.data_path,
                'max_samples': self.dataset.max_samples,
                'shuffle': self.dataset.shuffle,
                'seed': self.dataset.seed,
            },
            'enable_best_of_n': self.enable_best_of_n,
            'enable_score_based': self.enable_score_based,
            'enable_majority_vote': self.enable_majority_vote,
            'device': self.device,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'save_intermediate_results': self.save_intermediate_results,
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        judge_config = JudgeConfig(**config_dict.get('judge', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'judge', 'dataset']}
        
        return cls(
            model=model_config,
            judge=judge_config,
            dataset=dataset_config,
            **main_config
        )


def get_default_config() -> ExperimentConfig:
    """Get default configuration for experiments."""
    return ExperimentConfig()
