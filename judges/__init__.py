"""
Base classes for LLM judges.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from dataset import DataSample
from generator import GeneratedResponse


@dataclass
class JudgmentResult:
    """Base class for judgment results."""
    sample_id: str
    method: str  # "best_of_n" or "score_based"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class BestOfNResult(JudgmentResult):
    """Result from Best-of-N judgment."""
    best_response_idx: int = 0
    best_response: GeneratedResponse = None
    judge_reasoning: str = ""
    confidence: float = 0.0
    all_responses: List[GeneratedResponse] = None
    is_correct: bool = False  # Whether the selected response is correct (verified by Math-Verify)
    verification_reasoning: str = ""  # Reasoning from Math-Verify
    pass_at_n: bool = False  # Whether ANY response is correct (Pass@N metric)
    response_correctness: List[bool] = None  # Individual correctness for each response
    
    def __post_init__(self):
        super().__post_init__()
        self.method = "best_of_n"
        if self.all_responses is None:
            self.all_responses = []
        if self.response_correctness is None:
            self.response_correctness = []


@dataclass
class ScoreBasedResult(JudgmentResult):
    """Result from score-based judgment."""
    scores: List[float] = None  # One score per response (from LLM judge)
    reasoning: List[str] = None  # One reasoning per response (from LLM judge)
    responses: List[GeneratedResponse] = None
    average_score: float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    best_response_idx: int = 0  # Index of highest-scored response
    is_correct: bool = False  # Whether the highest-scored response is correct (verified by Math-Verify)
    verification_reasoning: str = ""  # Reasoning from Math-Verify for the highest-scored response
    pass_at_n: bool = False  # Whether ANY response is correct (Pass@N metric)
    response_correctness: List[bool] = None  # Individual correctness for each response
    
    def __post_init__(self):
        super().__post_init__()
        self.method = "score_based"
        if self.scores is None:
            self.scores = []
        if self.reasoning is None:
            self.reasoning = []
        if self.responses is None:
            self.responses = []
        if self.response_correctness is None:
            self.response_correctness = []
        if self.scores:
            self.average_score = sum(self.scores) / len(self.scores)
            best = max(self.scores)
            self.best_score = best
            self.worst_score = min(self.scores)
            self.best_response_idx = self.scores.index(best)


@dataclass
class MajorityVoteResult(JudgmentResult):
    """Result from majority voting (no LLM judge, just answer extraction + voting)."""
    extracted_answers: List[str] = None  # Extracted answer from each response
    answer_counts: dict = None  # Count of each unique answer
    majority_answer: str = ""  # The most common answer
    majority_count: int = 0  # How many responses had the majority answer
    responses: List[GeneratedResponse] = None
    is_correct: bool = False  # Whether the majority answer is correct (verified by Math-Verify)
    verification_reasoning: str = ""  # Reasoning from Math-Verify
    pass_at_n: bool = False  # Whether ANY response is correct (Pass@N metric)
    response_correctness: List[bool] = None  # Individual correctness for each response
    
    def __post_init__(self):
        super().__post_init__()
        self.method = "majority_vote"
        if self.extracted_answers is None:
            self.extracted_answers = []
        if self.answer_counts is None:
            self.answer_counts = {}
        if self.responses is None:
            self.responses = []
        if self.response_correctness is None:
            self.response_correctness = []


class BaseJudge(ABC):
    """Abstract base class for LLM judges."""
    
    def __init__(self, config, judge_manager):
        self.config = config
        self.judge_manager = judge_manager
    
    @abstractmethod
    def evaluate(self, sample: DataSample, responses: List[GeneratedResponse]) -> JudgmentResult:
        """Evaluate responses for a sample."""
        pass
