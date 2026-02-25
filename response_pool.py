"""
Response pool loader for pre-generated responses.

This module provides utilities to load pre-generated responses from a response pool,
enabling reuse of the same responses across multiple experiments for fair comparisons.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from generator import GeneratedResponse
from dataset import DataSample


class ResponsePoolLoader:
    """Loads pre-generated responses from a response pool."""
    
    def __init__(self, pool_path: str):
        """Initialize the response pool loader.
        
        Args:
            pool_path: Path to the response pool directory (e.g., "response_pools/math500_8responses")
        """
        self.pool_path = Path(pool_path)
        
        if not self.pool_path.exists():
            raise FileNotFoundError(f"Response pool not found: {pool_path}")
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._metadata: Optional[Dict[str, Any]] = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Load and return pool metadata."""
        if self._metadata is None:
            # Try to load combined metadata or first GPU's metadata
            metadata_path = self.pool_path / "metadata.json"
            if not metadata_path.exists():
                # Look for GPU-specific metadata
                for f in self.pool_path.glob("metadata_gpu*.json"):
                    metadata_path = f
                    break
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        
        return self._metadata
    
    def _load_sample_data(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Load response data for a specific sample."""
        if sample_id in self._cache:
            return self._cache[sample_id]
        
        # Construct path from sample_id (e.g., "test/algebra/123.json")
        sample_path = self.pool_path / sample_id
        
        if not sample_path.exists():
            return None
        
        with open(sample_path) as f:
            data = json.load(f)
        
        self._cache[sample_id] = data
        return data
    
    def load_responses(self, sample: DataSample) -> Optional[List[GeneratedResponse]]:
        """Load pre-generated responses for a sample.
        
        Args:
            sample: The data sample to load responses for
            
        Returns:
            List of GeneratedResponse objects, or None if not found in pool
        """
        data = self._load_sample_data(sample.sample_id)
        
        if data is None:
            return None
        
        responses = []
        for resp_data in data.get("responses", []):
            response = GeneratedResponse(
                text=resp_data["text"],
                sample_id=resp_data["sample_id"],
                response_id=resp_data["response_id"],
                generation_time=resp_data.get("generation_time", 0.0),
                metadata=resp_data.get("metadata"),
            )
            responses.append(response)
        
        return responses
    
    def get_response_correctness(self, sample: DataSample) -> Optional[List[bool]]:
        """Get pre-computed correctness for responses.
        
        Args:
            sample: The data sample
            
        Returns:
            List of booleans indicating correctness, or None if not found
        """
        data = self._load_sample_data(sample.sample_id)
        
        if data is None:
            return None
        
        return data.get("response_correctness")
    
    def get_sample_data(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Get all data for a sample including responses and correctness.
        
        Args:
            sample: The data sample
            
        Returns:
            Dictionary with responses and metadata, or None if not found
        """
        return self._load_sample_data(sample.sample_id)
    
    def list_available_samples(self) -> List[str]:
        """List all sample IDs available in the pool."""
        samples = []
        for json_file in self.pool_path.rglob("*.json"):
            # Skip metadata files
            if json_file.name.startswith("metadata"):
                continue
            # Get relative path as sample_id
            rel_path = json_file.relative_to(self.pool_path)
            samples.append(str(rel_path))
        return sorted(samples)
    
    def verify_sample_match(self, sample: DataSample) -> bool:
        """Verify that a sample matches the one in the pool.
        
        Checks that the question and ground truth match to catch any misalignment.
        
        Args:
            sample: The data sample to verify
            
        Returns:
            True if sample matches, False otherwise
        """
        data = self._load_sample_data(sample.sample_id)
        
        if data is None:
            return False
        
        pool_question = data.get("question", "")
        pool_ground_truth = data.get("ground_truth", "")
        
        # Compare questions (strip whitespace for robustness)
        if sample.question.strip() != pool_question.strip():
            return False
        
        if sample.answer.strip() != pool_ground_truth.strip():
            return False
        
        return True
