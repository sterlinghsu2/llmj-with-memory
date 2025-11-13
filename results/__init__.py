"""
Results collection, analysis, and presentation system.
"""
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from judges import BestOfNResult, ScoreBasedResult, MajorityVoteResult, JudgmentResult


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    experiment_name: str
    timestamp: str
    total_samples: int
    
    # Best-of-N results
    best_of_n_avg_confidence: float = 0.0
    best_of_n_accuracy: float = 0.0
    best_of_n_accuracy_conditional: float = 0.0  # Accuracy excluding samples where no response is correct
    
    # Score-based results  
    score_based_avg_score: float = 0.0
    score_based_std_score: float = 0.0
    score_based_accuracy: float = 0.0  # Accuracy of highest-scored responses (verified by Math-Verify)
    score_based_accuracy_conditional: float = 0.0  # Accuracy excluding samples where no response is correct
    
    # Majority Vote results (baseline)
    majority_vote_accuracy: float = 0.0  # Accuracy of majority answers (verified by Math-Verify)
    majority_vote_accuracy_conditional: float = 0.0  # Accuracy excluding samples where no response is correct
    majority_vote_avg_agreement: float = 0.0  # Average votes for majority answer / total responses
    pass_at_n_rate: float = 0.0  # Percentage of samples where at least one response is correct
    num_samples_with_correct_response: int = 0  # Number of samples where at least one response is correct
    
    # Model configuration
    model_name: str = ""
    judge_model_name: str = ""
    temperature: float = 0.0
    num_instances: int = 0


class ResultsCollector:
    """Collects and manages experimental results."""
    
    def __init__(self, config):
        self.config = config
        self.best_of_n_results: List[BestOfNResult] = []
        self.score_based_results: List[ScoreBasedResult] = []
        self.majority_vote_results: List['MajorityVoteResult'] = []
        self.experiment_start_time = datetime.now()
    
    def add_best_of_n_result(self, result: BestOfNResult):
        """Add a Best-of-N judgment result."""
        self.best_of_n_results.append(result)
    
    def add_score_based_result(self, result: ScoreBasedResult):
        """Add a score-based judgment result."""
        self.score_based_results.append(result)
    
    def add_majority_vote_result(self, result: 'MajorityVoteResult'):
        """Add a majority vote result."""
        self.majority_vote_results.append(result)
    
    def add_results(self, results: List[JudgmentResult]):
        """Add multiple results of mixed types."""
        for result in results:
            if isinstance(result, BestOfNResult):
                self.add_best_of_n_result(result)
            elif isinstance(result, ScoreBasedResult):
                self.add_score_based_result(result)
            elif isinstance(result, MajorityVoteResult):
                self.add_majority_vote_result(result)
    
    def calculate_summary(self) -> ExperimentSummary:
        """Calculate summary statistics for the experiment."""
        summary = ExperimentSummary(
            experiment_name=self.config.experiment_name,
            timestamp=self.experiment_start_time.isoformat(),
            total_samples=len(self.best_of_n_results) or len(self.score_based_results),
            model_name=self.config.model.name,
            judge_model_name=self.config.model.name,  # Judge uses same model as generator
            temperature=self.config.model.temperature,
            num_instances=self.config.model.num_instances,
        )
        
        # Calculate Best-of-N statistics
        if self.best_of_n_results:
            confidences = [r.confidence for r in self.best_of_n_results if r.confidence is not None]
            summary.best_of_n_avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate overall accuracy from Math-Verify verification
            correct_count = sum(1 for r in self.best_of_n_results if r.is_correct)
            summary.best_of_n_accuracy = correct_count / len(self.best_of_n_results) if self.best_of_n_results else 0.0
            
            # Calculate conditional accuracy (only samples where at least one response is correct)
            samples_with_correct = [r for r in self.best_of_n_results if r.pass_at_n]
            if samples_with_correct:
                conditional_correct_count = sum(1 for r in samples_with_correct if r.is_correct)
                summary.best_of_n_accuracy_conditional = conditional_correct_count / len(samples_with_correct)
            else:
                summary.best_of_n_accuracy_conditional = 0.0
        
        # Calculate Score-based statistics
        if self.score_based_results:
            all_scores = []
            for result in self.score_based_results:
                all_scores.extend(result.scores)
            
            if all_scores:
                summary.score_based_avg_score = np.mean(all_scores)
                summary.score_based_std_score = np.std(all_scores)
            
            # Calculate overall accuracy from Math-Verify verification of highest-scored responses
            correct_count = sum(1 for r in self.score_based_results if r.is_correct)
            summary.score_based_accuracy = correct_count / len(self.score_based_results) if self.score_based_results else 0.0
            
            # Calculate conditional accuracy (only samples where at least one response is correct)
            samples_with_correct = [r for r in self.score_based_results if r.pass_at_n]
            if samples_with_correct:
                conditional_correct_count = sum(1 for r in samples_with_correct if r.is_correct)
                summary.score_based_accuracy_conditional = conditional_correct_count / len(samples_with_correct)
            else:
                summary.score_based_accuracy_conditional = 0.0
        
        # Calculate Majority Vote statistics
        if self.majority_vote_results:
            # Calculate overall accuracy
            correct_count = sum(1 for r in self.majority_vote_results if r.is_correct)
            summary.majority_vote_accuracy = correct_count / len(self.majority_vote_results) if self.majority_vote_results else 0.0
            
            # Calculate Pass@N rate (percentage of samples where at least one response is correct)
            pass_at_n_count = sum(1 for r in self.majority_vote_results if r.pass_at_n)
            summary.pass_at_n_rate = pass_at_n_count / len(self.majority_vote_results) if self.majority_vote_results else 0.0
            summary.num_samples_with_correct_response = pass_at_n_count
            
            # Calculate conditional accuracy (only samples where at least one response is correct)
            # This is the "fair" judge accuracy that excludes impossible cases
            samples_with_correct = [r for r in self.majority_vote_results if r.pass_at_n]
            if samples_with_correct:
                conditional_correct_count = sum(1 for r in samples_with_correct if r.is_correct)
                summary.majority_vote_accuracy_conditional = conditional_correct_count / len(samples_with_correct)
            else:
                summary.majority_vote_accuracy_conditional = 0.0
            
            # Calculate average agreement (majority count / total responses)
            agreements = []
            for result in self.majority_vote_results:
                if result.majority_count > 0:
                    # Calculate agreement as majority_count / num_responses
                    num_responses = len(result.responses)
                    if num_responses > 0:
                        agreements.append(result.majority_count / num_responses)
            summary.majority_vote_avg_agreement = np.mean(agreements) if agreements else 0.0
        
        return summary
    
    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save all results to files."""
        if output_dir is None:
            output_dir = self.config.experiment_dir
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # Save detailed results as JSON
        detailed_results = {
            'experiment_config': self.config.to_dict(),
            'summary': asdict(self.calculate_summary()),
            'best_of_n_results': [asdict(r) for r in self.best_of_n_results],
            'score_based_results': [asdict(r) for r in self.score_based_results],
            'majority_vote_results': [asdict(r) for r in self.majority_vote_results],
            'timestamp': datetime.now().isoformat(),
        }
        
        detailed_file = os.path.join(output_dir, 'detailed_results.json')
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        saved_files['detailed_results'] = detailed_file
        
        # Save summary as JSON
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(asdict(self.calculate_summary()), f, indent=2, default=str)
        saved_files['summary'] = summary_file
        
        # Save Best-of-N results as CSV
        if self.best_of_n_results:
            best_of_n_file = os.path.join(output_dir, 'best_of_n_results.csv')
            with open(best_of_n_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'best_response_idx', 'confidence', 
                               'judge_reasoning', 'num_responses'])
                
                for result in self.best_of_n_results:
                    writer.writerow([
                        result.sample_id,
                        result.best_response_idx,
                        result.confidence,
                        result.judge_reasoning[:200] + "..." if len(result.judge_reasoning) > 200 else result.judge_reasoning,
                        len(result.all_responses)
                    ])
            saved_files['best_of_n_csv'] = best_of_n_file
        
        # Save Score-based results as CSV
        if self.score_based_results:
            score_based_file = os.path.join(output_dir, 'score_based_results.csv')
            with open(score_based_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'avg_score', 'best_score', 'worst_score', 
                               'num_responses', 'all_scores'])
                
                for result in self.score_based_results:
                    writer.writerow([
                        result.sample_id,
                        result.average_score,
                        result.best_score,
                        result.worst_score,
                        len(result.responses),
                        ','.join(map(str, result.scores))
                    ])
            saved_files['score_based_csv'] = score_based_file
        
        return saved_files


class ResultsAnalyzer:
    """Analyzes and compares experimental results."""
    
    def __init__(self, results_collector: ResultsCollector):
        self.collector = results_collector
    
    def analyze_best_of_n_performance(self) -> Dict[str, Any]:
        """Analyze Best-of-N judgment performance."""
        if not self.collector.best_of_n_results:
            return {}
        
        results = self.collector.best_of_n_results
        
        # Calculate confidence distribution
        confidences = [r.confidence for r in results if r.confidence is not None]
        
        # Analyze selection patterns
        selection_counts = {}
        for result in results:
            idx = result.best_response_idx
            selection_counts[idx] = selection_counts.get(idx, 0) + 1
        
        return {
            'total_judgments': len(results),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'std_confidence': np.std(confidences) if confidences else 0.0,
            'confidence_distribution': {
                'min': np.min(confidences) if confidences else 0.0,
                'max': np.max(confidences) if confidences else 0.0,
                'median': np.median(confidences) if confidences else 0.0,
                'q25': np.percentile(confidences, 25) if confidences else 0.0,
                'q75': np.percentile(confidences, 75) if confidences else 0.0,
            },
            'selection_patterns': selection_counts,
            'most_selected_position': max(selection_counts.items(), key=lambda x: x[1])[0] if selection_counts else None,
        }
    
    def analyze_score_based_performance(self) -> Dict[str, Any]:
        """Analyze score-based judgment performance."""
        if not self.collector.score_based_results:
            return {}
        
        results = self.collector.score_based_results
        
        # Collect all scores
        all_scores = []
        sample_averages = []
        sample_variances = []
        
        for result in results:
            all_scores.extend(result.scores)
            sample_averages.append(result.average_score)
            if len(result.scores) > 1:
                sample_variances.append(np.var(result.scores))
        
        return {
            'total_judgments': len(results),
            'total_individual_scores': len(all_scores),
            'overall_stats': {
                'mean': np.mean(all_scores) if all_scores else 0.0,
                'std': np.std(all_scores) if all_scores else 0.0,
                'min': np.min(all_scores) if all_scores else 0.0,
                'max': np.max(all_scores) if all_scores else 0.0,
                'median': np.median(all_scores) if all_scores else 0.0,
            },
            'sample_averages': {
                'mean': np.mean(sample_averages) if sample_averages else 0.0,
                'std': np.std(sample_averages) if sample_averages else 0.0,
                'min': np.min(sample_averages) if sample_averages else 0.0,
                'max': np.max(sample_averages) if sample_averages else 0.0,
            },
            'within_sample_variance': {
                'mean': np.mean(sample_variances) if sample_variances else 0.0,
                'std': np.std(sample_variances) if sample_variances else 0.0,
            },
            'score_distribution': np.histogram(all_scores, bins=10)[0].tolist() if all_scores else [],
        }
    
    def compare_methods(self) -> Dict[str, Any]:
        """Compare Best-of-N and Score-based methods."""
        best_of_n_analysis = self.analyze_best_of_n_performance()
        score_based_analysis = self.analyze_score_based_performance()
        
        comparison = {
            'best_of_n_available': bool(self.collector.best_of_n_results),
            'score_based_available': bool(self.collector.score_based_results),
            'best_of_n_analysis': best_of_n_analysis,
            'score_based_analysis': score_based_analysis,
        }
        
        # Add cross-method comparisons if both methods were used
        if self.collector.best_of_n_results and self.collector.score_based_results:
            # Find samples evaluated by both methods
            best_of_n_samples = {r.sample_id for r in self.collector.best_of_n_results}
            score_based_samples = {r.sample_id for r in self.collector.score_based_results}
            common_samples = best_of_n_samples.intersection(score_based_samples)
            
            comparison['common_samples'] = len(common_samples)
            comparison['total_samples_best_of_n'] = len(best_of_n_samples)
            comparison['total_samples_score_based'] = len(score_based_samples)
        
        return comparison
