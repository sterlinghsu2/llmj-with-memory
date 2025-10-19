"""
Main pipeline for LLM-as-a-Judge experiments.
Orchestrates the entire workflow from data loading to results presentation.
"""
import logging
import time
import os
from typing import List, Dict, Any, Optional
import traceback

from config import ExperimentConfig, get_default_config
from dataset import DatasetManager, DataSample
from generator import ModelManager
from judge_manager import JudgeModelManager
from judges.best_of_n_judge import BestOfNJudge
from judges.score_judge import ScoreBasedJudge
from results import ResultsCollector, ResultsAnalyzer


class ExperimentPipeline:
    """Main pipeline for running LLM-as-a-Judge experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        self.dataset_manager = None
        self.model_manager = None
        self.judge_manager = None
        self.best_of_n_judge = None
        self.score_based_judge = None
        self.results_collector = None
        self.results_analyzer = None
        
        self.start_time = None
        self.end_time = None
        self.total_samples = 0
        self.processed_samples = 0
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the experiment."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Ensure experiment directory exists before creating log file
        os.makedirs(self.config.experiment_dir, exist_ok=True)
        
        log_file = os.path.join(self.config.experiment_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        try:
            self.logger.info("Loading dataset...")
            self.dataset_manager = DatasetManager(self.config)
            samples = self.dataset_manager.load_dataset()
            self.total_samples = len(samples)
            self.logger.info(f"Loaded {self.total_samples} samples")
            
            self.logger.info("Loading models...")
            self.model_manager = ModelManager(self.config)
            
            # Judge shares model instance with generator for memory efficiency
            self.logger.info("Initializing judge with shared model instance")
            self.judge_manager = JudgeModelManager(self.config, shared_model=self.model_manager.model)
            
            if self.config.enable_best_of_n:
                self.logger.info("Initializing Best-of-N judge...")
                self.best_of_n_judge = BestOfNJudge(self.config, self.judge_manager)
            
            if self.config.enable_score_based:
                self.logger.info("Initializing Score-based judge...")
                self.score_based_judge = ScoreBasedJudge(self.config, self.judge_manager)
            
            if self.config.enable_majority_vote:
                self.logger.info("Initializing Majority Vote baseline...")
                from judges.majority_vote_judge import MajorityVoteJudge
                self.majority_vote_judge = MajorityVoteJudge(self.config)
            
            self.logger.info("Initializing results system...")
            self.results_collector = ResultsCollector(self.config)
            self.results_analyzer = ResultsAnalyzer(self.results_collector)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def process_sample(self, sample: DataSample) -> Dict[str, Any]:
        """Process a single sample through the pipeline."""
        sample_results = {}
        
        try:
            self.logger.debug(f"Generating responses for sample {sample.sample_id}")
            responses = self.model_manager.generate_responses(sample)
            sample_results['responses'] = responses
            sample_results['num_responses'] = len(responses)
            
            if self.config.enable_best_of_n and self.best_of_n_judge:
                self.logger.debug(f"Applying Best-of-N judgment for sample {sample.sample_id}")
                best_of_n_result = self.best_of_n_judge.evaluate(sample, responses)
                self.results_collector.add_best_of_n_result(best_of_n_result)
                sample_results['best_of_n'] = best_of_n_result
            
            if self.config.enable_score_based and self.score_based_judge:
                self.logger.debug(f"Applying Score-based judgment for sample {sample.sample_id}")
                score_based_result = self.score_based_judge.evaluate(sample, responses)
                self.results_collector.add_score_based_result(score_based_result)
                sample_results['score_based'] = score_based_result
            
            if self.config.enable_majority_vote and self.majority_vote_judge:
                self.logger.debug(f"Applying Majority Vote for sample {sample.sample_id}")
                majority_vote_result = self.majority_vote_judge.evaluate(sample, responses)
                self.results_collector.add_majority_vote_result(majority_vote_result)
                sample_results['majority_vote'] = majority_vote_result
            
            self.processed_samples += 1
            
            if self.config.verbose and self.processed_samples % 10 == 0:
                progress = (self.processed_samples / self.total_samples) * 100
                self.logger.info(f"Progress: {self.processed_samples}/{self.total_samples} ({progress:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error processing sample {sample.sample_id}: {e}")
            self.logger.error(traceback.format_exc())
            sample_results['error'] = str(e)
        
        return sample_results
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.start_time = time.time()
        
        try:
            self.initialize_components()
            
            config_file = self.config.save()
            self.logger.info(f"Configuration saved to: {config_file}")
            
            self.logger.info(f"Processing {self.total_samples} samples...")
            all_sample_results = []
            
            for sample in self.dataset_manager:
                sample_result = self.process_sample(sample)
                all_sample_results.append(sample_result)
                
                if (self.config.save_intermediate_results and 
                    self.processed_samples % 50 == 0):
                    self._save_intermediate_results()
            
            self.logger.info("Calculating final results...")
            summary = self.results_collector.calculate_summary()
            analysis = self.results_analyzer.compare_methods()
            
            self.logger.info("Saving results...")
            saved_files = self.results_collector.save_results()
            
            plot_files = {}
            
            self.end_time = time.time()
            total_time = self.end_time - self.start_time
            
            final_results = {
                'experiment_config': self.config.to_dict(),
                'summary': summary,
                'analysis': analysis,
                'runtime_stats': {
                    'total_time_seconds': total_time,
                    'total_samples': self.total_samples,
                    'processed_samples': self.processed_samples,
                    'samples_per_second': self.processed_samples / total_time if total_time > 0 else 0,
                    'start_time': self.start_time,
                    'end_time': self.end_time,
                },
                'saved_files': saved_files,
                'plot_files': plot_files,
                'sample_results': all_sample_results,
            }
            
            self.logger.info(f"Experiment completed successfully in {total_time:.2f} seconds")
            self.logger.info(f"Processed {self.processed_samples}/{self.total_samples} samples")
            self.logger.info(f"Results saved to: {self.config.experiment_dir}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_intermediate_results(self):
        """Save intermediate results during processing."""
        try:
            intermediate_file = os.path.join(
                self.config.experiment_dir, 
                f'intermediate_results_{self.processed_samples}.json'
            )
            
            summary = self.results_collector.calculate_summary()
            intermediate_data = {
                'processed_samples': self.processed_samples,
                'total_samples': self.total_samples,
                'timestamp': time.time(),
                'summary': summary.__dict__,
            }
            
            import json
            with open(intermediate_file, 'w') as f:
                json.dump(intermediate_data, f, indent=2, default=str)
                
            self.logger.info(f"Intermediate results saved: {intermediate_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def print_summary(self):
        """Print a summary of the experiment results."""
        if not self.results_collector:
            print("No results to summarize")
            return
        
        summary = self.results_collector.calculate_summary()
        analysis = self.results_analyzer.compare_methods()
        
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY: {summary.experiment_name}")
        print("="*60)
        
        print(f"Total Samples: {summary.total_samples}")
        print(f"Model: {summary.model_name}")
        print(f"Judge Model: {summary.judge_model_name}")
        print(f"Temperature: {summary.temperature}")
        print(f"Responses per Sample: {summary.num_instances}")
        
        if summary.best_of_n_avg_confidence > 0:
            print(f"\nBest-of-N Results:")
            print(f"  Average Confidence: {summary.best_of_n_avg_confidence:.4f}")
            print(f"  Accuracy (All samples): {summary.best_of_n_accuracy:.2%}")
            print(f"  Accuracy (Conditional on Pass@N): {summary.best_of_n_accuracy_conditional:.2%}")
            
            best_of_n_analysis = analysis.get('best_of_n_analysis', {})
            if best_of_n_analysis:
                print(f"  Confidence Std: {best_of_n_analysis.get('std_confidence', 0):.4f}")
                most_selected = best_of_n_analysis.get('most_selected_position')
                if most_selected is not None:
                    print(f"  Most Selected Position: {most_selected}")
        
        if summary.score_based_avg_score > 0:
            print(f"\nScore-based Results:")
            print(f"  Average LLM Score: {summary.score_based_avg_score:.4f} ± {summary.score_based_std_score:.4f}")
            print(f"  Accuracy (All samples): {summary.score_based_accuracy:.2%}")
            print(f"  Accuracy (Conditional on Pass@N): {summary.score_based_accuracy_conditional:.2%}")
        
        if summary.majority_vote_accuracy > 0:
            print(f"\nMajority Vote Results (Baseline):")
            print(f"  Pass@N Rate: {summary.pass_at_n_rate:.2%} ({summary.num_samples_with_correct_response}/{summary.total_samples} samples)")
            print(f"  Accuracy (All samples): {summary.majority_vote_accuracy:.2%}")
            print(f"  Accuracy (Conditional on Pass@N): {summary.majority_vote_accuracy_conditional:.2%}")
            print(f"  Average Agreement: {summary.majority_vote_avg_agreement:.2%}")
        
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            print(f"\nRuntime: {total_time:.2f} seconds")
            print(f"Processing Rate: {self.processed_samples / total_time:.2f} samples/second")
        
        print(f"\nResults Directory: {self.config.experiment_dir}")
        print("="*60)


def run_experiment(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run an experiment with default or custom configuration.
    
    Args:
        config: Optional experiment configuration. Uses default if None.
        
    Returns:
        Dictionary with experiment results and metadata.
    """
    if config is None:
        config = get_default_config()
    
    pipeline = ExperimentPipeline(config)
    results = pipeline.run_experiment()
    pipeline.print_summary()
    
    return results


def run_quick_test(num_samples: int = 5) -> Dict[str, Any]:
    """
    Run a quick test experiment with a small number of samples.
    
    Args:
        num_samples: Number of samples to process for testing.
        
    Returns:
        Dictionary with experiment results and metadata.
    """
    config = get_default_config()
    config.experiment_name = f"quick_test_{num_samples}_samples"
    config.dataset.max_samples = num_samples
    config.model.num_instances = 3  # Fewer responses for faster testing
    
    print(f"Running quick test with {num_samples} samples...")
    return run_experiment(config)


if __name__ == "__main__":
    # Run a quick test when script is executed directly
    run_quick_test(3)
