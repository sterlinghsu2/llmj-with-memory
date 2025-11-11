"""
Entry point script for running LLM-as-a-Judge experiments.
"""
import os
import argparse
import sys
from typing import Optional

from config import ExperimentConfig, get_default_config
from pipeline import run_experiment, run_quick_test


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment settings
    parser.add_argument("--name", type=str, default="llm_judge_experiment",
                       help="Name of the experiment")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory for results")
    
    # Model settings (now configured in config.py only)
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="math500",
                       help="Dataset to use (currently only math500)")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to dataset file (auto-download if not provided)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--skip-samples", type=int, default=0,
                       help="Number of samples to skip from start (for parallel processing)")
    parser.add_argument("--shuffle", action="store_true",
                       help="Shuffle the dataset")
    
    # Evaluation settings
    parser.add_argument("--disable-best-of-n", action="store_true",
                       help="Disable Best-of-N evaluation")
    parser.add_argument("--disable-score-based", action="store_true",
                       help="Disable score-based evaluation")
    
    # System settings
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-intermediate-save", action="store_true",
                       help="Disable saving intermediate results")
    
    # Special modes
    parser.add_argument("--quick-test", type=int, default=None, metavar="N",
                       help="Run a quick test with N samples")
    parser.add_argument("--config-file", type=str, default=None,
                       help="Load configuration from JSON file")
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle quick test mode
        if args.quick_test is not None:
            print(f"Running quick test with {args.quick_test} samples...")
            results = run_quick_test(args.quick_test)
            print("Quick test completed successfully!")
            return
        
        # Load configuration
        if args.config_file:
            print(f"Loading configuration from {args.config_file}")
            config = ExperimentConfig.load(args.config_file)
        else:
            config = get_default_config()
            
            # Override only experiment-level settings from command line
            config.experiment_name = args.name
            config.output_dir = args.output_dir
            config.enable_best_of_n = not args.disable_best_of_n
            config.enable_score_based = not args.disable_score_based
            config.batch_size = args.batch_size
            config.verbose = args.verbose
            config.save_intermediate_results = not args.no_intermediate_save
            
            # Update dataset configs
            config.dataset.name = args.dataset
            config.dataset.data_path = args.data_path
            config.dataset.max_samples = args.max_samples
            config.dataset.skip_samples = args.skip_samples
            config.dataset.shuffle = args.shuffle
            config.model.seed = args.seed
            config.judge.seed = args.seed
            config.dataset.seed = args.seed
            
            # Recreate experiment directory with updated name
            config.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
            os.makedirs(config.experiment_dir, exist_ok=True)
        
        # Print configuration summary (all model settings now from config.py)
        print("\nExperiment Configuration:")
        print(f"  Name: {config.experiment_name}")
        print(f"  Model: {config.model.name}")
        print(f"  Judge Model: {config.model.name}")
        print(f"  Temperature: {config.model.temperature}")
        print(f"  Judge Temperature: {config.judge.temperature}")
        print(f"  Max Tokens: {config.model.max_tokens}")
        print(f"  Responses per Sample: {config.model.num_instances}")
        print(f"  Dataset: {config.dataset.name}")
        if config.dataset.max_samples:
            print(f"  Max Samples: {config.dataset.max_samples}")
        print(f"  Best-of-N: {config.enable_best_of_n}")
        print(f"  Score-based: {config.enable_score_based}")
        print(f"  Majority Vote: {config.enable_majority_vote}")
        print(f"  Verification: Math-Verify (always enabled)")
        print(f"  Output Dir: {config.experiment_dir}")
        print()
        
        # Run the experiment
        results = run_experiment(config)
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {config.experiment_dir}")
        
        # Print file locations
        saved_files = results.get('saved_files', {})
        # plot_files = results.get('plot_files', {})  # Disabled - no plotting
        
        if saved_files:
            print("\nSaved files:")
            for key, path in saved_files.items():
                print(f"  {key}: {path}")
        
        # Plotting disabled
        # if plot_files:
        #     print("\nGenerated plots:")
        #     for key, path in plot_files.items():
        #         print(f"  {key}: {path}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
