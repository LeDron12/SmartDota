import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import yaml
import json
import sys
from datetime import datetime
import shutil

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import yaml
import json
import sys
from datetime import datetime
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.experiments.src.core.factory import TransformerFactory
from src.experiments.src.core.dataset_factory import DatasetFactory
from src.experiments.src.core.model_trainer import ModelTrainer
from src.experiments.src.utils.config import load_config, validate_transformer_configs
from src.experiments.src.transformers.dataset_converter import DatasetConverter
from src.experiments.src.transformers.starz_dataset_converter import StarzDatasetConverter
from src.experiments.src.transformers.hero_features import HeroFeaturesTransformer
from src.experiments.src.transformers.hero_stats_transformer import HeroStatsTransformer
from src.experiments.src.transformers.player_stats_transformer import PlayerStatsTransformer
from src.experiments.src.transformers.player_hero_stats_transformer import PlayerHeroStatsTransformer
from src.experiments.src.transformers.player_ratings_transformer import PlayerRatingsTransformer
logger = logging.getLogger(__name__)

def setup_logging(run_dir: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    file_handler = logging.FileHandler(run_dir / 'experiment.log')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_run_directory(config_path: Path) -> Path:
    """Create directory structure for the experiment run."""
    config_name = config_path.stem
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    run_dir = project_root / 'src' / 'experiments' / 'runs' / config_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_config(config: Dict[str, Any], run_dir: Path) -> None:
    """Save configuration to the run directory."""
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

def get_available_transformers() -> Dict[str, type]:
    """Get dictionary of available transformers."""
    return {
        'DatasetConverter': DatasetConverter,
        'StarzDatasetConverter': StarzDatasetConverter,
        'HeroFeaturesTransformer': HeroFeaturesTransformer,
        'HeroStatsTransformer': HeroStatsTransformer,
        'PlayerStatsTransformer': PlayerStatsTransformer,
        'PlayerHeroStatsTransformer': PlayerHeroStatsTransformer,
        'PlayerRatingsTransformer': PlayerRatingsTransformer
    }

def create_time_windows(
    df: pd.DataFrame,
    n_windows: int,
    window_hours: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame, List[bool], List[bool]]]:
    """
    Create time-based train/validation splits using sliding windows.
    
    Args:
        df: Input DataFrame with 'start_time' column
        n_windows: Number of validation windows to create
        window_hours: Size of each window in hours
        
    Returns:
        List of tuples (train_df, val_df, train_labels, val_labels)
    """
    # Convert start_time to datetime
    df['pd_start_time'] = pd.to_datetime(df['start_time'], unit='s')
    
    # Sort by start_time
    df = df.sort_values('pd_start_time')
    
    # Get max time
    max_time = df['pd_start_time'].max()
    
    windows = []
    for i in range(n_windows):
        # Calculate window boundaries
        val_end = max_time - pd.Timedelta(hours=i * window_hours)
        val_start = val_end - pd.Timedelta(hours=window_hours)
        train_end = val_start
        
        # Split data
        val_mask = (df['pd_start_time'] > val_start) & (df['pd_start_time'] <= val_end)
        train_mask = df['pd_start_time'] <= train_end
        
        val_df = df[val_mask].copy()
        train_df = df[train_mask].copy()

        logging.info(f'Train DF dtypes: {train_df.dtypes}')
        logging.info(f'Val DF dtypes: {val_df.dtypes}')
        
        # Get corresponding labels
        val_labels = [label for i, label in enumerate(df['radiant_win']) if val_mask.iloc[i]]
        train_labels = [label for i, label in enumerate(df['radiant_win']) if train_mask.iloc[i]]
        
        logger.info(f"\nWindow {i+1}/{n_windows}:")
        logger.info(f"Train period: {train_df['pd_start_time'].min()} to {train_df['pd_start_time'].max()}")
        logger.info(f"Validation period: {val_df['pd_start_time'].min()} to {val_df['pd_start_time'].max()}")
        logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        
        windows.append((train_df, val_df, train_labels, val_labels))
    
    return windows

def run_time_cv(
    config_path: Path,
    n_windows: int,
    window_hours: int
) -> None:
    """
    Run time-based cross-validation experiment.
    
    Args:
        config_path: Path to configuration file
        n_windows: Number of validation windows
        window_hours: Size of each window in hours
    """
    # Create run directory
    run_dir = create_run_directory(config_path)
    
    # Set up logging
    logger = setup_logging(run_dir)
    logger.info(f"Starting time-based CV experiment with config: {config_path}")
    logger.info(f"Number of windows: {n_windows}, Window size: {window_hours} hours")
    
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate configurations
        if 'dataset' not in config:
            raise ValueError("Dataset configuration is required")
        if 'transformers' not in config:
            raise ValueError("Transformer configurations are required")
        validate_transformer_configs(config['transformers'])
        
        # Save configuration
        save_config(config, run_dir)
        
        # Load dataset
        dataset_factory = DatasetFactory()
        raw_df, _ = dataset_factory.load_dataset(config['dataset'], project_root)
        
        # Create time windows
        windows = create_time_windows(raw_df, n_windows, window_hours)
        
        # Initialize transformer factory
        transformer_factory = TransformerFactory()
        available_transformers = get_available_transformers()
        for name, transformer_class in available_transformers.items():
            transformer_factory.register_transformer(name, transformer_class)
        
        # Create pipeline from config
        transformer_factory.create_pipeline(config['transformers'])
        
        # Set pipeline directory for saving transformers
        pipeline_dir = run_dir / "pipeline"
        logging.info(f"Pipeline directory: {pipeline_dir}")
        transformer_factory.set_pipeline_dir(pipeline_dir)
        
        # Store metrics for each window
        all_metrics = []
        
        # Process each window
        for i, (train_df, val_df, train_labels, val_labels) in enumerate(windows):
            logger.info(f"\nProcessing window {i+1}/{n_windows}")

            train_df['radiant_win'] = train_labels # for stats calculations
            val_df['radiant_win'] = val_labels # for stats calculations
            
            # Transform data
            transformed_train_df, categorical_features = transformer_factory.fit_transform(train_df)
            transformed_val_df, _ = transformer_factory.transform(val_df)
            
            # Create window-specific directory
            window_dir = run_dir / f"window_{i+1}"
            window_dir.mkdir(parents=True, exist_ok=True)
            
            # Train and evaluate models
            model_trainer = ModelTrainer(
                save_path=window_dir,
                config=config,
                categorical_features=categorical_features
            )

            assert 'radiant_win' not in transformed_train_df.columns
            assert 'radiant_win' not in transformed_val_df.columns
            
            metrics = model_trainer.train_and_evaluate(
                transformed_train_df, train_labels,
                transformed_val_df, val_labels
            )
            
            all_metrics.append(metrics)
            
            # Save best model for this window
            model_trainer.save_best_model(f"window_{i+1}")
        
        # Aggregate and save overall metrics
        aggregate_metrics = {}
        for model_name in all_metrics[0].keys():
            aggregate_metrics[model_name] = {
                metric: np.mean([window_metrics[model_name][metric] for window_metrics in all_metrics])
                for metric in all_metrics[0][model_name].keys()
            }
        
        # Save aggregated metrics
        with open(run_dir / 'aggregated_metrics.json', 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        # # Save individual model metrics
        # for model_name in all_metrics[0].keys():
        #     model_metrics = {
        #         'window_metrics': [
        #             {metric: window_metrics[model_name][metric] 
        #              for metric in window_metrics[model_name].keys()}
        #             for window_metrics in all_metrics
        #         ],
        #         'average_metrics': aggregate_metrics[model_name]
        #     }
            
        #     # Save to model-specific file
        #     with open(run_dir / f'{model_name.lower()}_metrics.json', 'w') as f:
        #         json.dump(model_metrics, f, indent=2)
        
        logger.info("\nAggregated metrics across all windows:")
        for model_name, metrics in aggregate_metrics.items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
        
        logger.info("Time-based CV experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run time-based cross-validation experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--n-windows', type=int, default=5, help='Number of validation windows')
    parser.add_argument('--window-hours', type=int, default=24, help='Size of each window in hours')
    
    args = parser.parse_args()
    run_time_cv(Path(args.config), args.n_windows, args.window_hours)
