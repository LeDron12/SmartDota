import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import yaml
import json
import shutil
import copy
from typing import Dict, Any, List, Type, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.experiments.src.core.factory import TransformerFactory
from src.experiments.src.core.dataset_factory import DatasetFactory
from src.experiments.src.transformers.hero_features import HeroFeaturesTransformer
from src.experiments.src.transformers.dataset_converter import DatasetConverter
from src.experiments.src.transformers.starz_dataset_converter import StarzDatasetConverter
from src.experiments.src.transformers.hero_stats_transformer import HeroStatsTransformer
from src.experiments.src.transformers.player_stats_transformer import PlayerStatsTransformer
from src.experiments.src.transformers.player_hero_stats_transformer import PlayerHeroStatsTransformer
from src.experiments.src.transformers.player_ratings_transformer import PlayerRatingsTransformer
from src.experiments.src.utils.config import load_config, validate_transformer_configs
from src.experiments.src.core.model_trainer import ModelTrainer


def setup_logging(run_dir: Path) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        run_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(run_dir / 'experiment.log')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_run_directory(config_path: Path) -> Path:
    """
    Create directory structure for the experiment run.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Path to the run directory
    """
    # Get config name from path
    config_name = config_path.stem
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    
    # Create directory structure
    run_dir = project_root / 'src' / 'experiments' / 'runs' / config_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir

def save_config(config: Dict[str, Any], run_dir: Path) -> None:
    """
    Save configuration to the run directory.
    
    Args:
        config: Configuration dictionary
        run_dir: Directory to save the configuration
    """
    # Save as YAML
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save as JSON for easier parsing
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

def get_available_transformers() -> Dict[str, Type]:
    """
    Get dictionary of available transformers.
    
    Returns:
        Dictionary mapping transformer names to their classes
    """
    return {
        'DatasetConverter': DatasetConverter,
        'StarzDatasetConverter': StarzDatasetConverter,
        'HeroFeaturesTransformer': HeroFeaturesTransformer,
        'HeroStatsTransformer': HeroStatsTransformer,
        'PlayerStatsTransformer': PlayerStatsTransformer,
        'PlayerHeroStatsTransformer': PlayerHeroStatsTransformer,
        'PlayerRatingsTransformer': PlayerRatingsTransformer
        # Add more transformers here as they are implemented
    }

def save_transformed_data(
        train_df: pd.DataFrame, test_df: pd.DataFrame, labels_train: pd.Series, labels_test: pd.Series, 
        run_dir: Path, logger: logging.Logger, sample_size: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save transformed data with train/test split.
    
    Args:
        df: Transformed DataFrame
        run_dir: Directory to save data
        logger: Logger instance
        sample_size: Number of samples to save
        train_size: Proportion of data to use for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Splitting data into train and test sets...")

    train_df = copy.deepcopy(train_df)
    test_df = copy.deepcopy(test_df)
    train_df['radiant_win'] = labels_train
    test_df['radiant_win'] = labels_test
    
    # Log split statistics
    logger.info("\nData split statistics:")
    logger.info(f"Training set: {len(train_df)} matches")
    logger.info(f"Test set: {len(test_df)} matches")
    logger.info(f"Training set win rate: {train_df['radiant_win'].mean():.2%}")
    logger.info(f"Test set win rate: {test_df['radiant_win'].mean():.2%}")
    
    # Save to Excel with multiple sheets
    logger.info("Saving transformed data...")
    with pd.ExcelWriter(run_dir / 'transformed_data_sample.xlsx') as writer:
        train_df.sample(sample_size, random_state=42).to_excel(writer, sheet_name='Train', index=False)
        test_df.sample(sample_size, random_state=42).to_excel(writer, sheet_name='Test', index=False)
    
    logger.info(f"Saved transformed data to {run_dir / 'transformed_data.xlsx'}")
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")


def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Run a Dota 2 data processing experiment')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    config_path = Path(args.config)
    
    # Create run directory
    run_dir = create_run_directory(config_path)
    
    # Set up logging
    logger = setup_logging(run_dir)
    logger.info(f"Starting experiment with config: {config_path}")
    logger.info(f"Run directory: {run_dir}")
    
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate dataset configuration
        if 'dataset' not in config:
            raise ValueError("Dataset configuration is required")
        
        # Validate transformer configurations
        if 'transformers' not in config:
            raise ValueError("Transformer configurations are required")
        validate_transformer_configs(config['transformers'])
        
        # Save configuration
        save_config(config, run_dir)
        
        # Create dataset factory and load data
        dataset_factory = DatasetFactory()
        raw_df, labels = dataset_factory.load_dataset(config['dataset'], project_root)
        
        # Split into train and test sets using sklearn
        
        df_train, df_test, labels_train, labels_test = train_test_split(
            raw_df, labels, test_size=0.01, shuffle=False # Time based split
        )

        # ------------------------------

        # # Get max start time from the dataset
        # max_start_time = pd.Timestamp(raw_df['start_time'].max(), unit='s')
        
        # # Calculate cutoff time (6 hours before max time)
        # cutoff_time = max_start_time - pd.Timedelta(hours=3)
        
        # # Split data based on time
        # raw_df['pd_start_time'] = pd.to_datetime(raw_df['start_time'], unit='s')
        # df_train = raw_df[raw_df['pd_start_time'] <= cutoff_time]
        # df_test = raw_df[raw_df['pd_start_time'] > cutoff_time]
        
        # # Get corresponding labels
        # labels_train = [label for i, label in enumerate(labels) if pd.Timestamp(raw_df.iloc[i]['start_time'], unit='s') <= cutoff_time]
        # labels_test = [label for i, label in enumerate(labels) if pd.Timestamp(raw_df.iloc[i]['start_time'], unit='s') > cutoff_time]
        
        # logger.info(f"Time-based split - Train cutoff: {cutoff_time}, Max time: {max_start_time}")
        # logger.info(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")

        # ------------------------------

        # Create transformer factory and register available transformers
        transformer_factory = TransformerFactory()
        available_transformers = get_available_transformers()
        
        for name, transformer_class in available_transformers.items():
            transformer_factory.register_transformer(name, transformer_class)
        
        # Create pipeline from config
        transformer_factory.create_pipeline(config['transformers'])

        logger.info(f"Train data shape before transformation: {df_train.shape}")
        logger.info(f"Test data shape before transformation: {df_test.shape}")
        logger.info(f"Train labels shape before transformation: {len(labels_train)}")
        logger.info(f"Test labels shape before transformation: {len(labels_test)}")

        df_train['radiant_win'] = labels_train # for stats calculations
        df_test['radiant_win'] = labels_test # for stats calculations
        
        # Process data
        logger.info("Processing data through transformer pipeline")
        transformed_train_df, categorical_features = transformer_factory.fit_transform(df_train)
        transformed_test_df, _ = transformer_factory.transform(df_test)

        logger.info(f"Train data shape after transformation: {transformed_train_df.shape}")
        logger.info(f"Test data shape after transformation: {transformed_test_df.shape}")
        logger.info(f"Train labels shape after transformation: {len(labels_train)}")
        logger.info(f"Test labels shape after transformation: {len(labels_test)}")
        
        assert len(transformed_train_df) == len(labels_train)
        assert len(transformed_test_df) == len(labels_test)
        
        # Save transformed data and make a train/test split
        save_transformed_data(transformed_train_df, transformed_test_df, labels_train, labels_test, run_dir, logger)
        
        # Log feature statistics
        logger.info("\nFeature Statistics:")
        logger.info(f"Total features: {len(transformer_factory.get_feature_names())}")
        
        # Train models
        logger.info("\n=== Model Training and Evaluation ===")
        model_trainer = ModelTrainer(
            save_path=run_dir / "models",
            config=config,
            categorical_features=categorical_features
        )

        assert 'radiant_win' not in transformed_train_df.columns
        assert 'radiant_win' not in transformed_test_df.columns
        
        # Train and evaluate models
        metrics = model_trainer.train_and_evaluate(transformed_train_df, labels_train, transformed_test_df, labels_test)
        
        # Save best model
        model_trainer.save_best_model("")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 