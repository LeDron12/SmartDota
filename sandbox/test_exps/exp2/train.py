import logging
from pathlib import Path
import pandas as pd
from data_processor import DraftDataProcessor
from model_trainer import ModelTrainer
from metrics import plot_metrics_over_iterations, plot_threshold_metrics, save_metrics_comparison

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignore warnings about undefined metrics for plotting
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def main(base_path: Path):
    # Set up logging with absolute path
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(base_path / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set paths
    data_path = Path("/Users/ankamenskiy/SmartDota/cache")
    models_path = base_path / "results"
    graphics_path = models_path # / "graphics"
    models_path.mkdir(parents=True, exist_ok=True)
    graphics_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting draft win prediction experiment...")
    
    # Process data
    logger.info("\n=== Data Processing ===")
    processor = DraftDataProcessor(data_path)
    X_train, y_train, X_val, y_val = processor.get_processed_data()
    
    # Log data statistics
    logger.info("\nData Statistics:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Number of unique heroes: {len(pd.concat([X_train, X_val]).nunique())}")
    logger.info(f"Class balance (Radiant wins): {y_train.mean():.3f}")
    
    # Train and evaluate models
    logger.info("\n=== Model Training and Evaluation ===")
    trainer = ModelTrainer(save_path=models_path)
    trainer.X_val = X_val  # Store validation data for plotting
    trainer.y_val = y_val
    results = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)
    
    # Get best model
    best_model_name, best_metrics = trainer.get_best_model()
    
    # Create plots
    logger.info("\n=== Creating Visualization Plots ===")
    plot_metrics_over_iterations(trainer, graphics_path)
    plot_threshold_metrics(trainer, graphics_path)
    logger.info("Plots saved to graphics directory")
    
    # Create summary
    logger.info("\n=== Experiment Summary ===")
    logger.info(f"Best model: {best_model_name}")
    logger.info("Best model metrics:")
    for metric_name, value in best_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save metrics comparison
    save_metrics_comparison(results, models_path)

if __name__ == "__main__":
    base_path = Path("/Users/ankamenskiy/SmartDota/src/experiments/cursor_exps/exp2")
    main(base_path) 