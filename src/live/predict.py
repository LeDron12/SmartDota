#!/usr/bin/env python3

import os
import sys
import json
import yaml
import argparse
import logging
import catboost
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.base import BaseEstimator

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.data_new.fetch_stratz_matches import get_match_details, RateLimiter
from src.experiments.src.core.dataset_factory import DatasetFactory
from src.experiments.src.core.factory import TransformerFactory

logger = logging.getLogger(__name__)

def load_config(config_path: str = "predict_config.yaml") -> Dict[str, Any]:
    """Load prediction configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_pipeline(pipeline_dir: Path) -> TransformerFactory:
    """
    Load the serialized pipeline using TransformerFactory.
    
    Args:
        pipeline_dir: Path to the pipeline directory containing serialized transformers
        
    Returns:
        Loaded pipeline factory
    """
    # Load pipeline metadata
    metadata_path = pipeline_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Pipeline metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create factory and set pipeline directory
    factory = TransformerFactory()
    factory.set_pipeline_dir(pipeline_dir)
    
    # Load each transformer
    for transformer_info in metadata['transformers']:
        # Get the transformer class
        module_path = transformer_info['module']
        class_name = transformer_info['class']
        
        # Import the transformer class
        module = __import__(module_path, fromlist=[class_name])
        transformer_class = getattr(module, class_name)
        
        # Load the transformer
        transformer_path = pipeline_dir / transformer_info['path']
        transformer = transformer_class.load(str(transformer_path))
        # transformer.is_fitted = True
        
        # Add to pipeline
        factory._pipeline.append(transformer)
    
    factory._is_fitted = True
    return factory

def load_model_and_pipeline(config: Dict[str, Any]) -> tuple:
    """Load the trained model and transformation pipeline."""
    # Load CatBoost model
    model = catboost.CatBoostClassifier()
    model.load_model(config['model_path'])
    
    # Load pipeline
    pipeline_dir = Path(config['pipeline_dir'])
    pipeline = load_pipeline(pipeline_dir)
    
    return model, pipeline

def get_match_data(match_id: int) -> pd.DataFrame:
    """
    Get match data for prediction.
    
    Args:
        match_id: The ID of the match to get data for
        
    Returns:
        DataFrame containing match features ready for prediction
    """
    api_key = os.getenv('STRATZ_API_KEY')
    if not api_key:
        raise ValueError("STRATZ_API_KEY environment variable not set")
    rate_limiter = RateLimiter()
    query_name = 'get_match_by_id'
    path_to_gql = root_dir / 'src' / 'data_new' / 'graphQL'
    data, status = get_match_details(match_id, api_key, rate_limiter, query_name, path_to_gql)

    if status != 200:
        raise ValueError(f"Failed to fetch match data for match {match_id}")
    if not data or 'data' not in data or 'match' not in data['data']:
        raise ValueError(f"No match data found for match {match_id}")
    
    return [data['data']['match']]

def transform_features(data: pd.DataFrame, pipeline: TransformerFactory) -> pd.DataFrame:
    """
    Apply the transformation pipeline to the input data.
    
    Args:
        data: Raw match data
        pipeline: Pipeline factory with transform method
        
    Returns:
        Transformed features ready for model prediction
    """
    data = DatasetFactory().convert_matches_to_dataframe(data)
    logger.info(f"Transformed data: {data.head()}")
    logger.info(f"Transformed data shape: {data.shape}")
    logger.info(f"Transformed data columns: {data.columns}")
    transformed_data, _ = pipeline.transform(data)
    return transformed_data

def predict_match(match_id: int, model: BaseEstimator, pipeline: TransformerFactory, model_threshold: float) -> Dict[str, Any]:
    """
    Predict the outcome of a match.
    
    Args:
        match_id: The ID of the match to predict
        model: Trained model for prediction
        pipeline: Pipeline factory for feature transformation
        
    Returns:
        Dictionary containing prediction results
    """
    # Get match data
    match_data = get_match_data(match_id)
    
    # Transform features
    transformed_data = transform_features(match_data, pipeline)
    
    # Make prediction
    prediction = model.predict_proba(transformed_data)[0]
    radiant_win_prob = prediction[1]  # Probability of Radiant win
    
    return {
        'match_id': match_id,
        'radiant_win_probability': float(radiant_win_prob),
        'predicted_winner': 'Radiant' if radiant_win_prob > model_threshold else 'Dire',
        'confidence': float(abs(radiant_win_prob - model_threshold) / (1 - model_threshold))  # Scale to [0, 1] using threshold
    }

def main():
    """Main entry point for prediction."""
    parser = argparse.ArgumentParser(description="Predict Dota 2 match outcome")
    parser.add_argument("--match_id", type=int, required=True, help="Match ID to predict")
    parser.add_argument("--model_threshold", type=float, default=0.5, help="Model prediction threshold (default: 0.5)")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    if args.match_id:
        config['match_id'] = args.match_id
    if args.model_threshold:
        config['model_threshold'] = args.model_threshold
    
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline(config)
    
    # Get match ID from command line or use default
    match_id = config.get('match_id')
    model_threshold = config.get('model_threshold')
    if not match_id:
        raise ValueError("No match ID provided. Either pass it as a command line argument or set default_match_id in config.")
    
    # Make prediction
    result = predict_match(match_id, model, pipeline, model_threshold)
    
    # Print results
    logger.info("\nMatch Prediction Results:")
    logger.info(f"Match ID: {result['match_id']}")
    logger.info(f"Predicted Winner: {result['predicted_winner']}")
    logger.info(f"Radiant Win Probability: {result['radiant_win_probability']:.2%}")
    logger.info(f"Prediction Confidence: {result['confidence']:.2%}")

if __name__ == '__main__':
    main() 