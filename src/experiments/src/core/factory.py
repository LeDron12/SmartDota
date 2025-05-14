from typing import Dict, List, Type, Any, Optional
import pandas as pd
from .base import BaseTransformer
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TransformerFactory:
    """Factory class for creating and managing data transformers."""
    
    def __init__(self):
        """Initialize the transformer factory."""
        self._transformers: Dict[str, Type[BaseTransformer]] = {}
        self._pipeline: List[BaseTransformer] = []
        self._is_fitted = False
        self._pipeline_dir: Optional[Path] = None  # Add pipeline directory
    
    def register_transformer(self, name: str, transformer_class: Type[BaseTransformer]) -> None:
        """
        Register a transformer class with the factory.
        
        Args:
            name: Name to register the transformer under
            transformer_class: Transformer class to register
        """
        self._transformers[name] = transformer_class
        logger.info(f"Registered transformer: {name}")
    
    def create_pipeline(self, configs: List[Dict[str, Any]]) -> None:
        """
        Create a pipeline of transformers from configurations.
        
        Args:
            configs: List of transformer configurations
        """
        self._pipeline = []
        for config in configs:
            name = config.get('name')
            if name not in self._transformers:
                raise ValueError(f"Unknown transformer: {name}")
            
            transformer = self._transformers[name](config)
            if transformer.is_enabled:
                self._pipeline.append(transformer)
                logger.info(f"Added transformer to pipeline: {name}")
    
    def set_pipeline_dir(self, pipeline_dir: Path) -> None:
        """Set the directory for saving pipeline transformers."""
        self._pipeline_dir = pipeline_dir
        self._pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline(self) -> None:
        """Save all transformers in the pipeline to disk."""
        if not self._pipeline_dir:
            raise ValueError("Pipeline directory not set. Call set_pipeline_dir first.")
        
        # Save metadata about transformers
        metadata = {
            'transformers': [
                {
                    'name': transformer.name,
                    'module': transformer.__class__.__module__,
                    'class': transformer.__class__.__name__,
                    'path': f"{transformer.name}.pkl"
                }
                for transformer in self._pipeline
            ]
        }
        
        # Save metadata
        metadata_path = self._pipeline_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each transformer
        for transformer in self._pipeline:
            save_path = self._pipeline_dir / f"{transformer.name}.pkl"
            transformer.save(str(save_path))
            logger.info(f"Saved transformer {transformer.name} to {save_path}")
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the pipeline of transformers on the training data.
        
        Args:
            df: Training DataFrame
        """
        if not self._pipeline:
            logger.warning("No transformers in pipeline")
            return
        
        logger.info(f"Fitting pipeline on DataFrame with shape: {df.shape}")
        for transformer in self._pipeline:
            logger.info(f"Fitting transformer: {transformer.name}")
            transformer.fit(df.copy(deep=True))
        
        self._is_fitted = True
        logger.info("Pipeline fitting completed")
        
        # Save pipeline after fitting
        if self._pipeline_dir:
            self.save_pipeline()
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline of transformers to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self._pipeline:
            logger.warning("No transformers in pipeline")
            return df
        
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        logger.info(f"Transforming DataFrame with shape: {df.shape}")
        transformed_dfs = []
        categorical_features = []
        for transformer in self._pipeline:
            logger.info(f"Applying transformer: {transformer.name}")
            result_df = transformer.transform(df.copy(deep=True))
            logger.info(f"Result df shape: {result_df.shape}")
            logger.info(f"Added features by {transformer.name}: {result_df.columns}")
            logger.info(f"First few rows of transformed data:\n{result_df.head()}")
            transformed_dfs.append(result_df)
            categorical_features.extend(transformer.get_feature_names(categorical=True))

        # Skip the first transformer (DatasetConverter) and concatenate the rest
        # Reset index before concatenation to avoid any index-related issues
        transformed_dfs = [df.reset_index(drop=True) for df in transformed_dfs if not df.empty]
        final_df = pd.concat(transformed_dfs, axis=1)
        logger.info(f"Final DataFrame columns: {final_df.columns.tolist()}")
        return final_df, categorical_features
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features after transformation.
        
        Returns:
            List of feature names
        """
        feature_names = []
        for transformer in self._pipeline:
            feature_names.extend(transformer.get_feature_names())
        return feature_names 