from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import pickle
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseTransformer(ABC):
    """Base class for all data transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer with configuration.
        
        Args:
            config: Dictionary containing transformer configuration
        """
        self.config = config
        self.is_enabled = config.get('enabled', True)
        self.name = config.get('name', self.__class__.__name__)
        self._state = {}  # Dictionary to store transformer state
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the transformer and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> list:
        """
        Get the names of the features after transformation.
        
        Returns:
            List of feature names
        """
        return []
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the transformer.
        
        Returns:
            Dictionary containing transformer state
        """
        logger.debug(f"Getting state for transformer {self.name}")
        state = {}
        
        def _serialize_value(value: Any) -> Any:
            """Helper function to recursively serialize values."""
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_serialize_value(v) for v in value]
            elif isinstance(value, np.ndarray):
                return {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, pd.DataFrame):
                return {
                    'type': 'dataframe',
                    'data': value.to_dict(),
                    'columns': value.columns.tolist(),
                    'index': value.index.tolist()
                }
            elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                return value.item()  # Convert numpy scalar to Python scalar
            else:
                try:
                    return str(value)
                except:
                    logger.warning(f"Could not serialize value of type {type(value)}")
                    return None
        
        # Get all instance attributes
        for attr_name, attr_value in self.__dict__.items():
            # Skip private attributes and methods
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                continue
                
            # Skip methods and callables
            if callable(attr_value):
                continue
            
            # Skip config as it's already saved separately
            if attr_name == 'config':
                continue
                
            # Serialize the value
            serialized_value = _serialize_value(attr_value)
            if serialized_value is not None:
                state[attr_name] = serialized_value
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the transformer.
        
        Args:
            state: Dictionary containing transformer state
        """
        logger.debug(f"Setting state for transformer {self.name}")
        
        def _deserialize_value(value: Any) -> Any:
            """Helper function to recursively deserialize values."""
            if isinstance(value, dict):
                # Check for special types
                if 'type' in value:
                    if value['type'] == 'ndarray':
                        return np.array(value['data'], dtype=value['dtype'])
                    elif value['type'] == 'dataframe':
                        df = pd.DataFrame(value['data'], columns=value['columns'])
                        df.index = value['index']
                        return df
                
                # Regular dictionary
                return {k: _deserialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_deserialize_value(v) for v in value]
            else:
                return value
        
        for attr_name, attr_value in state.items():
            # Skip private attributes
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                continue
            
            # Deserialize the value
            deserialized_value = _deserialize_value(attr_value)
            setattr(self, attr_name, deserialized_value)
    
    def save(self, path: str) -> None:
        """
        Save transformer state to disk.
        
        Args:
            path: Path to save the transformer state
        """
        logger.info(f"Saving transformer {self.name} state to {path}")
        
        # Get serialized state
        state = self.get_state()
        
        # Save state and config
        save_dict = {
            'name': self.name,
            'config': self.config,
            'state': state
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Successfully saved transformer state to {path}")
        except Exception as e:
            logger.error(f"Failed to save transformer state to {path}: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'BaseTransformer':
        """
        Load transformer state from disk.
        
        Args:
            path: Path to load the transformer state from
            
        Returns:
            Loaded transformer instance
        """
        logger.info(f"Loading transformer state from {path}")
        
        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)
            logger.debug(f"Successfully loaded pickle file from {path}")
        except Exception as e:
            logger.error(f"Failed to load transformer state from {path}: {str(e)}")
            raise
        
        # Create new instance with saved config
        transformer = cls(save_dict['config'])
        logger.debug(f"Created new transformer instance with name: {transformer.name}")
        
        # Restore state
        transformer.set_state(save_dict['state'])
        transformer._is_fitted = True
        
        logger.info(f"Successfully loaded and restored transformer state from {path}")
        return transformer