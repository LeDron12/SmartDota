import pandas as pd
from typing import Dict, Any, List
import logging
from ..core.base import BaseTransformer

logger = logging.getLogger(__name__)

class DatasetConverter(BaseTransformer):
    """Transformer for converting raw match data to DataFrame format."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset converter.
        
        Args:
            config: Dictionary containing transformer configuration
        """
        super().__init__(config)
        self._feature_names = None
    
    def fit(self, df: pd.DataFrame) -> 'DatasetConverter':
        """
        Fit the transformer to the data.
        
        Args:
            df: Input DataFrame (raw match data)
            
        Returns:
            self for method chaining
        """
        # Create feature names based on the first match
        if len(df) > 0:
            first_match = df.iloc[0]
            self._feature_names = list(first_match.keys())
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw match data into a DataFrame.
        
        Args:
            df: Input DataFrame containing raw match data
            
        Returns:
            DataFrame with processed match data
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Convert to DataFrame if it's not already
        if not self._feature_names:
            self._feature_names = list(df.columns)
        
        # Ensure all required columns are present
        required_columns = {'match_id', 'radiant_team', 'dire_team', 'radiant_win'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Log data statistics
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        return pd.DataFrame()
    
    def get_feature_names(self, categorical: bool = False) -> List[str]:
        """
        Get the names of the features after transformation.
        
        Returns:
            List of feature names
        """
        if categorical:
            return []
        return self._feature_names if self._feature_names else [] 