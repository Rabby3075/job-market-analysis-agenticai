"""
JSON Helper Utility
Handles serialization of numpy and pandas data types for API responses
"""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas objects to JSON-serializable Python types
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle inf, -inf, and NaN values
        if np.isinf(obj):
            return None  # Convert inf to None
        elif np.isnan(obj):
            return None  # Convert NaN to None
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.NaT):
        return None
    
    # Handle datetime types
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    
    # Handle lists
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    
    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    
    # Handle sets
    elif isinstance(obj, set):
        return list(convert_to_serializable(item) for item in obj)
    
    # Handle other types that might have numpy values
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    
    # Return as-is if it's already serializable
    else:
        try:
            # Check for problematic values
            if isinstance(obj, float):
                if np.isinf(obj) or np.isnan(obj):
                    return None
            
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If it can't be serialized, convert to string
            return str(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely convert object to JSON string
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    serializable_obj = convert_to_serializable(obj)
    return json.dumps(serializable_obj, **kwargs)

def clean_analysis_results(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean analysis results to ensure they're JSON serializable
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Cleaned analysis results
    """
    if not isinstance(analysis, dict):
        return {}
    
    cleaned = {}
    
    for key, value in analysis.items():
        try:
            # Special handling for pandas DataFrames and Series
            if hasattr(value, 'to_dict'):
                if hasattr(value, 'columns'):  # DataFrame
                    cleaned[key] = value.to_dict('records')
                else:  # Series
                    cleaned[key] = value.to_dict()
            else:
                cleaned[key] = convert_to_serializable(value)
        except Exception as e:
            # If conversion fails, skip this key or convert to string
            cleaned[key] = f"Error converting {key}: {str(e)}"
    
    return cleaned
