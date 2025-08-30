"""
Data Processing Agent
Handles data cleaning, validation, and preprocessing
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessorAgent:
    """Agent responsible for data preprocessing and cleaning"""
    
    def __init__(self):
        self.processed_data = {}
        self.data_summary = {}
        self.multi_sheet_data = {}  # Store multiple sheets separately
        
    def process_dataset(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Process a dataset file and return cleaned data with summary
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Tuple of (cleaned_dataframe, processing_summary)
        """
        try:
            logger.info(f"Processing dataset: {file_path}")
            
            # Load data based on file type
            load_result = self._load_data(file_path)
            if load_result is None:
                return None, {}
            
            # Handle both single DataFrame and tuple (DataFrame, summary) returns
            if isinstance(load_result, tuple):
                df, load_summary = load_result
            else:
                df = load_result
                load_summary = {}
            
            if df is None:
                return None, {}
            
            # Store original data info
            original_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # Clean and preprocess
            df_cleaned = self._clean_data(df)
            df_processed = self._preprocess_data(df_cleaned)
            
            # Generate summary
            summary = self._generate_summary(df_processed, original_info)
            
            # Merge with load summary if available
            if load_summary:
                summary.update(load_summary)
            
            # Store processed data
            dataset_name = Path(file_path).stem
            self.processed_data[dataset_name] = df_processed
            self.data_summary[dataset_name] = summary
            
            # Save preprocessed data as CSV
            self._save_preprocessed_data(dataset_name, df_processed)
            
            logger.info(f"Successfully processed {dataset_name}")
            return df_processed, summary
            
        except Exception as e:
            logger.error(f"Error processing dataset {file_path}: {str(e)}")
            return None, {}
    
    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats with support for multiple sheets"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV data: {df.shape}")
                # Store as single sheet for consistency
                dataset_name_from_path = Path(file_path).stem
                self.multi_sheet_data[dataset_name_from_path] = {'Sheet1': df}
                return df, {}
                
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Check if Excel file has multiple sheets
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) > 1:
                    logger.info(f"Excel file has {len(sheet_names)} sheets: {sheet_names}")
                    
                    # Load all sheets separately and store them
                    all_sheets = {}
                    for sheet_name in sheet_names:
                        try:
                            sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                            if not sheet_df.empty:
                                all_sheets[sheet_name] = sheet_df
                                logger.info(f"Loaded sheet '{sheet_name}': {sheet_df.shape}")
                        except Exception as e:
                            logger.warning(f"Could not load sheet '{sheet_name}': {str(e)}")
                            continue
                    
                    if all_sheets:
                        # Store sheets separately for tabbed interface
                        # Get dataset name from file path
                        dataset_name_from_path = Path(file_path).stem
                        self.multi_sheet_data[dataset_name_from_path] = all_sheets
                        
                        # Process each sheet individually and store processed versions
                        processed_sheets = {}
                        for sheet_name, sheet_df in all_sheets.items():
                            try:
                                processed_sheet = self._clean_data(sheet_df)
                                processed_sheets[sheet_name] = processed_sheet
                                logger.info(f"Processed sheet '{sheet_name}': {processed_sheet.shape}")
                            except Exception as e:
                                logger.warning(f"Could not process sheet '{sheet_name}': {str(e)}")
                                processed_sheets[sheet_name] = sheet_df  # Keep original if processing fails
                        
                        # Store processed sheets
                        self.multi_sheet_data[dataset_name_from_path] = processed_sheets
                        
                        # Return the first sheet as main data (for backward compatibility)
                        first_sheet_name = list(processed_sheets.keys())[0]
                        logger.info(f"Loaded and processed {len(processed_sheets)} sheets separately, returning first sheet: {first_sheet_name}")
                        return processed_sheets[first_sheet_name], {'sheets': list(processed_sheets.keys())}
                    else:
                        logger.error("No valid sheets found in Excel file")
                        return None, {}
                else:
                    # Single sheet Excel file
                    df = pd.read_excel(file_path)
                    logger.info(f"Loaded single sheet Excel data: {df.shape}")
                    # Store as single sheet for consistency
                    dataset_name_from_path = Path(file_path).stem
                    self.multi_sheet_data[dataset_name_from_path] = {'Sheet1': df}
                    return df, {}
                    
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
                logger.info(f"Loaded JSON data: {df.shape}")
                # Store as single sheet for consistency
                dataset_name_from_path = Path(file_path).stem
                self.multi_sheet_data[dataset_name_from_path] = {'Sheet1': df}
                return df, {}
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None, {}
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data"""
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df_clean.columns = [self._clean_column_name(col) for col in df_clean.columns]
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Clean data types
        df_clean = self._clean_data_types(df_clean)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column names for consistency"""
        if pd.isna(col_name):
            return 'unnamed_column'
        
        # Convert to string and clean
        col_str = str(col_name).strip()
        
        # Remove special characters and replace spaces with underscores
        col_clean = re.sub(r'[^\w\s]', '', col_str)
        col_clean = re.sub(r'\s+', '_', col_clean)
        col_clean = col_clean.lower()
        
        # Ensure it's not empty
        if not col_clean:
            col_clean = 'unnamed_column'
        
        return col_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)
            
            if missing_pct > 0.5:  # More than 50% missing
                # Drop column if too many missing values
                df_clean = df_clean.drop(columns=[col])
                logger.info(f"Dropped column {col} due to {missing_pct:.1%} missing values")
                
            elif missing_pct > 0:  # Some missing values
                # Fill based on data type
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Numeric: fill with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Categorical: fill with mode
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and optimize data types"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            # Try to convert to numeric if possible
            if df_clean[col].dtype == 'object':
                # Check if it's actually numeric
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If most values are numeric, convert
                    non_null_count = numeric_series.notna().sum()
                    if non_null_count / len(df_clean) > 0.8:  # 80% numeric
                        df_clean[col] = numeric_series
                        logger.info(f"Converted column {col} to numeric")
                
                # Check for date columns
                elif self._looks_like_date(df_clean[col]):
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        logger.info(f"Converted column {col} to datetime")
                    except:
                        pass
        
        return df_clean
    
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if a series looks like it contains dates"""
        if series.dtype != 'object':
            return False
        
        # Sample some values to check
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Check if values contain common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',  # Month names
        ]
        
        for pattern in date_patterns:
            if sample.astype(str).str.contains(pattern, regex=True, case=False).any():
                return True
        
        return False
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional preprocessing steps"""
        df_processed = df.copy()
        
        # Standardize text columns
        text_columns = df_processed.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].astype(str).str.strip()
        
        # Handle outliers in numeric columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_processed[col] = self._handle_outliers(df_processed[col])
        
        return df_processed
    
    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """Handle outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        series_clean = series.copy()
        series_clean[series_clean < lower_bound] = lower_bound
        series_clean[series_clean > upper_bound] = upper_bound
        
        return series_clean
    
    def _generate_summary(self, df: pd.DataFrame, original_info: Dict) -> Dict:
        """Generate comprehensive data summary"""
        summary = {
            'dataset_name': 'processed_dataset',
            'original_shape': original_info['shape'],
            'processed_shape': df.shape,
            'columns': {
                'total': len(df.columns),
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'statistics': {}
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Convert numpy types to Python types for JSON serialization
            numeric_stats = df[numeric_cols].describe().to_dict()
            summary['statistics']['numeric'] = {}
            for col, stats in numeric_stats.items():
                summary['statistics']['numeric'][col] = {k: float(v) if pd.notna(v) else None for k, v in stats.items()}
        
        # Add value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['statistics']['categorical'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 columns
                value_counts = df[col].value_counts().head(10)
                summary['statistics']['categorical'][col] = {str(k): int(v) for k, v in value_counts.items()}
        
        return summary
    
    def get_processed_data(self, dataset_name: str = None) -> Dict:
        """Get processed data by name or all data"""
        if dataset_name:
            return {dataset_name: self.processed_data.get(dataset_name)}
        return self.processed_data
    
    def get_data_summary(self, dataset_name: str = None) -> Dict:
        """Get data summary by name or all summaries"""
        if dataset_name and dataset_name in self.processed_data:
            df = self.processed_data[dataset_name]
            # Generate summary on demand
            original_info = {
                'shape': df.shape,
                'name': dataset_name
            }
            return self._generate_summary(df, original_info)
        return {}
    
    def get_processed_dataframe(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get processed dataframe by name"""
        return self.processed_data.get(dataset_name)
    
    def get_sheet_data(self, dataset_name: str, sheet_name: str = None) -> Optional[pd.DataFrame]:
        """Get specific sheet data for a dataset"""
        if dataset_name in self.multi_sheet_data:
            if sheet_name:
                return self.multi_sheet_data[dataset_name].get(sheet_name)
            else:
                # Return first sheet if no specific sheet requested
                first_sheet = list(self.multi_sheet_data[dataset_name].keys())[0]
                return self.multi_sheet_data[dataset_name][first_sheet]
        return None
    
    def get_all_sheets(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """Get all sheets for a dataset"""
        # Try exact match first
        if dataset_name in self.multi_sheet_data:
            return self.multi_sheet_data[dataset_name]
        
        # Try name variations (handle space vs underscore mismatch)
        possible_names = [
            dataset_name,  # Original name
            dataset_name.replace(' ', '_'),  # Replace spaces with underscores
            dataset_name.replace('_', ' '),  # Replace underscores with spaces
            dataset_name.replace(' ', ''),   # Remove all spaces
            dataset_name.replace('_', ''),   # Remove all underscores
        ]
        
        for name_variant in possible_names:
            if name_variant in self.multi_sheet_data:
                return self.multi_sheet_data[name_variant]
        
        return {}
    
    def _save_preprocessed_data(self, dataset_name: str, df: pd.DataFrame):
        """Save preprocessed data as CSV in data/preprocessed folder"""
        try:
            # Create preprocessed directory if it doesn't exist
            os.makedirs("data/preprocessed", exist_ok=True)
            
            # If this dataset has multiple sheets, save each sheet as separate CSV
            if dataset_name in self.multi_sheet_data:
                multi_sheet_dir = os.path.join("data/preprocessed", f"{dataset_name}_sheets")
                os.makedirs(multi_sheet_dir, exist_ok=True)
                
                # Save each sheet as a separate CSV file
                for sheet_name, sheet_df in self.multi_sheet_data[dataset_name].items():
                    # Clean sheet name for filename (remove special characters)
                    clean_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_sheet_name = clean_sheet_name.replace(' ', '_')
                    
                    # Save individual sheet as CSV
                    sheet_csv_path = os.path.join(multi_sheet_dir, f"{clean_sheet_name}.csv")
                    sheet_df.to_csv(sheet_csv_path, index=False)
                    logger.info(f"Saved sheet '{sheet_name}' to: {sheet_csv_path}")
                    
                # Save a summary of all sheets
                summary_file = os.path.join(multi_sheet_dir, "_sheets_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Total Sheets: {len(self.multi_sheet_data[dataset_name])}\n")
                    f.write("Sheets:\n")
                    for sheet_name, sheet_df in self.multi_sheet_data[dataset_name].items():
                        f.write(f"  - {sheet_name}: {sheet_df.shape[0]} rows, {sheet_df.shape[1]} columns\n")
                logger.info(f"Saved sheets summary to: {summary_file}")
                    
        except Exception as e:
            logger.error(f"Error saving preprocessed data for {dataset_name}: {str(e)}")
    
    def has_multiple_sheets(self, dataset_name: str) -> bool:
        """Check if a dataset has multiple sheets"""
        # Try exact match first
        if dataset_name in self.multi_sheet_data:
            return len(self.multi_sheet_data[dataset_name]) > 1
        
        # Try name variations (handle space vs underscore mismatch)
        possible_names = [
            dataset_name,  # Original name
            dataset_name.replace(' ', '_'),  # Replace spaces with underscores
            dataset_name.replace('_', ' '),  # Replace underscores with spaces
            dataset_name.replace(' ', ''),   # Remove all spaces
            dataset_name.replace('_', ''),   # Remove all underscores
        ]
        
        for name_variant in possible_names:
            if name_variant in self.multi_sheet_data:
                return len(self.multi_sheet_data[name_variant]) > 1
        
        return False
