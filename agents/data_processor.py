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
        # ABS recipe output directory
        self._abs_outdir = Path("data/preprocessed")
        self._abs_outdir.mkdir(parents=True, exist_ok=True)
    
    # ---------------- ABS-specific helpers (integrated) ----------------
    def _abs_read_data1(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name="Data1", header=0)
        # Notebook logic skips first 9 metadata rows
        df = df.iloc[9:].reset_index(drop=True)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _abs_num(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _abs_clean_industry(self, name: str) -> str:
        s = str(name).strip()
        if s.startswith("Standard Error of Job Vacancies"):
            m = re.search(r"Standard Error of Job Vacancies\s*;\s*(.*?)\s*;", s)
            return f"SE_{(m.group(1) if m else s)}"
        if s.startswith("Job Vacancies"):
            m = re.search(r"Job Vacancies\s*;\s*(.*?)\s*;", s)
            return (m.group(1) if m else s)
        return s

    def _abs_clean_sector(self, name: str, tag: str) -> str:
        s = str(name).strip()
        if s.startswith("Standard Error of Job Vacancies"):
            m = re.search(r"Standard Error of Job Vacancies\s*;\s*(?:Private|Public)?\s*;?\s*(.*?)\s*;", s)
            return f"SE_{(m.group(1) if m else s)}_{tag}"
        if s.startswith("Job Vacancies"):
            if "Seasonally Adjusted" in s:
                return f"Australia_{tag}_Seasonal"
            if "Trend" in s:
                return f"Australia_{tag}_Trend"
            m = re.search(r"Job Vacancies\s*;\s*(?:Private|Public)?\s*;?\s*(.*?)\s*;", s)
            return f"{(m.group(1) if m else s)}_{tag}"
        return s

    def _abs_split(self, v: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        se_cols = [c for c in v.columns if str(c).startswith("SE_")]
        vac = v[["Date"] + [c for c in v.columns if c not in se_cols]].copy()
        se = pd.concat([v[["Date"]], v[se_cols]], axis=1)
        return vac, se

    def _abs_melt(self, vac: pd.DataFrame, idname: str) -> pd.DataFrame:
        return vac.melt(id_vars="Date", var_name=idname, value_name="Vacancies_thousands")

    def _abs_detect_table_type(self, path: str) -> str:
        try:
            ix = pd.read_excel(path, sheet_name="Index", header=None)
            txt = " ".join(map(str, ix.fillna("").astype(str).head(8).values.ravel())).lower()
            if "private sector" in txt:
                return "private"
            if "public sector" in txt:
                return "public"
            if "industry" in txt:
                return "industry"
        except Exception:
            pass
        return "total"

    def _abs_save(self, prefix: str, vac: pd.DataFrame, se: pd.DataFrame, long: pd.DataFrame):
        vac_path = self._abs_outdir / f"{prefix}_vacancies_clean.csv"
        se_path = self._abs_outdir / f"{prefix}_standard_error.csv"
        long_path = self._abs_outdir / f"{prefix}_vacancies_clean_long.csv"
        try:
            vac.to_csv(str(vac_path), index=False)
            logger.info(f"ABS save: {vac_path}")
        except Exception as e:
            logger.warning(f"ABS save FAILED for {vac_path}: {e}")
        try:
            se.to_csv(str(se_path), index=False)
            logger.info(f"ABS save: {se_path}")
        except Exception as e:
            logger.warning(f"ABS save FAILED for {se_path}: {e}")
        try:
            long.to_csv(str(long_path), index=False)
            logger.info(f"ABS save: {long_path}")
        except Exception as e:
            logger.warning(f"ABS save FAILED for {long_path}: {e}")

    def _abs_proc_industry(self, path: str, interpolate: bool = False) -> pd.DataFrame:
        df = self._abs_read_data1(path).rename(columns=lambda c: self._abs_clean_industry(c))
        df = df.loc[:, ~df.columns.duplicated()]
        # Drop Series ID / Unit columns to mirror notebook filtering
        for drop_col in ("Series ID", "Unit"):
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
        if interpolate:
            df = df.set_index("Date").interpolate("time").reset_index()
        df = self._abs_num(df)
        vac, se = self._abs_split(df)
        long = self._abs_melt(vac, "Industry")
        self._abs_save("industry", vac, se, long)
        # alias used in notebooks
        alias_path = self._abs_outdir / "job_vacancies_clean.csv"
        try:
            vac.to_csv(str(alias_path), index=False)
            logger.info(f"ABS save (alias): {alias_path}")
        except Exception as e:
            logger.warning(f"ABS save FAILED (alias) for {alias_path}: {e}")
        return vac

    def _abs_proc_sector(self, path: str, tag: str, interpolate: bool = False) -> pd.DataFrame:
        df = self._abs_read_data1(path).rename(columns=lambda c: self._abs_clean_sector(c, tag))
        df = df.loc[:, ~df.columns.duplicated()]
        for drop_col in ("Series ID", "Unit"):
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
        df = df[df["Date"] >= "1983-11-01"].reset_index(drop=True)
        if interpolate:
            nums = df.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce").interpolate("linear")
            df[nums.columns] = nums
        df = self._abs_num(df)
        vac, se = self._abs_split(df)
        long = self._abs_melt(vac, "Region")
        self._abs_save(tag.lower(), vac, se, long)
        return vac

    def _abs_preprocess_workbook(self, path: str, interpolate: bool = False) -> str:
        kind = self._abs_detect_table_type(path)
        if kind == "industry":
            self._abs_proc_industry(path, interpolate=interpolate)
        elif kind == "private":
            self._abs_proc_sector(path, "Private", interpolate=interpolate)
        elif kind == "public":
            self._abs_proc_sector(path, "Public", interpolate=interpolate)
        else:
            self._abs_proc_sector(path, "Total", interpolate=interpolate)
        return kind
        
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
            
            # Clean and preprocess with safe fallback for ABS pipelines
            try:
                df_cleaned = self._clean_data(df)
                df_processed = self._preprocess_data(df_cleaned)
            except Exception as e:
                logger.error(f"Safe fallback: skipping generic cleaning due to error: {e}")
                df_processed = df
            
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
                                # Prefer tailored cleaning for ABS 'Data1' sheet where applicable
                                if sheet_name.strip().lower() == 'data1':
                                    processed_sheet = self._process_abs_data1_sheet(sheet_df, str(file_path))
                                else:
                                    processed_sheet = self._clean_data(sheet_df)
                                processed_sheets[sheet_name] = processed_sheet
                                logger.info(f"Processed sheet '{sheet_name}': {processed_sheet.shape}")
                            except Exception as e:
                                logger.warning(f"Could not process sheet '{sheet_name}': {str(e)}")
                                processed_sheets[sheet_name] = sheet_df  # Keep original if processing fails
                        
                        # Store processed sheets
                        self.multi_sheet_data[dataset_name_from_path] = processed_sheets
                        
                        # Return 'Data1' as the primary sheet if present, otherwise the first sheet
                        preferred_sheet_name = None
                        for candidate in ['Data1', 'data1', 'DATA1']:
                            if candidate in processed_sheets:
                                preferred_sheet_name = candidate
                                break
                        if preferred_sheet_name is None:
                            preferred_sheet_name = list(processed_sheets.keys())[0]
                        logger.info(
                            f"Loaded and processed {len(processed_sheets)} sheets separately, returning primary sheet: {preferred_sheet_name}"
                        )
                        # ABS recipes are already applied inside Data1 processing above.
                        # Skip a second pass here to avoid double-processing.
                        return processed_sheets[preferred_sheet_name], {'sheets': list(processed_sheets.keys())}
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
                    # Single-sheet files: skip ABS recipe here; Data1-specific logic
                    # will be applied when needed in the processing stage.
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

    def _process_abs_data1_sheet(self, df: pd.DataFrame, source_path: Optional[str] = None) -> pd.DataFrame:
        """Specialized cleaner for ABS 'Data1' sheets.

        Rules:
        - Keep existing column headers (first row already contains long names).
        - Drop metadata rows like 'Unit', 'Series Type', 'Data Type', etc.
        - Keep only rows from the first parsable date onward in the first column.
        - Parse first column to datetime; coerce all other columns to numeric.
        - Remove empty/placeholder columns that contain no meaningful data.
        """
        # Try to follow exact ABS recipe if source workbook path is known
        if source_path is not None:
            try:
                kind = self._abs_detect_table_type(source_path)
                base = self._abs_read_data1(source_path)
                if kind == 'industry':
                    base = base.rename(columns=lambda c: self._abs_clean_industry(c))
                elif kind == 'private':
                    base = base.rename(columns=lambda c: self._abs_clean_sector(c, 'Private'))
                elif kind == 'public':
                    base = base.rename(columns=lambda c: self._abs_clean_sector(c, 'Public'))
                else:
                    base = base.rename(columns=lambda c: self._abs_clean_sector(c, 'Total'))
                base = base.loc[:, ~base.columns.duplicated()]
                for drop_col in ("Series ID", "Unit"):
                    if drop_col in base.columns:
                        base = base.drop(columns=[drop_col])
                if kind in {'private','public','total'}:
                    base = base[base["Date"] >= "1983-11-01"].reset_index(drop=True)
                base = self._abs_num(base)
                vac, _ = self._abs_split(base)
                return vac
            except Exception:
                pass
        working_df = df.copy()

        # Drop fully empty rows/cols first
        working_df = working_df.dropna(how='all').dropna(axis=1, how='all')

        if working_df.shape[1] == 0:
            return working_df

        first_col_name = working_df.columns[0]

        # Drop well-known ABS metadata rows in the first column
        metadata_labels = {
            'unit', 'series type', 'data type', 'frequency', 'collection month',
            'series start', 'series end', 'no. obs', 'no. obs.', 'series id'
        }
        mask_meta = working_df[first_col_name].astype(str).str.strip().str.lower().isin(metadata_labels)
        working_df = working_df[~mask_meta]

        # Keep only rows from the first date onward
        date_series = pd.to_datetime(working_df[first_col_name], errors='coerce', dayfirst=True)
        if date_series.notna().any():
            first_date_idx = date_series.first_valid_index()
            if first_date_idx is not None:
                working_df = working_df.loc[first_date_idx:]
                date_series = pd.to_datetime(working_df[first_col_name], errors='coerce', dayfirst=True)

        # Assign parsed dates back
        working_df[first_col_name] = date_series

        # Clean column names
        working_df.columns = [self._clean_column_name(c) for c in working_df.columns]

        # Convert remaining columns to numeric where possible
        for col in working_df.columns[1:]:
            working_df[col] = pd.to_numeric(working_df[col], errors='coerce')

        # Remove empty columns BEFORE handling missing values
        # This is crucial for ABS data where many columns are completely empty
        working_df = self._remove_empty_columns_abs(working_df)

        # Standard cleaning pipeline
        working_df = self._handle_missing_values(working_df)
        working_df = self._preprocess_data(working_df)

        return working_df
    
    def _remove_empty_columns_abs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty/placeholder columns from ABS Data1 sheets.
        
        This method is specifically designed for ABS job vacancy data where
        many columns contain placeholder data that should be removed.
        
        Based on the user's observation that columns B-I and M-R are empty
        in the Excel files, this method identifies and removes such columns.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with empty/placeholder columns removed
        """
        if df.empty or df.shape[1] <= 1:
            return df
            
        columns_to_remove = []
        
        for col in df.columns:
            # Skip the first column (usually dates/IDs)
            if col == df.columns[0]:
                continue
                
            col_data = df[col]
            
            # Check if column is completely empty (all NaN)
            if col_data.isna().all():
                columns_to_remove.append(col)
                logger.info(f"Removing completely empty column: {col}")
                continue
            
            # Check if column has very few non-null values (< 5% of total rows)
            non_null_count = col_data.notna().sum()
            total_rows = len(col_data)
            non_null_ratio = non_null_count / total_rows if total_rows > 0 else 0
            
            if non_null_ratio < 0.05:  # Less than 5% non-null values
                columns_to_remove.append(col)
                logger.info(f"Removing nearly empty column: {col} ({non_null_count}/{total_rows} non-null values, {non_null_ratio:.1%} ratio)")
                continue
            
            # For ABS job vacancy data, check for columns with very low variation
            # that might be placeholder data filled during processing
            non_null_data = col_data.dropna()
            if len(non_null_data) > 20:  # Only check if we have enough data
                unique_values = non_null_data.nunique()
                unique_ratio = unique_values / len(non_null_data)
                
                # Check for columns with very low variation (likely placeholder data)
                if unique_ratio < 0.15:  # Less than 15% unique values
                    columns_to_remove.append(col)
                    logger.info(f"Removing low-variation ABS column: {col} ({unique_values}/{len(non_null_data)} unique values, {unique_ratio:.1%} ratio)")
                    continue
        
        # Remove identified columns
        if columns_to_remove:
            df_cleaned = df.drop(columns=columns_to_remove)
            logger.info(f"Removed {len(columns_to_remove)} empty/placeholder columns from ABS data: {columns_to_remove}")
            return df_cleaned
        
        return df
    
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
        
        return df_processed
    
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
                    
                    # Save individual sheet as CSV (ensure dates are ISO strings to avoid #### in Excel)
                    sheet_csv_path = os.path.join(multi_sheet_dir, f"{clean_sheet_name}.csv")
                    try:
                        for col in sheet_df.columns:
                            if np.issubdtype(sheet_df[col].dtype, np.datetime64):
                                sheet_df[col] = pd.to_datetime(sheet_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    except Exception:
                        pass
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
