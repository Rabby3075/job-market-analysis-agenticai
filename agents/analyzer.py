"""
Data Analyzer Agent
Performs analysis and generates insights from job market data
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobMarketAnalyzer:
    """Agent responsible for analyzing job market data and generating insights"""
    
    def __init__(self):
        self.analysis_results = {}
        self.insights = {}
        
    def analyze_job_market_data(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Perform comprehensive analysis of job market data
        
        Args:
            df: Processed dataframe
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing analysis results and insights
        """
        try:
            logger.info(f"Analyzing job market data: {dataset_name}")
            
            analysis = {
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'overview': self._generate_overview(df),
                'trends': self._analyze_trends(df),
                'geographic_analysis': self._analyze_geography(df),
                'industry_analysis': self._analyze_industry(df),
                'sector_analysis': self._analyze_sector(df),
                'key_insights': [],
                'recommendations': []
            }
            
            # Generate insights based on analysis
            analysis['key_insights'] = self._generate_insights(analysis)
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Store results
            self.analysis_results[dataset_name] = analysis
            
            logger.info(f"Analysis completed for {dataset_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {str(e)}")
            return {}
    
    def _generate_overview(self, df: pd.DataFrame) -> Dict:
        """Generate high-level overview of the data"""
        overview = {
            'total_records': len(df),
            'time_period': self._extract_time_period(df),
            'data_coverage': self._assess_data_coverage(df),
            'key_metrics': self._extract_key_metrics(df)
        }
        return overview
    
    def _extract_time_period(self, df: pd.DataFrame) -> Dict:
        """Extract time period information from the data"""
        time_info = {
            'start_date': None,
            'end_date': None,
            'period_type': 'unknown'
        }
        
        # Look for date columns
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            for col in date_columns:
                if df[col].notna().any():
                    time_info['start_date'] = df[col].min().isoformat()
                    time_info['end_date'] = df[col].max().isoformat()
                    time_info['period_type'] = 'date_range'
                    break
        
        # Look for period columns (e.g., "May-2025", "Q1 2025")
        period_columns = [col for col in df.columns if any(word in col.lower() for word in ['period', 'quarter', 'month', 'year'])]
        if period_columns:
            time_info['period_type'] = 'categorical_periods'
            time_info['unique_periods'] = df[period_columns[0]].nunique()
        
        return time_info
    
    def _assess_data_coverage(self, df: pd.DataFrame) -> Dict:
        """Assess the coverage and quality of the data"""
        coverage = {
            'total_columns': len(df.columns),
            'complete_records': df.dropna().shape[0],
            'completeness_rate': (df.dropna().shape[0] / len(df)) * 100,
            'column_completeness': {}
        }
        
        for col in df.columns:
            coverage['column_completeness'][col] = {
                'non_null_count': df[col].notna().sum(),
                'completeness_rate': (df[col].notna().sum() / len(df)) * 100
            }
        
        return coverage
    
    def _extract_key_metrics(self, df: pd.DataFrame) -> Dict:
        """Extract key metrics from the data"""
        metrics = {}
        
        # Look for job vacancy numbers
        vacancy_columns = [col for col in df.columns if any(word in col.lower() for word in ['vacancy', 'job', 'position', 'opening'])]
        if vacancy_columns:
            for col in vacancy_columns:
                if df[col].dtype in ['int64', 'float64']:
                    metrics[f'{col}_total'] = int(df[col].sum()) if pd.notna(df[col].sum()) else 0
                    metrics[f'{col}_average'] = float(df[col].mean()) if pd.notna(df[col].mean()) else 0.0
                    metrics[f'{col}_median'] = float(df[col].median()) if pd.notna(df[col].median()) else 0.0
        
        # Look for percentage changes
        change_columns = [col for col in df.columns if any(word in col.lower() for word in ['change', 'growth', 'increase', 'decrease'])]
        if change_columns:
            for col in change_columns:
                if df[col].dtype in ['int64', 'float64']:
                    mean_val = df[col].mean()
                    if pd.notna(mean_val):
                        metrics[f'{col}_average'] = float(mean_val)
                        metrics[f'{col}_trend'] = 'positive' if mean_val > 0 else 'negative'
        
        return metrics
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in the data"""
        trends = {
            'temporal_trends': {},
            'growth_patterns': {},
            'seasonality': {}
        }
        
        # Analyze numeric columns for trends
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(df) > 1:  # Need at least 2 points for trend
                # Simple trend analysis
                values = df[col].dropna()
                if len(values) > 1:
                    # Calculate simple linear trend
                    x = np.arange(len(values))
                    if len(x) > 1:
                        slope = np.polyfit(x, values, 1)[0]
                        trends['temporal_trends'][col] = {
                            'slope': float(slope),
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                            'trend_strength': float(abs(slope))
                        }
        
        return trends
    
    def _analyze_geography(self, df: pd.DataFrame) -> Dict:
        """Analyze geographic distribution of job vacancies"""
        geography = {
            'states_territories': {},
            'regional_patterns': {},
            'geographic_concentration': {}
        }
        
        # Look for geographic columns
        geo_columns = [col for col in df.columns if any(word in col.lower() for word in ['state', 'territory', 'region', 'area', 'nsw', 'vic', 'qld', 'sa', 'wa', 'tas', 'nt', 'act'])]
        
        if geo_columns:
            for col in geo_columns:
                if df[col].dtype == 'object':
                    value_counts = df[col].value_counts()
                    geography['states_territories'][col] = {
                        'unique_locations': len(value_counts),
                        'top_locations': value_counts.head(5).to_dict(),
                        'distribution': value_counts.to_dict()
                    }
                    
                    # Calculate concentration
                    total = value_counts.sum()
                    if total > 0:
                        concentration = float((value_counts.max() / total) * 100)
                        geography['geographic_concentration'][col] = {
                            'most_concentrated': str(value_counts.index[0]),
                            'concentration_percentage': concentration
                        }
        
        return geography
    
    def _analyze_industry(self, df: pd.DataFrame) -> Dict:
        """Analyze industry distribution of job vacancies"""
        industry = {
            'industry_breakdown': {},
            'top_industries': {},
            'industry_trends': {}
        }
        
        # Look for industry columns
        industry_columns = [col for col in df.columns if any(word in col.lower() for word in ['industry', 'sector', 'field', 'manufacturing', 'health', 'education', 'construction'])]
        
        if industry_columns:
            for col in industry_columns:
                if df[col].dtype == 'object':
                    value_counts = df[col].value_counts()
                    industry['industry_breakdown'][col] = {
                        'total_industries': len(value_counts),
                        'top_10_industries': value_counts.head(10).to_dict(),
                        'industry_distribution': value_counts.to_dict()
                    }
                    
                    # Identify top industries
                    top_5 = value_counts.head(5)
                    industry['top_industries'][col] = {
                        'names': top_5.index.tolist(),
                        'values': top_5.values.tolist()
                    }
        
        return industry
    
    def _analyze_sector(self, df: pd.DataFrame) -> Dict:
        """Analyze public vs private sector patterns"""
        sector = {
            'sector_breakdown': {},
            'sector_comparison': {},
            'sector_trends': {}
        }
        
        # Look for sector columns
        sector_columns = [col for col in df.columns if any(word in col.lower() for word in ['sector', 'public', 'private', 'government'])]
        
        if sector_columns:
            for col in sector_columns:
                if df[col].dtype == 'object':
                    value_counts = df[col].value_counts()
                    sector['sector_breakdown'][col] = {
                        'sectors': value_counts.to_dict(),
                        'dominant_sector': value_counts.index[0] if len(value_counts) > 0 else None
                    }
                    
                    # Compare sectors if we have numeric data
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        sector_comparison = {}
                        for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                            sector_means = df.groupby(col)[num_col].mean()
                            sector['sector_comparison'][f'{col}_vs_{num_col}'] = {str(k): float(v) for k, v in sector_means.to_dict().items() if pd.notna(v)}
        
        return sector
    
    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Overview insights
        if 'overview' in analysis:
            overview = analysis['overview']
            if 'key_metrics' in overview:
                metrics = overview['key_metrics']
                for metric, value in metrics.items():
                    if 'total' in metric and isinstance(value, (int, float)) and value > 0:
                        insights.append(f"Total {metric.replace('_total', '')}: {value:,.0f}")
                    if 'trend' in metric and isinstance(value, str):
                        insights.append(f"Overall trend is {value}")
        
        # Trend insights
        if 'trends' in analysis:
            trends = analysis['trends']
            if 'temporal_trends' in trends:
                for metric, trend_info in trends['temporal_trends'].items():
                    direction = trend_info['trend_direction']
                    insights.append(f"{metric} shows {direction} trend")
        
        # Geographic insights
        if 'geographic_analysis' in analysis:
            geo = analysis['geographic_analysis']
            if 'geographic_concentration' in geo:
                for col, conc_info in geo['geographic_concentration'].items():
                    if conc_info['concentration_percentage'] > 50:
                        insights.append(f"High geographic concentration in {conc_info['most_concentrated']} ({conc_info['concentration_percentage']:.1f}%)")
        
        # Industry insights
        if 'industry_analysis' in analysis:
            industry = analysis['industry_analysis']
            if 'top_industries' in industry:
                for col, top_info in industry['top_industries'].items():
                    if top_info['names']:
                        insights.append(f"Top industry: {top_info['names'][0]}")
        
        return insights[:10]  # Limit to 10 insights
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        if 'overview' in analysis and 'data_coverage' in analysis['overview']:
            coverage = analysis['overview']['data_coverage']
            if coverage['completeness_rate'] < 80:
                recommendations.append("Improve data collection to reduce missing values")
        
        # Trend-based recommendations
        if 'trends' in analysis:
            trends = analysis['trends']
            if 'temporal_trends' in trends:
                for metric, trend_info in trends['temporal_trends'].items():
                    if trend_info['trend_direction'] == 'decreasing':
                        recommendations.append(f"Investigate declining trend in {metric}")
                    elif trend_info['trend_direction'] == 'increasing':
                        recommendations.append(f"Capitalize on growth in {metric}")
        
        # Geographic recommendations
        if 'geographic_analysis' in analysis:
            geo = analysis['geographic_analysis']
            if 'geographic_concentration' in geo:
                for col, conc_info in geo['geographic_concentration'].items():
                    if conc_info['concentration_percentage'] > 70:
                        recommendations.append(f"Consider expanding beyond {conc_info['most_concentrated']} to diversify geographic presence")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def get_analysis_results(self, dataset_name: str = None) -> Dict:
        """Get analysis results by name or all results"""
        if dataset_name:
            return {dataset_name: self.analysis_results.get(dataset_name)}
        return self.analysis_results
    
    def get_insights(self, dataset_name: str = None) -> Dict:
        """Get insights by name or all insights"""
        if dataset_name:
            return {dataset_name: self.insights.get(dataset_name)}
        return self.insights
