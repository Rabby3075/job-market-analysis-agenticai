"""
Data Visualization Utility
Generates interactive charts and graphs for job market analysis
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobMarketVisualizer:
    """Utility for creating interactive visualizations of job market data"""
    
    def __init__(self):
        self.charts = {}
        
    def create_dashboard(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            df: Processed dataframe
            analysis_results: Results from the analyzer
            
        Returns:
            Dictionary of chart figures
        """
        try:
            logger.info("Creating visualization dashboard")
            
            dashboard = {}
            
            # Overview charts
            dashboard['overview'] = self._create_overview_chart(df, analysis_results)
            dashboard['trends'] = self._create_trends_chart(df, analysis_results)
            dashboard['geography'] = self._create_geography_chart(df, analysis_results)
            dashboard['industry'] = self._create_industry_chart(df, analysis_results)
            dashboard['sector'] = self._create_sector_chart(df, analysis_results)
            
            # Store charts
            self.charts.update(dashboard)
            
            logger.info(f"Dashboard created with {len(dashboard)} charts")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {}
    
    def _create_overview_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        """Create overview summary chart"""
        try:
            # Extract key metrics
            overview = analysis_results.get('overview', {})
            key_metrics = overview.get('key_metrics', {})
            
            # Create summary cards
            fig = go.Figure()
            
            # Add summary statistics as text
            if key_metrics:
                metrics_text = "<br>".join([
                    f"<b>{k.replace('_', ' ').title()}:</b> {v:,.0f}" if isinstance(v, (int, float)) else f"<b>{k.replace('_', ' ').title()}:</b> {v}"
                    for k, v in list(key_metrics.items())[:5]
                ])
                
                fig.add_trace(go.Scatter(
                    x=[0.5], y=[0.5],
                    mode='text',
                    text=[metrics_text],
                    textposition='middle center',
                    showlegend=False,
                    textfont=dict(size=14)
                ))
            
            fig.update_layout(
                title="Job Market Overview",
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
                yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating overview chart: {str(e)}")
            return go.Figure()
    
    def _create_trends_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        """Create trends analysis chart"""
        try:
            trends = analysis_results.get('trends', {})
            temporal_trends = trends.get('temporal_trends', {})
            
            if not temporal_trends:
                return self._create_empty_chart("No trend data available")
            
            # Create subplot for multiple trends
            fig = sp.make_subplots(
                rows=len(temporal_trends), cols=1,
                subplot_titles=list(temporal_trends.keys()),
                vertical_spacing=0.1
            )
            
            for i, (metric, trend_info) in enumerate(temporal_trends.items()):
                if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        # Add trend line
                        x = np.arange(len(values))
                        y = values.values
                        
                        # Calculate trend line
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        trend_line = p(x)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x, y=y,
                                mode='lines+markers',
                                name=f'{metric} (Actual)',
                                line=dict(color='blue'),
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x, y=trend_line,
                                mode='lines',
                                name=f'{metric} (Trend)',
                                line=dict(color='red', dash='dash'),
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
            
            fig.update_layout(
                title="Trends Analysis",
                height=300 * len(temporal_trends),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trends chart: {str(e)}")
            return self._create_empty_chart("Error creating trends chart")
    
    def _create_geography_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        """Create geographic distribution chart"""
        try:
            geography = analysis_results.get('geographic_analysis', {})
            states_territories = geography.get('states_territories', {})
            
            if not states_territories:
                return self._create_empty_chart("No geographic data available")
            
            # Find the first geographic column with data
            geo_col = None
            geo_data = None
            
            for col, data in states_territories.items():
                if col in df.columns and 'top_locations' in data:
                    geo_col = col
                    geo_data = data['top_locations']
                    break
            
            if not geo_col or not geo_data:
                return self._create_empty_chart("No geographic data available")
            
            # Create bar chart
            locations = list(geo_data.keys())
            values = list(geo_data.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=locations,
                    y=values,
                    marker_color='lightblue',
                    text=values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Job Vacancies by {geo_col.replace('_', ' ').title()}",
                xaxis_title=geo_col.replace('_', ' ').title(),
                yaxis_title="Number of Vacancies",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating geography chart: {str(e)}")
            return self._create_empty_chart("Error creating geography chart")
    
    def _create_industry_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        """Create industry distribution chart"""
        try:
            industry = analysis_results.get('industry_analysis', {})
            top_industries = industry.get('top_industries', {})
            
            if not top_industries:
                return self._create_empty_chart("No industry data available")
            
            # Find the first industry column with data
            industry_col = None
            industry_data = None
            
            for col, data in top_industries.items():
                if col in df.columns and 'names' in data and 'values' in data:
                    industry_col = col
                    industry_data = data
                    break
            
            if not industry_col or not industry_data:
                return self._create_empty_chart("No industry data available")
            
            # Create horizontal bar chart for better readability
            names = industry_data['names']
            values = industry_data['values']
            
            fig = go.Figure(data=[
                go.Bar(
                    y=names,
                    x=values,
                    orientation='h',
                    marker_color='lightgreen',
                    text=values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Top Industries by {industry_col.replace('_', ' ').title()}",
                xaxis_title="Number of Vacancies",
                yaxis_title="Industry",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating industry chart: {str(e)}")
            return self._create_empty_chart("Error creating industry chart")
    
    def _create_sector_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        """Create sector comparison chart"""
        try:
            sector = analysis_results.get('sector_analysis', {})
            sector_breakdown = sector.get('sector_breakdown', {})
            
            if not sector_breakdown:
                return self._create_empty_chart("No sector data available")
            
            # Find the first sector column with data
            sector_col = None
            sector_data = None
            
            for col, data in sector_breakdown.items():
                if col in df.columns and 'sectors' in data:
                    sector_col = col
                    sector_data = data['sectors']
                    break
            
            if not sector_col or not sector_data:
                return self._create_empty_chart("No sector data available")
            
            # Create pie chart
            labels = list(sector_data.keys())
            values = list(sector_data.values())
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    textinfo='label+percent'
                )
            ])
            
            fig.update_layout(
                title=f"Job Vacancies by {sector_col.replace('_', ' ').title()}",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector chart: {str(e)}")
            return self._create_empty_chart("Error creating sector chart")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.5],
            mode='text',
            text=[message],
            textposition='middle center',
            showlegend=False,
            textfont=dict(size=14, color='gray')
        ))
        
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_custom_chart(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                           title: str = None, **kwargs) -> go.Figure:
        """
        Create a custom chart based on specified parameters
        
        Args:
            df: Dataframe to visualize
            chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram')
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            **kwargs: Additional chart parameters
            
        Returns:
            Plotly figure object
        """
        try:
            if x_col not in df.columns or y_col not in df.columns:
                return self._create_empty_chart(f"Columns {x_col} or {y_col} not found")
            
            if chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, title=title, **kwargs)
            elif chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, title=title, **kwargs)
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, title=title, **kwargs)
            elif chart_type == 'pie':
                fig = px.pie(df, values=y_col, names=x_col, title=title, **kwargs)
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=x_col, title=title, **kwargs)
            else:
                return self._create_empty_chart(f"Unsupported chart type: {chart_type}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating custom chart: {str(e)}")
            return self._create_empty_chart(f"Error creating {chart_type} chart")
    
    def get_charts(self, chart_name: str = None) -> Dict:
        """Get charts by name or all charts"""
        if chart_name:
            return {chart_name: self.charts.get(chart_name)}
        return self.charts
    
    def export_charts(self, output_dir: str = 'charts', format: str = 'html') -> List[str]:
        """
        Export all charts to files
        
        Args:
            output_dir: Directory to save charts
            format: Export format ('html', 'png', 'pdf')
            
        Returns:
            List of exported file paths
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        exported_files = []
        
        for chart_name, fig in self.charts.items():
            try:
                if format == 'html':
                    filepath = os.path.join(output_dir, f"{chart_name}.html")
                    fig.write_html(filepath)
                elif format == 'png':
                    filepath = os.path.join(output_dir, f"{chart_name}.png")
                    fig.write_image(filepath)
                elif format == 'pdf':
                    filepath = os.path.join(output_dir, f"{chart_name}.pdf")
                    fig.write_image(filepath)
                
                exported_files.append(filepath)
                logger.info(f"Exported {chart_name} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error exporting {chart_name}: {str(e)}")
        
        return exported_files

