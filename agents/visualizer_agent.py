"""
Visualization Agent
Turns processed data and analysis into interactive charts
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
    """Agent for creating interactive visualizations of job market data"""

    def __init__(self):
        self.charts = {}

    def create_dashboard(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, go.Figure]:
        logger.info("Creating visualization dashboard (notebook-style)")
        dashboard: Dict[str, go.Figure] = {}

        def make_unique(names: list[str]) -> list[str]:
            seen: Dict[str, int] = {}
            unique: list[str] = []
            for n in names:
                key = str(n)
                if key not in seen:
                    seen[key] = 0
                    unique.append(key)
                else:
                    seen[key] += 1
                    unique.append(f"{key} ({seen[key]})")
            return unique

        # Ensure dataframe has unique column names to avoid plotly assembly issues
        try:
            df = df.copy()
            df.columns = make_unique([str(c) for c in df.columns])
        except Exception as e:
            logger.warning(f"Column rename for uniqueness failed: {e}")
        try:
            if 'Date' in df.columns:
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
        except Exception:
            pass

        # State line chart + heatmap (detect state columns by robust name patterns)
        import re
        def slug(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
        state_labels = [
            'new_south_wales', 'victoria', 'queensland', 'south_australia',
            'western_australia', 'tasmania', 'northern_territory', 'australian_capital_territory'
        ]
        state_cols = []
        for c in df.columns:
            sc = slug(c)
            if any(lbl in sc for lbl in state_labels):
                state_cols.append(c)
            # also treat suffix-based state totals as state series
            elif sc.endswith('_total') or sc.endswith('_public') or sc.endswith('_private'):
                # try to exclude pure Australia totals if present
                if 'australia' not in sc:
                    state_cols.append(c)
        if 'Australia' in df.columns:
            state_cols = [c for c in state_cols if 'australia' not in str(c).lower()]
        # Ensure unique column names for plotting
        state_cols = list(dict.fromkeys(state_cols))
        # Industry columns heuristic (everything else numeric that's not SE_ or state)
        industry_cols = [c for c in df.columns if c not in {'Date', 'Year'} and not str(c).startswith('SE_') and c not in state_cols]

        try:
            if state_cols and 'Year' in df.columns:
                work = df[['Year'] + state_cols].copy()
                for c in state_cols:
                    work[c] = pd.to_numeric(work[c], errors='coerce')
                annual = work.groupby('Year', as_index=False)[state_cols].mean(numeric_only=True)
                fig_state_lines = go.Figure()
                for c in state_cols:
                    fig_state_lines.add_trace(go.Scatter(x=annual['Year'], y=annual[c], mode='lines', name=str(c)))
                fig_state_lines.update_layout(title='State job vacancies (annual mean)', xaxis_title='Year', yaxis_title='Vacancies')
                dashboard['state_lines'] = fig_state_lines

                growth = annual.set_index('Year')[state_cols].pct_change() * 100
                ylabels = make_unique(list(growth.columns))
                heat = go.Figure(data=go.Heatmap(z=growth.T.values, x=growth.index, y=ylabels, colorscale='RdYlGn', zmid=0))
                heat.update_layout(title='YoY growth by state (%)', xaxis_title='Year', yaxis_title='State')
                dashboard['state_growth_heatmap'] = heat
        except Exception as e:
            logger.warning(f"State charts failed: {e}")

        try:
            if industry_cols and 'Year' in df.columns:
                work = df[['Year'] + industry_cols].copy()
                for c in industry_cols:
                    work[c] = pd.to_numeric(work[c], errors='coerce')
                annual = work.groupby('Year', as_index=False)[industry_cols].mean(numeric_only=True)
                fig_ind = go.Figure()
                for c in industry_cols[:6]:  # limit for readability
                    fig_ind.add_trace(go.Scatter(x=annual['Year'], y=annual[c], mode='lines', name=str(c)))
                fig_ind.update_layout(title='Industry job vacancies (annual mean)', xaxis_title='Year', yaxis_title='Vacancies')
                dashboard['industry_lines'] = fig_ind

                growth = annual.set_index('Year')[industry_cols].pct_change() * 100
                ylabels = make_unique(list(growth.columns))
                heat = go.Figure(data=go.Heatmap(z=growth.T.values, x=growth.index, y=ylabels, colorscale='RdYlGn', zmid=0))
                heat.update_layout(title='YoY growth by industry (%)', xaxis_title='Year', yaxis_title='Industry')
                dashboard['industry_growth_heatmap'] = heat
        except Exception as e:
            logger.warning(f"Industry charts failed: {e}")

        # Generic fallback if no charts were produced
        try:
            if not dashboard:
                work = df.copy()
                if 'Date' in work.columns and 'Year' not in work.columns:
                    work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
                    work['Year'] = work['Date'].dt.year
                candidate_cols: list[str] = []
                for c in work.columns:
                    if c in {'Date', 'Year'} or str(c).startswith('SE_'):
                        continue
                    nums = pd.to_numeric(work[c], errors='coerce')
                    non_null_ratio = nums.notna().mean() if len(nums) else 0
                    if non_null_ratio >= 0.3:  # keep columns with some data
                        work[c] = nums
                        candidate_cols.append(c)
                # de-duplicate candidate columns
                candidate_cols = list(dict.fromkeys(candidate_cols))
                if 'Year' in work.columns and candidate_cols:
                    # cap number of series for readability
                    series = candidate_cols[:8]
                    annual = work.groupby('Year', as_index=False)[series].mean(numeric_only=True)
                    fig_lines = go.Figure()
                    for c in series:
                        fig_lines.add_trace(go.Scatter(x=annual['Year'], y=annual[c], mode='lines', name=str(c)))
                    fig_lines.update_layout(title='Annual averages (auto-detected)', xaxis_title='Year', yaxis_title='Value')
                    dashboard['auto_lines'] = fig_lines

                    growth = annual.set_index('Year')[series].pct_change() * 100
                    ylabels = make_unique(series)
                    heat = go.Figure(data=go.Heatmap(z=growth.T.values, x=growth.index, y=ylabels, colorscale='RdYlGn', zmid=0))
                    heat.update_layout(title='YoY growth (auto-detected) %', xaxis_title='Year', yaxis_title='Series')
                    dashboard['auto_growth_heatmap'] = heat
        except Exception as e:
            logger.warning(f"Fallback charts failed: {e}")

        # If still empty, produce a minimal placeholder figure so UI never shows empty
        if not dashboard:
            try:
                fig = go.Figure()
                fig.update_layout(title='No plottable numeric series found',
                                  xaxis={'visible': False}, yaxis={'visible': False},
                                  annotations=[{
                                      'text': 'Preprocessed sheet has no numeric columns after cleaning.',
                                      'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.5,
                                      'showarrow': False
                                  }])
                dashboard['notice'] = fig
            except Exception as e:
                logger.warning(f"Failed to build placeholder chart: {e}")

        self.charts = dashboard
        return dashboard

    def create_dashboard_from_notebook(self, df: pd.DataFrame, notebook_path: str | None) -> Dict[str, go.Figure]:
        """Lightweight heuristic that reads a notebook and decides which chart sets to build.
        We do not execute the notebook; we scan text to infer intent (states/industry, YoY, lines, heatmaps).
        """
        try:
            wants_states = False
            wants_industry = False
            wants_yoy = True
            wants_lines = True
            if notebook_path and os.path.exists(notebook_path):
                import json as _json
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = _json.load(f)
                text = "\n".join(
                    ["\n".join(c.get('source', [])) for c in nb.get('cells', []) if isinstance(c.get('source', []), list)]
                ).lower()
                for token in ["state", "states", "territories", "nsw", "victoria", "queensland"]:
                    if token in text:
                        wants_states = True
                        break
                for token in ["industry", "industries", "sector"]:
                    if token in text:
                        wants_industry = True
                        break
                wants_yoy = ("yoy" in text) or ("year-over-year" in text) or ("pct_change" in text)
                # lines are the default; heatmap if mentions
                wants_lines = True
            # Build full dashboard, then filter sections based on inferred wishes
            full = self.create_dashboard(df, analysis_results={})
            filtered: Dict[str, go.Figure] = {}
            for key, fig in full.items():
                name = key.lower()
                if ("state" in name and not wants_states):
                    continue
                if ("industry" in name and not wants_industry):
                    continue
                if ("heatmap" in name and not wants_yoy):
                    continue
                if ("lines" in name and not wants_lines):
                    continue
                filtered[key] = fig
            # If filtering removed everything, fall back to full
            return filtered or full
        except Exception as e:
            logger.warning(f"Notebook-driven visualization failed; using default. Error: {e}")
            return self.create_dashboard(df, analysis_results={})

    def _create_overview_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        try:
            overview = analysis_results.get('overview', {})
            key_metrics = overview.get('key_metrics', {})
            fig = go.Figure()
            if key_metrics:
                metrics_text = "<br>".join([
                    f"<b>{k.replace('_', ' ').title()}:</b> {v:,.0f}" if isinstance(v, (int, float)) else f"<b>{k.replace('_', ' ').title()}:</b> {v}"
                    for k, v in list(key_metrics.items())[:5]
                ])
                fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode='text', text=[metrics_text], textposition='middle center', showlegend=False, textfont=dict(size=14)))
            fig.update_layout(title="Job Market Overview", xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]), yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]), height=300, margin=dict(l=20, r=20, t=40, b=20))
            return fig
        except Exception as e:
            logger.error(f"Error creating overview chart: {str(e)}")
            return go.Figure()

    def _create_trends_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        try:
            trends = analysis_results.get('trends', {})
            temporal_trends = trends.get('temporal_trends', {})
            if not temporal_trends:
                return self._create_empty_chart("No trend data available")
            num_rows = max(1, len(temporal_trends))
            safe_spacing = 0.0 if num_rows == 1 else min(0.04, 1.0 / (num_rows + 1))
            fig = sp.make_subplots(rows=num_rows, cols=1, subplot_titles=list(temporal_trends.keys()), vertical_spacing=safe_spacing)
            for i, (metric, trend_info) in enumerate(temporal_trends.items()):
                if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        x = np.arange(len(values))
                        y = values.values
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        trend_line = p(x)
                        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'{metric} (Actual)', line=dict(color='blue'), showlegend=False), row=i+1, col=1)
                        fig.add_trace(go.Scatter(x=x, y=trend_line, mode='lines', name=f'{metric} (Trend)', line=dict(color='red', dash='dash'), showlegend=False), row=i+1, col=1)
            max_height = 1200
            est_height = 280 * num_rows
            fig.update_layout(title="Trends Analysis", height=min(est_height, max_height), showlegend=False)
            return fig
        except Exception as e:
            logger.error(f"Error creating trends chart: {str(e)}")
            return self._create_empty_chart("Error creating trends chart")

    def _create_geography_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        try:
            geography = analysis_results.get('geographic_analysis', {})
            states_territories = geography.get('states_territories', {})
            if not states_territories:
                return self._create_empty_chart("No geographic data available")
            geo_col = None
            geo_data = None
            for col, data in states_territories.items():
                if col in df.columns and 'top_locations' in data:
                    geo_col = col
                    geo_data = data['top_locations']
                    break
            if not geo_col or not geo_data:
                return self._create_empty_chart("No geographic data available")
            locations = list(geo_data.keys())
            values = list(geo_data.values())
            fig = go.Figure(data=[go.Bar(x=locations, y=values, marker_color='lightblue', text=values, textposition='auto')])
            fig.update_layout(title=f"Job Vacancies by {geo_col.replace('_', ' ').title()}", xaxis_title=geo_col.replace('_', ' ').title(), yaxis_title="Number of Vacancies", height=400)
            return fig
        except Exception as e:
            logger.error(f"Error creating geography chart: {str(e)}")
            return self._create_empty_chart("Error creating geography chart")

    def _create_industry_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        try:
            industry = analysis_results.get('industry_analysis', {})
            top_industries = industry.get('top_industries', {})
            if not top_industries:
                return self._create_empty_chart("No industry data available")
            industry_col = None
            industry_data = None
            for col, data in top_industries.items():
                if col in df.columns and 'names' in data and 'values' in data:
                    industry_col = col
                    industry_data = data
                    break
            if not industry_col or not industry_data:
                return self._create_empty_chart("No industry data available")
            names = industry_data['names']
            values = industry_data['values']
            fig = go.Figure(data=[go.Bar(y=names, x=values, orientation='h', marker_color='lightgreen', text=values, textposition='auto')])
            fig.update_layout(title=f"Top Industries by {industry_col.replace('_', ' ').title()}", xaxis_title="Number of Vacancies", yaxis_title="Industry", height=400)
            return fig
        except Exception as e:
            logger.error(f"Error creating industry chart: {str(e)}")
            return self._create_empty_chart("Error creating industry chart")

    def _create_sector_chart(self, df: pd.DataFrame, analysis_results: Dict) -> go.Figure:
        try:
            sector = analysis_results.get('sector_analysis', {})
            sector_breakdown = sector.get('sector_breakdown', {})
            if not sector_breakdown:
                return self._create_empty_chart("No sector data available")
            sector_col = None
            sector_data = None
            for col, data in sector_breakdown.items():
                if col in df.columns and 'sectors' in data:
                    sector_col = col
                    sector_data = data['sectors']
                    break
            if not sector_col or not sector_data:
                return self._create_empty_chart("No sector data available")
            labels = list(sector_data.keys())
            values = list(sector_data.values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, textinfo='label+percent')])
            fig.update_layout(title=f"Job Vacancies by {sector_col.replace('_', ' ').title()}", height=400)
            return fig
        except Exception as e:
            logger.error(f"Error creating sector chart: {str(e)}")
            return self._create_empty_chart("Error creating sector chart")

    # ABS-specific helpers
    def _extract_year(self, series: pd.Series) -> pd.Series:
        try:
            s = pd.to_datetime(series, errors='coerce')
            return s.dt.year
        except Exception:
            return pd.Series([None] * len(series))

    def _normalize(self, s: str) -> str:
        return str(s).lower().replace(' ', '_')

    def _create_state_heatmap(self, df: pd.DataFrame) -> go.Figure:
        try:
            if 'Date' not in df.columns:
                return self._create_empty_chart("No Date column")
            states = ['New South Wales','Victoria','Queensland','South Australia','Western Australia','Tasmania','Northern Territory','Australian Capital Territory','Australia']
            norm_states = [self._normalize(s) for s in states]
            # include variants like job_vacancies_new_south_wales_
            state_cols = []
            for c in df.columns:
                lc = self._normalize(c)
                if any(ns in lc for ns in norm_states):
                    state_cols.append(c)
            if not state_cols:
                return self._create_empty_chart("No state columns detected")
            work = df[['Date'] + state_cols].copy()
            work['Year'] = self._extract_year(work['Date'])
            # ensure numeric
            for c in state_cols:
                work[c] = pd.to_numeric(work[c], errors='coerce')
            melt = work.melt(id_vars=['Year'], value_vars=state_cols, var_name='State', value_name='Vacancies').dropna(subset=['Year', 'Vacancies'])
            pivot = melt.pivot_table(index='State', columns='Year', values='Vacancies', aggfunc='mean')
            fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Blues'))
            fig.update_layout(title='State vacancies heatmap (mean by year)', xaxis_title='Year', yaxis_title='State', height=500)
            return fig
        except Exception as e:
            logger.error(f"State heatmap error: {e}")
            return self._create_empty_chart("Error building state heatmap")

    def _create_state_trendlines(self, df: pd.DataFrame) -> go.Figure:
        try:
            if 'Date' not in df.columns:
                return self._create_empty_chart("No Date column")
            states = ['New South Wales','Victoria','Queensland','South Australia','Western Australia','Tasmania','Northern Territory','Australian Capital Territory']
            norm_states = [self._normalize(s) for s in states]
            state_cols = []
            for c in df.columns:
                lc = self._normalize(c)
                if any(ns in lc for ns in norm_states):
                    state_cols.append(c)
            if not state_cols:
                return self._create_empty_chart("No state columns detected")
            work = df[['Date'] + state_cols].copy()
            work['Year'] = self._extract_year(work['Date'])
            for c in state_cols:
                work[c] = pd.to_numeric(work[c], errors='coerce')
            annual = work.groupby('Year')[state_cols].mean().dropna(how='all')
            fig = go.Figure()
            for col in state_cols:
                if col in annual.columns:
                    fig.add_trace(go.Scatter(x=annual.index, y=annual[col], mode='lines', name=col))
            fig.update_layout(title='State vacancy trends (annual mean)', xaxis_title='Year', yaxis_title='Vacancies', height=500)
            return fig
        except Exception as e:
            logger.error(f"State trends error: {e}")
            return self._create_empty_chart("Error building state trends")

    def _create_industry_trendlines(self, df: pd.DataFrame) -> go.Figure:
        try:
            if 'Date' not in df.columns:
                return self._create_empty_chart("No Date column")
            non_cols = {'Date', 'Series ID', 'Unit'}
            cand = [c for c in df.columns if c not in non_cols]
            if not cand:
                return self._create_empty_chart("No industry columns detected")
            work = df[['Date'] + cand].copy()
            work['Year'] = self._extract_year(work['Date'])
            for c in cand:
                work[c] = pd.to_numeric(work[c], errors='coerce')
            annual = work.groupby('Year')[cand].mean().dropna(how='all')
            diffs = {}
            for c in cand:
                series = annual[c].dropna()
                if len(series) >= 2:
                    diffs[c] = series.iloc[-1] - series.iloc[0]
            top_up = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
            top_down = sorted(diffs.items(), key=lambda x: x[1])[:5]
            fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Top increasing industries", "Top decreasing industries"))
            for name, _ in top_up:
                fig.add_trace(go.Scatter(x=annual.index, y=annual[name], mode='lines', name=name), row=1, col=1)
            for name, _ in top_down:
                fig.add_trace(go.Scatter(x=annual.index, y=annual[name], mode='lines', name=name), row=2, col=1)
            fig.update_layout(title='Industry vacancy trends (annual mean)', height=700)
            return fig
        except Exception as e:
            logger.error(f"Industry trends error: {e}")
            return self._create_empty_chart("Error building industry trends")

    def _create_empty_chart(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode='text', text=[message], textposition='middle center', showlegend=False, textfont=dict(size=14, color='gray')))
        fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]), yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]), height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def get_charts(self, chart_name: str = None) -> Dict:
        if chart_name:
            return {chart_name: self.charts.get(chart_name)}
        return self.charts

    def export_charts(self, output_dir: str = 'charts', format: str = 'html') -> List[str]:
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


