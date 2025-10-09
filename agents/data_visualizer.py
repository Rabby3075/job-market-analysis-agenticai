"""
DataVisualizer agent

Interactive charts for ABS industry dataset.
Note: File saving has been removed per requirements; this agent focuses on in-memory charts.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Visualization agent for interactive industry charts and saving outputs.

    Focuses on ABS industry dataset long- and wide-form inputs.
    """

    def __init__(self) -> None:
        self.charts: Dict[str, go.Figure] = {}

    # -------------------------------
    # Public API (Interactive)  # ADDED: interactive chart API for Streamlit
    # -------------------------------
    def prepare_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a tidy long dataframe with columns: Date, Industry, Value, Year.

        Robust to duplicate/unnamed columns and mixed types.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["Date", "Industry", "Value", "Year"])  # ADDED

        data = df.copy()
        # Ensure unique column names
        data = data.loc[:, ~pd.Index(data.columns).duplicated()]  # ADDED
        # Standardize date column name
        if "Date" not in data.columns:
            data.columns = [str(c) for c in data.columns]
            data.rename(columns={data.columns[0]: "Date"}, inplace=True)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

        # Keep only numeric value columns
        value_cols: List[str] = [c for c in data.columns if c != "Date"]
        for c in value_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        keep_cols = ["Date"] + [c for c in value_cols if data[c].notna().any()]
        data = data.loc[:, keep_cols]

        long = (
            data.melt(id_vars=["Date"], var_name="Industry", value_name="Value")
            .dropna(subset=["Value"])  # ADDED
        )
        long["Year"] = long["Date"].dt.year
        return long

    # ADDED: core plots used by Streamlit tabs
    def chart_multiline(self, long: pd.DataFrame, industries: Optional[List[str]] = None) -> go.Figure:
        d = long.copy()
        if industries:
            d = d[d["Industry"].isin(industries)]
        fig = px.line(d.sort_values(["Industry", "Date"]), x="Date", y="Value", color="Industry",
                      title="Multi-line time series: vacancies by industry")
        return fig

    def chart_indexed(self, long: pd.DataFrame, base: str = "2019-01-01", industries: Optional[List[str]] = None) -> go.Figure:
        d = long.copy().sort_values(["Industry", "Date"])
        if industries:
            d = d[d["Industry"].isin(industries)]
        # Robust base date parsing with fallback
        base_dt = pd.to_datetime(base, errors="coerce")
        if pd.isna(base_dt):
            try:
                base_dt = pd.to_datetime("2019-01-01")
            except Exception:
                base_dt = d["Date"].min()

        def _base_val(group: pd.DataFrame) -> float:
            if group.empty:
                return np.nan
            idx = (group["Date"] - base_dt).abs().idxmin()
            return float(group.loc[idx, "Value"]) if pd.notna(group.loc[idx, "Value"]) else np.nan

        base_vals = d.groupby("Industry", as_index=True).apply(_base_val).rename("BaseVal")
        m = d.merge(base_vals, left_on="Industry", right_index=True, how="left")
        m["Indexed"] = (m["Value"] / m["BaseVal"]) * 100.0
        fig = px.line(m, x="Date", y="Indexed", color="Industry",
                      title=f"Indexed line (Base={base_dt.date()}, Index=100)")
        # Shade COVID
        fig.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="LightSalmon", opacity=0.2, line_width=0)
        fig.update_yaxes(title="Index (Base=100)")
        return fig

    def chart_rolling_mean(self, long: pd.DataFrame, window: int = 4, industries: Optional[List[str]] = None) -> go.Figure:
        d = long.copy().sort_values(["Industry", "Date"]).reset_index(drop=True)
        if industries:
            d = d[d["Industry"].isin(industries)]
        d["Rolling"] = d.groupby("Industry")["Value"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        fig = px.line(d, x="Date", y="Rolling", color="Industry", title=f"{window}-quarter rolling average")
        return fig

    def chart_latest_bar(self, long: pd.DataFrame) -> go.Figure:
        if long.empty:
            return go.Figure()
        last = long["Date"].max()
        cur = (
            long[long["Date"] == last]
            .groupby("Industry", as_index=False)["Value"].sum()
        )
        # Exclude any aggregate totals from rankings
        cur = cur[~cur["Industry"].str.contains(r"^\s*total\b|\ball\s*industr", case=False, regex=True)].copy()
        cur = cur.sort_values("Value", ascending=True)
        fig = go.Figure(go.Bar(y=cur["Industry"], x=cur["Value"], orientation="h"))
        fig.update_layout(title=f"Top industries — latest period ({pd.to_datetime(last).date()})", xaxis_title="Vacancies")
        return fig

    def chart_latest_pie(self, long: pd.DataFrame) -> go.Figure:
        if long.empty:
            return go.Figure()
        last = long["Date"].max()
        cur = (
            long[long["Date"] == last]
            .groupby("Industry", as_index=False)["Value"].sum()
        )
        # Exclude any aggregate totals like "Total All Industries"
        cur = cur[~cur["Industry"].str.contains(r"^\s*total\b|\ball\s*industr", case=False, regex=True)].copy()
        fig = px.pie(cur, names="Industry", values="Value", title=f"Share of vacancies by industry — {pd.to_datetime(last).date()}")
        return fig

    def chart_stacked_composition(self, long: pd.DataFrame) -> go.Figure:
        """Stacked bar: each bar is a Year; segments are industries (composition over time)."""
        if long.empty:
            return go.Figure()
        d = long.copy()
        d["Year"] = pd.to_datetime(d["Date"]).dt.year
        # Aggregate to yearly totals per industry
        yearly = d.groupby(["Year", "Industry"], as_index=False)["Value"].sum()
        # Exclude aggregate industries if any
        yearly = yearly[~yearly["Industry"].str.contains(r"^\s*total\b|\ball\s*industr", case=False, regex=True)]
        fig = px.bar(yearly, x="Year", y="Value", color="Industry", barmode="stack",
                     title="Industry composition over time — stacked by industry")
        fig.update_layout(xaxis_title="Year", yaxis_title="Vacancies")
        return fig

    def chart_yoy_heatmap(self, long: pd.DataFrame) -> go.Figure:
        d = long.sort_values(["Industry", "Date"]).copy()
        d["YoY"] = d.groupby("Industry")["Value"].pct_change(periods=4) * 100
        pivot = d.pivot_table(index="Industry", columns="Year", values="YoY", aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn", origin="lower",
                        labels=dict(color="YoY %"), title="Year-on-Year % change heatmap")
        return fig

    def chart_growth_vs_size_bubble(self, long: pd.DataFrame) -> go.Figure:
        d = long.sort_values(["Industry", "Date"]).copy()
        first = d.groupby("Industry", as_index=False).first().rename(columns={"Value": "First"})
        last = d.groupby("Industry", as_index=False).last().rename(columns={"Value": "Last"})
        m = first[["Industry", "Date", "First"]].merge(last[["Industry", "Date", "Last"]], on="Industry", suffixes=("_first", "_last"))
        # Compute CAGR-like rate per year; fallback to simple pct change if zero years
        years = (pd.to_datetime(m["Date_last"]) - pd.to_datetime(m["Date_first"])) / pd.Timedelta(days=365.25)
        pct = (m["Last"] / m["First"]).replace([np.inf, -np.inf], np.nan)
        growth = (pct ** (1 / years.replace(0, np.nan))) - 1
        m["GrowthRate"] = (growth * 100).fillna(((m["Last"] - m["First"]) / m["First"]) * 100)
        m["Size"] = m["Last"].abs()
        fig = px.scatter(m, x="Size", y="GrowthRate", size="Size", color="Industry",
                         title="Growth vs Size — bubble = latest size",
                         labels={"Size": "Total vacancies (latest)", "GrowthRate": "Growth rate % / year"})
        fig.update_layout(xaxis_type="log")
        return fig

    def chart_delta_between(self, long: pd.DataFrame, start: str = "2019-01-01", end: Optional[str] = None) -> go.Figure:
        d = long.copy()
        dates = sorted(d["Date"].unique())
        if not dates:
            return go.Figure()
        start_dt = pd.to_datetime(start)
        nearest_start = min(dates, key=lambda t: abs(pd.Timestamp(t) - start_dt))
        end_dt = pd.to_datetime(end) if end else pd.to_datetime(dates[-1])
        nearest_end = min(dates, key=lambda t: abs(pd.Timestamp(t) - end_dt))
        s = d[d["Date"] == nearest_start].groupby("Industry")["Value"].sum()
        e = d[d["Date"] == nearest_end].groupby("Industry")["Value"].sum()
        idx = sorted(set(s.index) | set(e.index))
        s = s.reindex(idx).fillna(0)
        e = e.reindex(idx).fillna(0)
        delta = (e - s).sort_values(ascending=True)
        fig = go.Figure(go.Bar(y=delta.index.tolist(), x=delta.values.tolist(), orientation="h"))
        fig.update_layout(title=f"Change by industry: {pd.Timestamp(nearest_start).date()} → {pd.Timestamp(nearest_end).date()}",
                          xaxis_title="Δ Vacancies")
        return fig

    # -------------------------------
    # Saving utilities
    # -------------------------------
    def save_pngs(self, figures: Dict[str, go.Figure], outdir: str = "charts", prefix: Optional[str] = None) -> Dict[str, str]:
        """Save provided Plotly figures as PNGs into outdir. Returns map of names to file paths.

        Requires kaleido to be installed.
        """
        os.makedirs(outdir, exist_ok=True)
        try:
            import kaleido  # noqa: F401
            kaleido_ok = True
        except Exception:
            kaleido_ok = False
        if not kaleido_ok:
            return {}

        saved: Dict[str, str] = {}
        for name, fig in figures.items():
            safe_name = name.replace(" ", "_")
            base = f"{prefix}_{safe_name}" if prefix else safe_name
            path = os.path.join(outdir, f"{base}.png")
            try:
                fig.write_image(path, width=1400, height=800, scale=2)
                saved[name] = path
            except Exception:
                continue
        return saved

    # -------------------------------
    # Batch generate (in-memory only)
    # -------------------------------
    def create_industry_charts(
        self,
        df: pd.DataFrame,
        base: Optional[str] = None,
    ) -> Dict[str, go.Figure]:
        """Create all requested industry charts and return as a dict (no saving)."""

        if df is None or df.empty:
            return {}

        # Clean columns and coerce date/numerics
        df = df.copy()
        # Ensure unique column names to avoid plotly duplicate key errors
        # Keep first occurrence when names repeat
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        # If duplicate "Date" column exists, keep the first
        if (df.columns == "Date").sum() > 1:
            first_date_idx = np.where(df.columns == "Date")[0][0]
            keep = [c for i, c in enumerate(df.columns) if i == first_date_idx or c != "Date"]
            df = df.loc[:, keep]

        # Convert date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            # Try to infer first column as date
            df.columns = [str(c) for c in df.columns]
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Keep only numeric value columns
        value_cols: List[str] = [c for c in df.columns if c != "Date"]
        for c in value_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop fully empty columns
        keep_cols = ["Date"] + [c for c in value_cols if df[c].notna().any()]
        df = df.loc[:, keep_cols]

        # Long form
        long = df.melt(id_vars=["Date"], var_name="Industry", value_name="Value").dropna(subset=["Value"]) 
        long["Year"] = long["Date"].dt.year

        dashboard: Dict[str, go.Figure] = {}


        def _index_by_base(data: pd.DataFrame, base_dt: pd.Timestamp) -> pd.DataFrame:
            # compute index = 100 at base_dt per industry (closest prev/next available)
            data = data.sort_values(["Industry", "Date"]).copy()
            def _base_val(group: pd.DataFrame) -> float:
                # pick nearest date's value
                if group.empty:
                    return np.nan
                idx = (group["Date"] - base_dt).abs().idxmin()
                return float(group.loc[idx, "Value"]) if pd.notna(group.loc[idx, "Value"]) else np.nan
            base_vals = (
                data.groupby("Industry", as_index=True)
                .apply(_base_val)
                .rename("BaseVal")
            )
            merged = data.merge(base_vals, left_on="Industry", right_index=True, how="left")
            merged["Indexed"] = (merged["Value"] / merged["BaseVal"]) * 100.0
            return merged

        # 1) Indexed Growth Timeline (COVID shaded)
        try:
            if base is None:
                base = str(long["Date"].min().date())
            base_dt = pd.to_datetime(base)
            idx_long = _index_by_base(long.copy(), base_dt)
            fig_idx = px.line(
                idx_long,
                x="Date",
                y="Indexed",
                color="Industry",
                title=f"Indexed growth (Base={base_dt.date()}, Index=100) — COVID shaded",
            )
            self._shade_covid(fig_idx)
            fig_idx.update_yaxes(title="Index (Base=100)")
            dashboard["indexed_growth"] = fig_idx
        except Exception:
            pass

        # 2) YoY Heatmap (Industry × Year)
        try:
            yoy = (
                long.sort_values(["Industry", "Date"])
                .assign(YoY=lambda d: d.groupby("Industry")["Value"].pct_change(periods=4) * 100)
            )
            pivot = yoy.pivot_table(index="Industry", columns="Year", values="YoY", aggfunc="mean")
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(color="YoY %"),
                title="Year-over-Year change (%) by Industry × Year",
            )
            dashboard["yoy_heatmap"] = fig_heat
        except Exception:
            pass

        # 3) Rank Flow (Bump Chart)
        try:
            ranks = (
                long.groupby(["Year", "Industry"], as_index=False)["Value"].mean()
                .assign(Rank=lambda d: d.groupby("Year")["Value"].rank(ascending=False, method="first"))
            )
            fig_bump = px.line(
                ranks,
                x="Year",
                y="Rank",
                color="Industry",
                markers=True,
                title="Industry rank flow by Year (1 = highest vacancies)",
            )
            fig_bump.update_yaxes(autorange="reversed", title="Rank")
            dashboard["rank_bump"] = fig_bump
        except Exception:
            pass

        # 4) Contribution Waterfall (2019 → 2021, 2021 → latest)
        def _waterfall_between(d1: pd.Timestamp, d2: pd.Timestamp, label: str) -> go.Figure:
            s = long[long["Date"] == d1].groupby("Industry")["Value"].sum()
            e = long[long["Date"] == d2].groupby("Industry")["Value"].sum()
            idx = sorted(set(s.index) | set(e.index))
            s = s.reindex(idx).fillna(0)
            e = e.reindex(idx).fillna(0)
            delta = (e - s).sort_values(ascending=False)
            steps: List[Dict[str, object]] = []
            steps.append(dict(type="absolute", label=f"{pd.Timestamp(d1).date()} total", value=float(s.sum())))
            for name, val in delta.items():
                steps.append(dict(type="relative", label=str(name), value=float(val)))
            steps.append(dict(type="total", label=f"{pd.Timestamp(d2).date()} total", value=float(e.sum())))
            fig = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=[st["type"] for st in steps],
                    x=[st["label"] for st in steps],
                    y=[st["value"] for st in steps],
                    connector={"line": {"width": 1}},
                )
            )
            fig.update_layout(title=f"Industry contribution to change: {label}", yaxis_title="Vacancies (Δ and totals)")
            return fig

        try:
            # choose quarter ends nearest to 2019-12, 2021-12, latest
            all_dates = sorted(long["Date"].unique())

            def nearest(ts: List[pd.Timestamp], target: str) -> pd.Timestamp:
                return min(ts, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(target)))

            d_2019 = nearest(all_dates, "2019-12-01")
            d_2021 = nearest(all_dates, "2021-12-01")
            d_last = all_dates[-1]

            dashboard["waterfall_2019_2021"] = _waterfall_between(pd.Timestamp(d_2019), pd.Timestamp(d_2021), "2019 → 2021 (COVID shock)")
            dashboard["waterfall_2021_latest"] = _waterfall_between(pd.Timestamp(d_2021), pd.Timestamp(d_last), "2021 → latest (recovery)")
        except Exception:
            pass

        # 5) Rolling Volatility Heatmap (σ of QoQ growth)
        try:
            tmp = long.sort_values(["Industry", "Date"]).copy()
            tmp["QoQ"] = tmp.groupby("Industry")["Value"].pct_change() * 100
            tmp["Vol4"] = tmp.groupby("Industry")["QoQ"].rolling(4, min_periods=4).std().reset_index(level=0, drop=True)
            pv = tmp.pivot_table(index="Industry", columns="Date", values="Vol4", aggfunc="mean")
            fig_vol = px.imshow(
                pv,
                aspect="auto",
                labels=dict(color="Rolling σ (4q) of QoQ %"),
                title="Volatility heatmap (rolling 4-quarter std of QoQ growth)",
            )
            dashboard["volatility_heatmap"] = fig_vol
        except Exception:
            pass

        # 6) Small Multiples (one mini line per industry, COVID shaded)
        try:
            fig_sm = px.line(long, x="Date", y="Value", facet_col="Industry", facet_col_wrap=4)
            fig_sm.update_layout(title="Industry small multiples (with COVID shading)", showlegend=False, height=900)
            self._shade_covid(fig_sm)
            dashboard["small_multiples"] = fig_sm
        except Exception:
            pass

        # 7) Industry Recovery Leaderboard (latest vs previous)
        try:
            last = long["Date"].max()
            prev = long[long["Date"] < last]["Date"].max()
            cur = long[long["Date"] == last].groupby("Industry", as_index=False)["Value"].sum()
            prv = (
                long[long["Date"] == prev].groupby("Industry", as_index=False)["Value"].sum().rename(columns={"Value": "Prev"})
            )
            m = cur.merge(prv, on="Industry", how="left").fillna(0.0)
            m["Delta"] = m["Value"] - m["Prev"]
            m = m.sort_values("Value", ascending=True)
            fig_lb = go.Figure(
                go.Bar(
                    y=m["Industry"],
                    x=m["Value"],
                    orientation="h",
                    text=[f"Δ {d:+.0f}" for d in m["Delta"]],
                    textposition="outside",
                )
            )
            fig_lb.update_layout(
                title=f"Latest ranking ({pd.to_datetime(last).date()}) with Δ vs previous",
                xaxis_title="Vacancies",
                yaxis_title=None,
                height=700,
            )
            dashboard["leaderboard_latest"] = fig_lb
        except Exception:
            pass

        # Save figures in-memory
        self.charts.update(dashboard)
        return dashboard

    # -------------------------------
    # Forecasting Methods
    # -------------------------------
    
    def _generate_forecast(self, data: pd.DataFrame, industry: str, forecast_years: int = 5) -> Tuple[pd.DataFrame, float, float]:
        """
        Generate forecast for a single industry using polynomial regression.
        
        Args:
            data: Long format data with Date, Industry, Value columns
            industry: Industry name to forecast
            forecast_years: Number of years to forecast ahead
            
        Returns:
            Tuple of (forecast_df, r2_score, confidence_interval)
        """
        try:
            # Filter data for specific industry
            industry_data = data[data['Industry'] == industry].copy()
            if len(industry_data) < 12:  # Need at least 1 year of data
                return pd.DataFrame(), 0.0, 0.0
            
            # Sort by date
            industry_data = industry_data.sort_values('Date')
            
            # Create time features
            industry_data['days_since_start'] = (industry_data['Date'] - industry_data['Date'].min()).dt.days
            industry_data['quarter'] = industry_data['Date'].dt.quarter
            industry_data['year'] = industry_data['Date'].dt.year
            
            # Prepare features for regression
            X = industry_data[['days_since_start', 'quarter']].values
            y = industry_data['Value'].values
            
            # Use polynomial features for non-linear trends
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(X)
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Calculate R² score
            y_pred = model.predict(X_poly)
            r2_score = model.score(X_poly, y)
            
            # Generate future dates
            last_date = industry_data['Date'].max()
            try:
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=90),  # Start from next quarter
                    periods=forecast_years * 4,  # Quarterly data
                    freq='Q'
                )
            except Exception as e:
                print(f"Date range generation error for {industry}: {e}")
                return pd.DataFrame(), 0.0, 0.0
            
            # Prepare future features
            future_data = pd.DataFrame({
                'Date': future_dates,
                'Industry': industry,
                'days_since_start': (future_dates - industry_data['Date'].min()).days,
                'quarter': future_dates.quarter
            })
            
            X_future = future_data[['days_since_start', 'quarter']].values
            X_future_poly = poly_features.transform(X_future)
            
            # Generate predictions
            future_predictions = model.predict(X_future_poly)
            
            # Calculate confidence interval (simplified)
            residuals = y - y_pred
            std_error = np.std(residuals)
            confidence_interval = 1.96 * std_error  # 95% confidence
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Industry': industry,
                'Value': future_predictions,
                'Year': future_dates.year,
                'Type': 'Forecast'
            })
            
            return forecast_df, r2_score, confidence_interval
            
        except Exception as e:
            print(f"Forecast error for {industry}: {e}")
            return pd.DataFrame(), 0.0, 0.0
    
    def chart_historical_and_forecast(self, long_data: pd.DataFrame, industries: Optional[List[str]] = None, 
                                    forecast_years: int = 5, confidence_level: float = 0.95) -> go.Figure:
        """
        Combined chart showing historical trends + 5-year predictions with confidence intervals.
        
        Args:
            long_data: Long format data with Date, Industry, Value columns
            industries: List of industries to include (None for all)
            forecast_years: Number of years to forecast ahead
            confidence_level: Confidence level for prediction intervals (0.95 = 95%)
            
        Returns:
            Plotly figure with historical and forecast data
        """
        try:
            if long_data is None or long_data.empty:
                return go.Figure()
            
            # Filter industries if specified
            if industries:
                long_data = long_data[long_data['Industry'].isin(industries)]
            
            # Get unique industries
            unique_industries = long_data['Industry'].unique()
            if len(unique_industries) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            # Color palette for industries - use a more reliable color list
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            
            for i, industry in enumerate(unique_industries):
                color = colors[i % len(colors)]
                
                # Historical data
                hist_data = long_data[long_data['Industry'] == industry].copy()
                hist_data = hist_data.sort_values('Date')
                
                # Add historical line
                fig.add_trace(go.Scatter(
                    x=hist_data['Date'],
                    y=hist_data['Value'],
                    mode='lines+markers',
                    name=f'{industry} (Historical)',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{industry}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Generate forecast
                forecast_df, r2_score, confidence_interval = self._generate_forecast(
                    long_data, industry, forecast_years
                )
                
                if not forecast_df.empty and r2_score > 0.3:  # Only show if reasonable fit
                    # Add forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Value'],
                        mode='lines',
                        name=f'{industry} (Forecast)',
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate=f'<b>{industry} (Forecast)</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Predicted: %{y:,.0f}<br>' +
                                     f'R²: {r2_score:.2f}<br>' +
                                     '<extra></extra>'
                    ))
                    
                    # Add confidence interval
                    upper_bound = forecast_df['Value'] + confidence_interval
                    lower_bound = forecast_df['Value'] - confidence_interval
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Convert color to rgba format safely
                    try:
                        if color.startswith('#'):
                            r = int(color[1:3], 16)
                            g = int(color[3:5], 16)
                            b = int(color[5:7], 16)
                            rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                        else:
                            # Fallback to a default color
                            rgba_color = 'rgba(128, 128, 128, 0.2)'
                    except (ValueError, IndexError):
                        rgba_color = 'rgba(128, 128, 128, 0.2)'
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=rgba_color,
                        name=f'{industry} (95% CI)',
                        hoverinfo='skip'
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'Historical Trends & {forecast_years}-Year Forecast',
                xaxis_title='Date',
                yaxis_title='Job Vacancies',
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Add vertical line to separate historical from forecast
            if not long_data.empty:
                try:
                    last_historical_date = long_data['Date'].max()
                    # Use add_shape instead of add_vline to avoid timestamp issues
                    fig.add_shape(
                        type="line",
                        x0=last_historical_date,
                        x1=last_historical_date,
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(dash="dot", color="gray", width=2)
                    )
                    # Add annotation separately
                    fig.add_annotation(
                        x=last_historical_date,
                        y=0.95,
                        yref="paper",
                        text="Historical End",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="gray"
                    )
                except Exception:
                    # Skip the vertical line if there are issues
                    pass
            
            # Add COVID shading
            self._shade_covid(fig)
            
            return fig
        
        except Exception as e:
            print(f"Forecast chart error: {e}")
            # Return a simple historical chart as fallback
            if long_data is not None and not long_data.empty:
                fig = go.Figure()
                for industry in long_data['Industry'].unique()[:5]:  # Limit to 5 industries
                    hist_data = long_data[long_data['Industry'] == industry]
                    fig.add_trace(go.Scatter(
                        x=hist_data['Date'],
                        y=hist_data['Value'],
                        mode='lines+markers',
                        name=industry,
                        line=dict(width=2)
                    ))
                fig.update_layout(
                    title='Historical Trends (Forecast Unavailable)',
                    xaxis_title='Date',
                    yaxis_title='Job Vacancies'
                )
                return fig
            return go.Figure()
    
    def chart_forecast_summary(self, long_data: pd.DataFrame, industries: Optional[List[str]] = None, 
                             forecast_years: int = 5) -> go.Figure:
        """
        Summary chart showing forecast accuracy and growth projections.
        
        Args:
            long_data: Long format data with Date, Industry, Value columns
            industries: List of industries to include (None for all)
            forecast_years: Number of years to forecast ahead
            
        Returns:
            Plotly figure with forecast summary
        """
        if long_data is None or long_data.empty:
            return go.Figure()
        
        # Filter industries if specified
        if industries:
            long_data = long_data[long_data['Industry'].isin(industries)]
        
        unique_industries = long_data['Industry'].unique()
        if len(unique_industries) == 0:
            return go.Figure()
        
        # Calculate forecast metrics for each industry
        forecast_metrics = []
        
        for industry in unique_industries:
            forecast_df, r2_score, confidence_interval = self._generate_forecast(
                long_data, industry, forecast_years
            )
            
            if not forecast_df.empty:
                # Calculate growth rate
                current_value = long_data[long_data['Industry'] == industry]['Value'].iloc[-1]
                future_value = forecast_df['Value'].iloc[-1]
                growth_rate = ((future_value - current_value) / current_value) * 100
                
                forecast_metrics.append({
                    'Industry': industry,
                    'R² Score': r2_score,
                    'Growth Rate (%)': growth_rate,
                    'Current Value': current_value,
                    'Forecast Value': future_value,
                    'Confidence': confidence_interval
                })
        
        if not forecast_metrics:
            return go.Figure()
        
        metrics_df = pd.DataFrame(forecast_metrics)
        
        # Create subplot with two charts
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Forecast Accuracy (R² Score)', 'Growth Projections (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² Score chart
        fig.add_trace(
            go.Bar(
                x=metrics_df['Industry'],
                y=metrics_df['R² Score'],
                name='R² Score',
                marker_color='lightblue',
                text=[f'{score:.2f}' for score in metrics_df['R² Score']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Growth rate chart
        colors = ['green' if x > 0 else 'red' for x in metrics_df['Growth Rate (%)']]
        fig.add_trace(
            go.Bar(
                x=metrics_df['Industry'],
                y=metrics_df['Growth Rate (%)'],
                name='Growth Rate',
                marker_color=colors,
                text=[f'{rate:+.1f}%' for rate in metrics_df['Growth Rate (%)']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Forecast Summary - {forecast_years} Year Projections',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
        
        return fig

    def _shade_covid(self, fig: go.Figure) -> None:
        """Add COVID period shading to a plotly figure"""
        try:
            fig.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="LightSalmon", opacity=0.2, line_width=0)
        except Exception:
            # Fallback: skip COVID shading if there are issues
            pass

    # ==================== IVI (IT Jobs) Visualizations ====================
    
    def chart_ivi_trend_over_time(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create trend over time chart for IT job vacancies"""
        try:
            logger.info("Creating IVI trend over time chart")
            
            # Get date columns (exclude non-date columns)
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            
            # Convert date columns to datetime and calculate monthly totals
            dates = []
            totals = []
            
            for col in date_columns:
                try:
                    date = pd.to_datetime(col)
                    total = df[col].sum()
                    dates.append(date)
                    totals.append(total)
                except:
                    continue
            
            if not dates or not totals:
                logger.warning("No valid date data found for trend chart")
                return None
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=totals,
                mode='lines+markers',
                name='IT Job Vacancies',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea'),
                hovertemplate='<b>%{x|%B %Y}</b><br>Vacancies: %{y:,.0f}<extra></extra>'
            ))
            
            # Add COVID period shading
            covid_start = pd.Timestamp('2020-03-01')
            covid_end = pd.Timestamp('2022-01-01')
            
            fig.add_vrect(
                x0=covid_start, x1=covid_end,
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below", line_width=0,
                annotation_text="COVID Period", annotation_position="top left"
            )
            
            fig.update_layout(
                title={
                    'text': "IT Job Vacancies Trend Over Time",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Date",
                yaxis_title="Job Vacancies",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI trend chart: {e}")
            return None

    def chart_ivi_state_distribution(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create state-wise vacancy distribution map with scatter plot"""
        try:
            logger.info("Creating IVI state distribution map")
            
            # Calculate total vacancies by state (exclude AUST - national total)
            numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            state_totals = df[df['state'] != 'AUST'].groupby('state')[numeric_cols].sum().sum(axis=1)
            
            # Map state codes to coordinates and names
            state_coords = {
                'NSW': {'name': 'New South Wales', 'lat': -31.2532, 'lon': 146.9211},
                'VIC': {'name': 'Victoria', 'lat': -37.8136, 'lon': 144.9631}, 
                'QLD': {'name': 'Queensland', 'lat': -23.4695, 'lon': 144.9778},
                'WA': {'name': 'Western Australia', 'lat': -25.2744, 'lon': 133.7751},
                'SA': {'name': 'South Australia', 'lat': -30.0002, 'lon': 136.2092},
                'TAS': {'name': 'Tasmania', 'lat': -41.4545, 'lon': 145.9707},
                'NT': {'name': 'Northern Territory', 'lat': -19.4914, 'lon': 132.5509},
                'ACT': {'name': 'Australian Capital Territory', 'lat': -35.2809, 'lon': 149.1300}
            }
            
            # Prepare data for scatter plot
            lats = []
            lons = []
            values = []
            text_labels = []
            sizes = []
            
            for state_code, total in state_totals.items():
                if state_code in state_coords:
                    lats.append(state_coords[state_code]['lat'])
                    lons.append(state_coords[state_code]['lon'])
                    values.append(total)
                    # Size proportional to vacancy count (scaled for visibility)
                    sizes.append(max(10, min(50, total / 10000)))
                    text_labels.append(f"{state_coords[state_code]['name']}<br>Vacancies: {total:,.0f}")
            
            # Create scatter plot on map
            fig = go.Figure()
            
            fig.add_trace(go.Scattergeo(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Job Vacancies"),
                    line=dict(width=2, color='white')
                ),
                text=text_labels,
                hovertemplate='%{text}<extra></extra>',
                name='IT Job Vacancies'
            ))
            
            fig.update_layout(
                title={
                    'text': "IT Job Vacancies by State/Territory",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                geo=dict(
                    scope='world',
                    showframe=True,
                    showcoastlines=True,
                    showland=True,
                    landcolor='lightgray',
                    showocean=True,
                    oceancolor='lightblue',
                    projection_type='equirectangular',
                    center=dict(lat=-25, lon=135),  # Center on Australia
                    lonaxis_range=[110, 155],  # Longitude range for Australia
                    lataxis_range=[-45, -10]   # Latitude range for Australia
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI state distribution map: {e}")
            return None

    def chart_ivi_top_occupations(self, df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
        """Create top occupations treemap visualization"""
        try:
            logger.info(f"Creating IVI top occupations treemap (top {top_n})")
            
            # Calculate total vacancies by occupation
            numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            occupation_totals = df.groupby('ANZSCO_TITLE')[numeric_cols].sum().sum(axis=1)
            
            # Get top N occupations
            top_occupations = occupation_totals.nlargest(top_n)
            
            # Create horizontal bar chart (more reliable than treemap)
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=top_occupations.index.tolist(),
                x=top_occupations.values.tolist(),
                orientation='h',
                marker=dict(
                    color=top_occupations.values.tolist(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Job Vacancies")
                ),
                text=[f'{val:,.0f}' for val in top_occupations.values],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Vacancies: %{x:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f"Top {top_n} IT Occupations by Vacancies",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Job Vacancies",
                yaxis_title="Occupation",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=150, r=50, t=80, b=50),
                height=500,
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI top occupations chart: {e}")
            return None

    def chart_ivi_yoy_growth(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create year-on-year growth chart"""
        try:
            logger.info("Creating IVI YoY growth chart")
            
            # Get date columns and calculate annual totals
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            dates = pd.to_datetime(date_columns, errors='coerce')
            
            annual_totals = {}
            for i, col in enumerate(date_columns):
                if i < len(dates) and pd.notna(dates[i]):
                    year = dates[i].year
                    if year not in annual_totals:
                        annual_totals[year] = 0
                    annual_totals[year] += df[col].sum()
            
            # Calculate YoY growth
            years_sorted = sorted(annual_totals.keys())
            growth_data = []
            growth_years = []
            
            for i in range(1, len(years_sorted)):
                current_year = years_sorted[i]
                previous_year = years_sorted[i-1]
                current_total = annual_totals[current_year]
                previous_total = annual_totals[previous_year]
                
                if previous_total > 0:
                    growth = ((current_total - previous_total) / previous_total) * 100
                    growth_data.append(growth)
                    growth_years.append(current_year)
            
            if not growth_data:
                logger.warning("No growth data available")
                return None
            
            # Create line chart with markers
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=growth_years,
                y=growth_data,
                mode='lines+markers',
                name='YoY Growth',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#667eea'),
                hovertemplate='<b>%{x}</b><br>Growth: %{y:+.1f}%<extra></extra>'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title={
                    'text': "Year-on-Year Growth in IT Job Vacancies",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Year",
                yaxis_title="Growth (%)",
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI YoY growth chart: {e}")
            return None

    def chart_ivi_occupation_state_heatmap(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create occupation vs state heatmap"""
        try:
            logger.info("Creating IVI occupation-state heatmap")
            
            # Calculate totals by occupation and state
            numeric_cols = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            heatmap_data = df.groupby(['ANZSCO_TITLE', 'state'])[numeric_cols].sum().sum(axis=1)
            
            # Create pivot table
            heatmap_pivot = heatmap_data.unstack(fill_value=0)
            
            # Create heatmap
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Viridis',
                hovertemplate='<b>%{y}</b><br>State: %{x}<br>Vacancies: %{z:,.0f}<extra></extra>',
                showscale=True,
                colorbar=dict(title="Job Vacancies")
            ))
            
            fig.update_layout(
                title={
                    'text': "IT Vacancies: Occupation vs State Heatmap",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="State/Territory",
                yaxis_title="Occupation",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=100, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI occupation-state heatmap: {e}")
            return None

    def chart_ivi_covid_impact(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create COVID impact analysis chart"""
        try:
            logger.info("Creating IVI COVID impact chart")
            
            # Get date columns and calculate monthly totals
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            dates = pd.to_datetime(date_columns, errors='coerce')
            
            monthly_totals = []
            valid_dates = []
            
            for i, col in enumerate(date_columns):
                if i < len(dates) and pd.notna(dates[i]):
                    monthly_totals.append(df[col].sum())
                    valid_dates.append(dates[i])
            
            if not monthly_totals:
                logger.warning("No valid date data for COVID impact analysis")
                return None
            
            # Create grouped bar chart for pre/during/post COVID
            covid_start = pd.Timestamp('2020-03-01')
            covid_end = pd.Timestamp('2022-01-01')
            
            pre_covid = []
            during_covid = []
            post_covid = []
            
            for i, date in enumerate(valid_dates):
                if date < covid_start:
                    pre_covid.append(monthly_totals[i])
                elif date >= covid_start and date < covid_end:
                    during_covid.append(monthly_totals[i])
                else:
                    post_covid.append(monthly_totals[i])
            
            # Calculate averages
            pre_avg = sum(pre_covid) / len(pre_covid) if pre_covid else 0
            during_avg = sum(during_covid) / len(during_covid) if during_covid else 0
            post_avg = sum(post_covid) / len(post_covid) if post_covid else 0
            
            # Create bar chart
            fig = go.Figure()
            
            periods = ['Pre-COVID', 'During COVID', 'Post-COVID']
            averages = [pre_avg, during_avg, post_avg]
            colors = ['#28a745', '#dc3545', '#007bff']
            
            fig.add_trace(go.Bar(
                x=periods,
                y=averages,
                marker=dict(color=colors),
                hovertemplate='<b>%{x}</b><br>Avg Vacancies: %{y:,.0f}<extra></extra>',
                text=[f'{avg:,.0f}' for avg in averages],
                textposition='auto'
            ))
            
            # Add percentage change annotations
            if pre_avg > 0:
                decline_pct = ((during_avg - pre_avg) / pre_avg) * 100
                recovery_pct = ((post_avg - during_avg) / during_avg) * 100 if during_avg > 0 else 0
                
                fig.add_annotation(
                    x=1, y=during_avg,
                    text=f"{decline_pct:+.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    ax=0, ay=-40
                )
                
                fig.add_annotation(
                    x=2, y=post_avg,
                    text=f"{recovery_pct:+.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="blue",
                    ax=0, ay=-40
                )
            
            fig.update_layout(
                title={
                    'text': "COVID Impact on IT Job Market",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Period",
                yaxis_title="Average Monthly Vacancies",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI COVID impact chart: {e}")
            return None

    # ==================== IVI Forecasting Visualizations ====================
    
    def chart_ivi_growth_rate_summary(self, df: pd.DataFrame, forecast_periods: int = 8) -> Optional[go.Figure]:
        """Create growth rate summary visualization"""
        try:
            logger.info(f"Creating IVI growth rate summary chart ({forecast_periods} periods)")
            
            # Get date columns and calculate monthly totals
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            dates = pd.to_datetime(date_columns, errors='coerce')
            
            monthly_totals = []
            valid_dates = []
            
            for i, col in enumerate(date_columns):
                if i < len(dates) and pd.notna(dates[i]):
                    monthly_totals.append(df[col].sum())
                    valid_dates.append(dates[i])
            
            if not monthly_totals or len(monthly_totals) < 12:
                logger.warning("Insufficient data for forecasting")
                return None
            
            # Create time series data
            ts_data = pd.DataFrame({
                'ds': valid_dates,
                'y': monthly_totals
            }).sort_values('ds')
            
            # Simple forecasting
            import numpy as np
            from sklearn.linear_model import LinearRegression
            
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data['y'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Get last historical value and forecast endpoint
            last_historical = ts_data['y'].iloc[-1]
            data_length = int(len(ts_data))
            forecast_endpoint = model.predict([[data_length + forecast_periods - 1]])[0]
            
            # Calculate growth rate
            growth_rate = ((forecast_endpoint - last_historical) / last_historical) * 100
            
            # Create figure with growth rate annotation
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=ts_data['ds'],
                y=ts_data['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea')
            ))
            
            # Forecast data
            last_date = ts_data['ds'].iloc[-1]
            # Use pd.offsets.MonthBegin for pandas 2.0+ compatibility
            from pandas.tseries.offsets import MonthBegin
            forecast_dates = pd.date_range(
                start=last_date + MonthBegin(1),
                periods=forecast_periods,
                freq='ME'
            )
            
            data_length = int(len(ts_data))
            forecast_X = np.arange(data_length, data_length + forecast_periods).reshape(-1, 1)
            forecast_y = model.predict(forecast_X)
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_y,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff6b6b', width=3, dash='dash'),
                marker=dict(size=6, color='#ff6b6b')
            ))
            
            # Add growth rate annotation
            fig.add_annotation(
                x=forecast_dates[-1],
                y=forecast_endpoint,
                text=f"Growth: {growth_rate:+.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green" if growth_rate > 0 else "red",
                ax=0, ay=-40,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
            
            fig.update_layout(
                title={
                    'text': f"IT Job Vacancies Growth Rate Summary ({forecast_periods} Periods)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Date",
                yaxis_title="Job Vacancies",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI growth rate summary chart: {e}")
            return None

    def chart_ivi_forecast_by_state(self, df: pd.DataFrame, forecast_periods: int = 8) -> Optional[go.Figure]:
        """Create forecast by state visualization"""
        try:
            logger.info(f"Creating IVI forecast by state chart ({forecast_periods} periods)")
            
            # Filter out AUST and get state data
            state_data = df[df['state'] != 'AUST'].copy()
            if state_data.empty:
                logger.warning("No state data available for forecasting")
                return None
            
            # Get date columns
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            dates = pd.to_datetime(date_columns, errors='coerce')
            
            # Calculate totals by state
            state_totals = {}
            for state in state_data['state'].unique():
                state_df = state_data[state_data['state'] == state]
                totals = []
                valid_dates = []
                
                for i, col in enumerate(date_columns):
                    if i < len(dates) and pd.notna(dates[i]):
                        totals.append(state_df[col].sum())
                        valid_dates.append(dates[i])
                
                if totals:
                    state_totals[state] = {
                        'dates': valid_dates,
                        'values': totals
                    }
            
            if not state_totals:
                logger.warning("No valid state data for forecasting")
                return None
            
            # Create figure
            fig = go.Figure()
            
            # State mapping for display names
            state_names = {
                'NSW': 'New South Wales',
                'VIC': 'Victoria',
                'QLD': 'Queensland',
                'WA': 'Western Australia',
                'SA': 'South Australia',
                'TAS': 'Tasmania',
                'NT': 'Northern Territory',
                'ACT': 'Australian Capital Territory'
            }
            
            colors = ['#667eea', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
            
            # Forecast for each state
            import numpy as np
            from sklearn.linear_model import LinearRegression
            
            for i, (state, data) in enumerate(state_totals.items()):
                if len(data['values']) < 6:  # Need minimum data points
                    continue
                
                # Prepare data for regression
                X = np.arange(len(data['values'])).reshape(-1, 1)
                y = np.array(data['values'])
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['values'],
                    mode='lines+markers',
                    name=f"{state_names.get(state, state)} (Historical)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4, color=colors[i % len(colors)])
                ))
                
                # Forecast data
                last_date = data['dates'][-1]
                # Use pd.offsets.MonthBegin for pandas 2.0+ compatibility
                from pandas.tseries.offsets import MonthBegin
                forecast_dates = pd.date_range(
                    start=last_date + MonthBegin(1),
                    periods=forecast_periods,
                    freq='ME'
                )
                
                data_length = int(len(data['values']))
                forecast_X = np.arange(data_length, data_length + forecast_periods).reshape(-1, 1)
                forecast_y = model.predict(forecast_X)
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_y,
                    mode='lines+markers',
                    name=f"{state_names.get(state, state)} (Forecast)",
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    marker=dict(size=4, color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title={
                    'text': f"IT Job Vacancies Forecast by State ({forecast_periods} Periods)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Date",
                yaxis_title="Job Vacancies",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI forecast by state chart: {e}")
            return None

    def chart_ivi_forecast_by_occupation(self, df: pd.DataFrame, forecast_periods: int = 8, top_n: int = 5) -> Optional[go.Figure]:
        """Create forecast by occupation visualization"""
        try:
            logger.info(f"Creating IVI forecast by occupation chart ({forecast_periods} periods, top {top_n})")
            
            # Get date columns
            date_columns = [col for col in df.columns if col not in ['ANZSCO_CODE', 'ANZSCO_TITLE', 'state']]
            dates = pd.to_datetime(date_columns, errors='coerce')
            
            # Calculate totals by occupation
            occupation_totals = {}
            for occupation in df['ANZSCO_TITLE'].unique():
                occ_df = df[df['ANZSCO_TITLE'] == occupation]
                totals = []
                valid_dates = []
                
                for i, col in enumerate(date_columns):
                    if i < len(dates) and pd.notna(dates[i]):
                        totals.append(occ_df[col].sum())
                        valid_dates.append(dates[i])
                
                if totals:
                    occupation_totals[occupation] = {
                        'dates': valid_dates,
                        'values': totals
                    }
            
            if not occupation_totals:
                logger.warning("No valid occupation data for forecasting")
                return None
            
            # Get top N occupations by total vacancies
            total_vacancies = {occ: sum(data['values']) for occ, data in occupation_totals.items()}
            top_occupations = sorted(total_vacancies.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Create figure
            fig = go.Figure()
            
            colors = ['#667eea', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            # Forecast for each top occupation
            import numpy as np
            from sklearn.linear_model import LinearRegression
            
            for i, (occupation, _) in enumerate(top_occupations):
                data = occupation_totals[occupation]
                
                if len(data['values']) < 6:  # Need minimum data points
                    continue
                
                # Prepare data for regression
                X = np.arange(len(data['values'])).reshape(-1, 1)
                y = np.array(data['values'])
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['values'],
                    mode='lines+markers',
                    name=f"{occupation[:30]}... (Historical)" if len(occupation) > 30 else f"{occupation} (Historical)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4, color=colors[i % len(colors)])
                ))
                
                # Forecast data
                last_date = data['dates'][-1]
                # Use pd.offsets.MonthBegin for pandas 2.0+ compatibility
                from pandas.tseries.offsets import MonthBegin
                forecast_dates = pd.date_range(
                    start=last_date + MonthBegin(1),
                    periods=forecast_periods,
                    freq='ME'
                )
                
                data_length = int(len(data['values']))
                forecast_X = np.arange(data_length, data_length + forecast_periods).reshape(-1, 1)
                forecast_y = model.predict(forecast_X)
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_y,
                    mode='lines+markers',
                    name=f"{occupation[:30]}... (Forecast)" if len(occupation) > 30 else f"{occupation} (Forecast)",
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    marker=dict(size=4, color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title={
                    'text': f"IT Job Vacancies Forecast by Top {top_n} Occupations ({forecast_periods} Periods)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2d3748'}
                },
                xaxis_title="Date",
                yaxis_title="Job Vacancies",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2d3748'),
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(color='#2d3748'),
                    tickfont=dict(color='#2d3748'),
                    gridcolor='lightgray'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IVI forecast by occupation chart: {e}")
            return None


