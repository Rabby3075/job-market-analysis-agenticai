"""
DataVisualizer agent

Interactive charts for ABS industry dataset.
Note: File saving has been removed per requirements; this agent focuses on in-memory charts.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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

        def _shade_covid(fig: go.Figure) -> None:
            fig.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="LightSalmon", opacity=0.2, line_width=0)

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
            _shade_covid(fig_idx)
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
            _shade_covid(fig_sm)
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


