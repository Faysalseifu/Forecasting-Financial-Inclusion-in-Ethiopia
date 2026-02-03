"""Streamlit dashboard for Ethiopia financial inclusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "ethiopia_fi_enriched_data.csv"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    access: list[float]
    usage: list[float]


@st.cache_data(show_spinner=False)
def load_access_series() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df = df[(df["indicator_code"] == "ACC_OWNERSHIP")]
        df = df[(df["gender"].fillna("all") == "all")]
        df = df[(df["location"].fillna("national") == "national")]
        df = df[df["value_numeric"].notna()].copy()
        df["year"] = pd.to_datetime(df["observation_date"]).dt.year
        df = df.groupby("year", as_index=False)["value_numeric"].mean()
        df = df.rename(columns={"value_numeric": "access"})
    else:
        df = pd.DataFrame(columns=["year", "access"])

    baseline = pd.DataFrame(
        {
            "year": [2014, 2017, 2021, 2024],
            "access": [22.0, 35.0, 46.0, 49.0],
        }
    )
    if df.empty:
        df = baseline
    else:
        df = (
            pd.concat([df, baseline])
            .drop_duplicates(subset=["year"], keep="last")
            .sort_values("year")
        )

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_usage_series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2014, 2017, 2021, 2024],
            "usage": [8.0, 12.0, 18.0, 21.0],
        }
    )


def build_forecast() -> pd.DataFrame:
    years = [2025, 2026, 2027]
    scenarios = {
        "Base": Scenario("Base", [50.0, 53.0, 56.0], [24.0, 28.0, 32.0]),
        "Optimistic": Scenario(
            "Optimistic", [52.0, 56.0, 60.0], [26.0, 33.0, 40.0]
        ),
        "Pessimistic": Scenario(
            "Pessimistic", [49.0, 51.0, 53.0], [23.0, 25.0, 28.0]
        ),
    }

    records: list[dict[str, float | int | str]] = []
    for scenario in scenarios.values():
        for year, access, usage in zip(years, scenario.access, scenario.usage):
            records.append(
                {
                    "year": year,
                    "scenario": scenario.name,
                    "access": access,
                    "usage": usage,
                }
            )
    return pd.DataFrame(records)


def create_trend_chart(access: pd.DataFrame, usage: pd.DataFrame, year_range: tuple[int, int]) -> go.Figure:
    merged = pd.merge(access, usage, on="year", how="outer").sort_values("year")
    merged = merged[(merged["year"] >= year_range[0]) & (merged["year"] <= year_range[1])]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["access"],
            mode="lines+markers",
            name="Access (Account Ownership)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["usage"],
            mode="lines+markers",
            name="Usage (Digital Payments)",
        )
    )

    events = {
        2021: "Telebirr Launch",
        2023: "M-Pesa Launch",
        2025: "NDPS 2026–2030",
    }
    for year, label in events.items():
        if year_range[0] <= year <= year_range[1]:
            fig.add_vline(x=year, line_dash="dot", line_color="#8f8f8f")
            fig.add_annotation(
                x=year,
                y=max(merged[["access", "usage"]].max()),
                text=label,
                showarrow=False,
                yanchor="bottom",
            )

    fig.update_layout(
        height=420,
        legend_title_text="Indicators",
        xaxis_title="Year",
        yaxis_title="Percent of adults (%)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def create_forecast_chart(forecast: pd.DataFrame, scenario: str) -> go.Figure:
    base = forecast[forecast["scenario"] == scenario]
    lower = forecast.groupby("year")["access"].min()
    upper = forecast.groupby("year")["access"].max()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lower.index,
            y=lower.values,
            line=dict(color="rgba(70,130,180,0.2)"),
            name="Access min",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=upper.index,
            y=upper.values,
            fill="tonexty",
            line=dict(color="rgba(70,130,180,0.2)"),
            name="Scenario range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=base["year"],
            y=base["access"],
            mode="lines+markers",
            name=f"Access ({scenario})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=base["year"],
            y=base["usage"],
            mode="lines+markers",
            name=f"Usage ({scenario})",
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title="Year",
        yaxis_title="Percent of adults (%)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def create_channel_chart() -> go.Figure:
    channel = pd.DataFrame(
        {
            "year": [2020, 2023, 2025],
            "registered_mmo_m": [12.2, 65.0, 139.5],
            "active_mmo_m": [3.0, 13.0, 28.0],
        }
    )
    channel["registered_mmo_m"] = channel["registered_mmo_m"].round(1)
    channel["active_mmo_m"] = channel["active_mmo_m"].round(1)
    channel_long = channel.melt(
        id_vars="year",
        var_name="metric",
        value_name="accounts_million",
    )
    channel_long["metric"] = channel_long["metric"].replace(
        {
            "registered_mmo_m": "Registered mobile money (M)",
            "active_mmo_m": "Active mobile money (M)",
        }
    )
    fig = px.bar(
        channel_long,
        x="year",
        y="accounts_million",
        color="metric",
        barmode="group",
        text_auto=True,
        title="Mobile money: registered vs. active",
    )
    fig.update_layout(height=380, xaxis_title="Year", yaxis_title="Accounts (millions)")
    return fig


def create_target_chart(current_access: float, current_usage: float) -> go.Figure:
    target_df = pd.DataFrame(
        {
            "metric": ["Access (NFIS-II 2025 goal)", "Usage (NDPS ambition)", "Access (2024)", "Usage (2024)"],
            "value": [70.0, 35.0, current_access, current_usage],
            "type": ["Target", "Target", "Current", "Current"],
        }
    )
    fig = px.bar(
        target_df,
        x="metric",
        y="value",
        color="type",
        text_auto=True,
        title="Progress toward access & usage goals",
    )
    fig.update_layout(height=380, xaxis_title="", yaxis_title="Percent of adults (%)")
    return fig


def render_overview_tab(access: pd.DataFrame, usage: pd.DataFrame) -> None:
    latest_access = access.loc[access["year"].idxmax(), "access"]
    latest_usage = usage.loc[usage["year"].idxmax(), "usage"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Access (2024)", f"{latest_access:.0f}%", "NFIS-II target 70% missed")
    col2.metric("Usage (2024)", f"{latest_usage:.0f}%", "+21% digital payments")
    col3.metric("Mobile money accounts", "139.5M", "2025 supply-side scale")
    col4.metric("NDPS 2030 target", "750% GDP", "Digital payments value")

    st.info(
        "Interoperable P2P transfers have surpassed ATM withdrawals, highlighting the shift to"
        " digital rails even as dormancy remains high."
    )

    col5, col6 = st.columns(2)
    col5.plotly_chart(create_target_chart(latest_access, latest_usage), use_container_width=True)

    progress = pd.DataFrame(
        {
            "year": [2014, 2017, 2021, 2024],
            "access": [22.0, 35.0, 46.0, latest_access],
            "goal": [70.0, 70.0, 70.0, 70.0],
        }
    )
    fig = px.line(
        progress,
        x="year",
        y=["access", "goal"],
        markers=True,
        title="NFIS-II access goal vs. actual",
    )
    fig.update_layout(height=380, yaxis_title="Percent of adults (%)")
    col6.plotly_chart(fig, use_container_width=True)


def render_trends_tab(access: pd.DataFrame, usage: pd.DataFrame) -> None:
    years = sorted(set(access["year"]).union(set(usage["year"])))
    min_year, max_year = min(years), max(years)
    year_range = st.slider("Select year range", min_year, max_year, (min_year, max_year))

    st.plotly_chart(create_trend_chart(access, usage, year_range), use_container_width=True)

    st.plotly_chart(create_channel_chart(), use_container_width=True)

    st.markdown(
        "**Note:** Registered mobile money accounts expanded rapidly (12.2M in 2020 to ~139.5M"
        " by 2025), but active usage remains ~15–25% of registrations."
    )


def render_forecasts_tab(forecast: pd.DataFrame) -> None:
    scenario = st.selectbox("Scenario", forecast["scenario"].unique().tolist(), index=0)
    st.plotly_chart(create_forecast_chart(forecast, scenario), use_container_width=True)

    subset = forecast[forecast["scenario"] == scenario].set_index("year")
    st.dataframe(subset[["access", "usage"]].style.format("{:.1f}"), use_container_width=True)

    st.markdown(
        "**Interpretation:** NDPS 2026–2030 could lift usage to 35%+ by 2027 if activation"
        " succeeds, while access improves more slowly due to dormancy and income constraints."
    )


def render_insights_tab(forecast: pd.DataFrame) -> None:
    scenario = st.radio("Scenario lens", forecast["scenario"].unique().tolist(), horizontal=True)
    subset = forecast[forecast["scenario"] == scenario]
    access_2027 = subset.loc[subset["year"] == 2027, "access"].iloc[0]
    usage_2027 = subset.loc[subset["year"] == 2027, "usage"].iloc[0]

    col1, col2 = st.columns(2)
    col1.progress(min(access_2027 / 70.0, 1.0))
    col1.caption(f"Access progress vs. 70% goal: {access_2027:.1f}%")
    col2.progress(min(usage_2027 / 35.0, 1.0))
    col2.caption(f"Usage progress vs. 35% benchmark: {usage_2027:.1f}%")

    st.markdown(
        "**Consortium questions answered**\n"
        "- **Drivers:** agents/infrastructure scale, NDPS policy reforms, Telebirr + M-Pesa launches.\n"
        "- **2025 changes:** NDPS momentum expected to lift usage, but activation lag persists.\n"
        "- **Outlook:** Optimistic 2027 access ~60% and usage ~40% if activation improves."
    )

    download = forecast.copy()
    download["generated_at"] = datetime.utcnow().strftime("%Y-%m-%d")
    csv_data = download.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast data",
        data=csv_data,
        file_name="ethiopia_fi_forecasts.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Ethiopia Financial Inclusion Dashboard", layout="wide")

    st.title("Forecasting Financial Inclusion in Ethiopia")
    st.caption(
        "Sources: Global Findex 2014–2024, NBE/NDPS (Dec 2025), operator releases."
    )

    access = load_access_series()
    usage = load_usage_series()
    forecast = build_forecast()

    tabs = st.tabs(["Overview", "Trends", "Forecasts", "Insights"])
    with tabs[0]:
        render_overview_tab(access, usage)
    with tabs[1]:
        render_trends_tab(access, usage)
    with tabs[2]:
        render_forecasts_tab(forecast)
    with tabs[3]:
        render_insights_tab(forecast)


if __name__ == "__main__":
    main()
