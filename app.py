"""
Sales Data Analysis & Visualization - Streamlit dashboard.

Interactive version of the analysis in SalesAnalysisVisualization.ipynb.
Explores 2019 electronics-store sales: revenue trends, best month, best city,
a US sales map, best hour to advertise, products frequently bought together,
top products, correlations, and price-category breakdowns.
"""

from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sales Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_dark"
ACCENT = "#2DD4BF"

DATA_PATH = "data.csv"
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@st.cache_data(show_spinner="Loading and cleaning sales data...")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and reproduce the notebook's cleaning steps."""
    df = pd.read_csv(path, index_col=None)

    # Drop fully empty rows and the repeated header rows ("Order Date" text).
    df = df.dropna(how="all")
    df = df[df["Order Date"].str[0:2] != "Or"]

    # Correct dtypes.
    df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"])
    df["Price Each"] = pd.to_numeric(df["Price Each"])

    # Derived columns.
    df["Month"] = df["Order Date"].str[0:2].astype("int32")
    df["Sale"] = df["Quantity Ordered"] * df["Price Each"]

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%y %H:%M")
    df["Date"] = df["Order Date"].dt.date
    df["Hour"] = df["Order Date"].dt.hour
    df["Minute"] = df["Order Date"].dt.minute

    def get_city(address: str) -> str:
        return address.split(",")[1].strip()

    def get_state(address: str) -> str:
        return address.split(",")[2].split(" ")[1]

    df["State"] = df["Purchase Address"].apply(get_state)
    df["City"] = df["Purchase Address"].apply(
        lambda x: f"{get_city(x)} ({get_state(x)})"
    )

    # Price categories via binning.
    bins = [0, 500, 1000, float("inf")]
    labels = ["Cheap", "Middle-priced", "Expensive"]
    df["Price Category"] = pd.cut(df["Price Each"], bins=bins, labels=labels)

    return df


def _fmt_delta(curr: float, prev: float) -> str | None:
    """Percentage change vs. the previous period, or None when unavailable."""
    if prev in (0, None) or pd.isna(prev):
        return None
    return f"{(curr - prev) / prev * 100:+.1f}% vs. prev. period"


def kpi_row(df: pd.DataFrame, prev: pd.DataFrame | None = None) -> None:
    """Render the KPI cards, with optional deltas vs. a comparison period."""
    def agg(frame: pd.DataFrame) -> dict:
        revenue = frame["Sale"].sum()
        orders = frame["Order ID"].nunique()
        units = int(frame["Quantity Ordered"].sum())
        return {
            "revenue": revenue,
            "orders": orders,
            "units": units,
            "aov": revenue / orders if orders else 0,
            "products": frame["Product"].nunique(),
        }

    now = agg(df)
    before = agg(prev) if prev is not None and not prev.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Total Revenue",
        f"${now['revenue']:,.0f}",
        _fmt_delta(now["revenue"], before["revenue"]) if before else None,
    )
    c2.metric(
        "Orders",
        f"{now['orders']:,}",
        _fmt_delta(now["orders"], before["orders"]) if before else None,
    )
    c3.metric(
        "Units Sold",
        f"{now['units']:,}",
        _fmt_delta(now["units"], before["units"]) if before else None,
    )
    c4.metric(
        "Avg Order Value",
        f"${now['aov']:,.2f}",
        _fmt_delta(now["aov"], before["aov"]) if before else None,
    )
    c5.metric("Products", now["products"])


def tab_trend(df: pd.DataFrame) -> None:
    st.subheader("How did revenue trend over the year?")
    granularity = st.radio(
        "Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True
    )
    ts = df.set_index("Order Date")["Sale"]
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[granularity]
    series = ts.resample(rule).sum()
    plot_df = pd.DataFrame({"Date": series.index, "Sales": series.values})

    fig = px.area(plot_df, x="Date", y="Sales")
    fig.update_traces(line_color=ACCENT, fillcolor="rgba(45,212,191,0.18)")
    fig.update_layout(yaxis_title="Sales in USD ($)", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    if not series.empty:
        peak_day = series.idxmax()
        st.info(
            f"Peak {granularity.lower()} sales of **${series.max():,.0f}** "
            f"around **{peak_day:%b %d, %Y}**."
        )


def tab_monthly(df: pd.DataFrame) -> None:
    st.subheader("Which month had the highest sales?")
    monthly = (
        df.groupby("Month")["Sale"].sum().reindex(range(1, 13)).fillna(0)
    )
    plot_df = pd.DataFrame(
        {
            "Month": [MONTH_NAMES[m - 1] for m in monthly.index],
            "Sales": monthly.values,
        }
    )
    fig = px.bar(
        plot_df,
        x="Month",
        y="Sales",
        text_auto=".2s",
        color="Sales",
        color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis_title="Sales in USD ($)", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    best = monthly.idxmax()
    st.info(
        f"**{MONTH_NAMES[best - 1]}** was the best month with "
        f"**${monthly.max():,.0f}** in sales."
    )


def tab_city(df: pd.DataFrame) -> None:
    st.subheader("Which city had the highest sales?")
    city = df.groupby("City")["Sale"].sum().sort_values(ascending=False)
    fig = px.bar(
        x=city.index,
        y=city.values,
        text_auto=".2s",
        color=city.values,
        color_continuous_scale="Oranges",
    )
    fig.update_layout(
        xaxis_title="City",
        yaxis_title="Sales in USD ($)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.info(
        f"**{city.index[0]}** led with **${city.iloc[0]:,.0f}** in sales."
    )

    st.markdown("**Sales by state**")
    state = df.groupby("State")["Sale"].sum().reset_index()
    fig_map = px.choropleth(
        state,
        locations="State",
        locationmode="USA-states",
        color="Sale",
        scope="usa",
        color_continuous_scale="Oranges",
        labels={"Sale": "Sales ($)"},
    )
    fig_map.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig_map, width="stretch")


def tab_hourly(df: pd.DataFrame) -> None:
    st.subheader("What time should we advertise to maximize orders?")
    hourly = df.groupby("Hour")["Order ID"].count().reindex(range(24)).fillna(0)
    fig = px.line(
        x=hourly.index,
        y=hourly.values,
        markers=True,
    )
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Number of orders")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    peak = hourly.idxmax()
    st.info(
        f"Order volume peaks around **{peak}:00**. Advertising shortly before "
        "the late-morning (~11:00) and evening (~19:00) peaks is ideal."
    )


def tab_products(df: pd.DataFrame) -> None:
    st.subheader("How much of each product was sold?")
    grp = df.groupby("Product")
    quantity = grp["Quantity Ordered"].sum()
    prices = grp["Price Each"].mean()

    products = quantity.index.tolist()
    fig = go.Figure()
    fig.add_bar(
        x=products,
        y=quantity.values,
        name="Quantity Ordered",
        marker_color="mediumseagreen",
    )
    fig.add_trace(
        go.Scatter(
            x=products,
            y=prices.values,
            name="Avg Price ($)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="royalblue"),
        )
    )
    fig.update_layout(
        yaxis=dict(title="Quantity Ordered"),
        yaxis2=dict(title="Avg Price ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        xaxis_tickangle=-90,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Cheaper products (batteries, cables) sell in the highest quantities, "
        "while expensive items sell far fewer units."
    )

    st.markdown("**Top products by revenue**")
    table = (
        grp.agg(
            Units=("Quantity Ordered", "sum"),
            Revenue=("Sale", "sum"),
            Avg_Price=("Price Each", "mean"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
            "Avg_Price": st.column_config.NumberColumn(
                "Avg Price", format="$%.2f"
            ),
            "Units": st.column_config.NumberColumn(format="%,d"),
        },
    )


def tab_together(df: pd.DataFrame) -> None:
    st.subheader("Which products are most often bought together?")
    top_n = st.slider("Number of pairs to show", 5, 20, 10)

    dup = df[df["Order ID"].duplicated(keep=False)].copy()
    dup["Grouped"] = dup.groupby("Order ID")["Product"].transform(
        lambda x: ",".join(x)
    )
    grouped = dup[["Order ID", "Grouped"]].drop_duplicates()

    count = Counter()
    for row in grouped["Grouped"]:
        row_list = row.split(",")
        count.update(Counter(combinations(row_list, 2)))

    pairs = count.most_common(top_n)
    if not pairs:
        st.warning("No multi-item orders in the current selection.")
        return
    pair_df = pd.DataFrame(
        {
            "Pair": [f"{a}  +  {b}" for (a, b), _ in pairs],
            "Times bought together": [c for _, c in pairs],
        }
    )
    fig = px.bar(
        pair_df,
        x="Times bought together",
        y="Pair",
        orientation="h",
        text_auto=True,
        color="Times bought together",
        color_continuous_scale="Purples",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"), coloraxis_showscale=False
    )
    st.plotly_chart(fig, width="stretch")


def tab_correlation(df: pd.DataFrame) -> None:
    st.subheader("Correlation between numeric variables")
    numeric = df[["Quantity Ordered", "Price Each", "Month", "Hour", "Sale"]]
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Strong positive correlation between Sale and Price Each; other "
        "variables show weak relationships."
    )


def tab_treemap(df: pd.DataFrame) -> None:
    st.subheader("Sales distribution")
    view = st.radio(
        "View by",
        ["Product", "Price Category → Product"],
        horizontal=True,
    )
    if view == "Product":
        path = ["Product"]
        scale = "deep"
    else:
        path = ["Price Category", "Product"]
        scale = "magma"

    fig = px.treemap(
        df,
        path=path,
        values="Sale",
        color="Sale",
        color_continuous_scale=scale,
    )
    fig.update_layout(margin=dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Share of sales by price category**")
    cat = df.groupby("Price Category", observed=True)["Sale"].sum()
    pie = px.pie(
        names=cat.index,
        values=cat.values,
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    st.plotly_chart(pie, width="stretch")


def main() -> None:
    st.title("📊 Sales Data Analysis & Visualization")
    st.markdown(
        "Interactive dashboard for a 2019 electronics store's sales data. "
        "Use the sidebar filters to slice the analysis."
    )

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            f"Could not find `{DATA_PATH}`. Make sure it sits next to `app.py`."
        )
        st.stop()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    cities = sorted(df["City"].unique())
    sel_cities = st.sidebar.multiselect("City", options=cities, default=cities)

    products = sorted(df["Product"].unique())
    sel_products = st.sidebar.multiselect(
        "Product", options=products, default=products
    )

    st.sidebar.caption(
        "Data source: `data.csv` — original notebook: "
        "`SalesAnalysisVisualization.ipynb`"
    )

    base_mask = (
        df["City"].isin(sel_cities) & df["Product"].isin(sel_products)
    )
    mask = base_mask & df["Date"].between(start_date, end_date)
    fdf = df[mask]

    # Comparison window: same length, immediately before the selected range.
    from datetime import timedelta

    span = end_date - start_date
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - span
    prev_mask = base_mask & df["Date"].between(prev_start, prev_end)
    pdf = df[prev_mask]

    if fdf.empty:
        st.warning("No data matches the current filters. Widen your selection.")
        st.stop()

    kpi_row(fdf, pdf)
    st.divider()

    tabs = st.tabs(
        [
            "📈 Revenue Trend",
            "🗓️ Monthly Sales",
            "🏙️ City & Map",
            "🕑 Best Hour",
            "📦 Products",
            "🔗 Sold Together",
            "🔥 Correlation",
            "🌳 Treemap & Categories",
            "🔍 Raw Data",
        ]
    )
    with tabs[0]:
        tab_trend(fdf)
    with tabs[1]:
        tab_monthly(fdf)
    with tabs[2]:
        tab_city(fdf)
    with tabs[3]:
        tab_hourly(fdf)
    with tabs[4]:
        tab_products(fdf)
    with tabs[5]:
        tab_together(fdf)
    with tabs[6]:
        tab_correlation(fdf)
    with tabs[7]:
        tab_treemap(fdf)
    with tabs[8]:
        st.subheader("Filtered data preview")
        st.dataframe(fdf.head(1000), width="stretch")
        st.download_button(
            "Download filtered data (CSV)",
            fdf.to_csv(index=False).encode("utf-8"),
            file_name="filtered_sales.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
