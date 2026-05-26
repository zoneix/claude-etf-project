"""Monte Carlo Portfolio Simulator — Streamlit front-end"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    FREQ,
    compute_path_drawdowns,
    get_price_history,
    get_ticker_info,
    portfolio_div_yield,
    portfolio_expense_ratio,
    run_simulation,
)

warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Portfolio Simulator",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 340px; background: #0d1117; }
        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem; font-weight: 700; }
        div[data-testid="stMetricDelta"] { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── UI constants ─────────────────────────────────────────────────────────────
COLORS = px.colors.qualitative.Set2
HIST_PERIODS = ["1y", "2y", "3y", "5y", "10y", "max"]

_PERIOD_TRADING_DAYS = {
    "1y": 252, "2y": 504, "3y": 756, "5y": 1260, "10y": 2520, "max": 9999,
}


# ─── Cached data helpers ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker: str) -> Optional[dict]:
    """Thin cached wrapper around simulation.get_ticker_info (5-min TTL)."""
    return get_ticker_info(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: tuple, period: str) -> Optional[pd.DataFrame]:
    """Thin cached wrapper around simulation.get_price_history (5-min TTL)."""
    return get_price_history(tickers, period)


# ─── Session state ────────────────────────────────────────────────────────────
if "holdings" not in st.session_state:
    st.session_state.holdings = []

if "sim_data" not in st.session_state:
    st.session_state.sim_data = None

# Track the last DRIP setting so we can invalidate stale results on change.
if "last_drip" not in st.session_state:
    st.session_state.last_drip = True

# Track last ER delta to invalidate stale results when the override changes.
if "last_er_delta" not in st.session_state:
    st.session_state.last_er_delta = 0.0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Portfolio Builder")
    st.divider()

    # ── Add Holding ───────────────────────────────────────────────────────────
    st.markdown("### Add Holding")
    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="e.g. SPY, AAPL, QQQ, BTC-USD",
        help="Enter any stock or ETF ticker symbol",
    ).strip().upper()

    specify_as = st.radio(
        "Specify holding as",
        ["Dollar Value ($)", "Share / Unit Count"],
        horizontal=True,
    )
    is_dollar = specify_as.startswith("Dollar")

    hold_amount = st.number_input(
        "Amount ($)" if is_dollar else "Shares / Units",
        min_value=0.0,
        value=10_000.0 if is_dollar else 10.0,
        step=500.0 if is_dollar else 0.5,
        format="%.2f",
    )

    if st.button("Add to Portfolio", type="primary", use_container_width=True):
        if not ticker_input:
            st.warning("Enter a ticker symbol first.")
        else:
            with st.spinner(f"Looking up {ticker_input}..."):
                info = fetch_info(ticker_input)

            if info is None:
                st.error(
                    f"Could not find **{ticker_input}**. "
                    "Check the symbol and try again (e.g. SPY, QQQ, AAPL)."
                )
            else:
                value = hold_amount if is_dollar else hold_amount * info["price"]
                shares = hold_amount / info["price"] if is_dollar else hold_amount

                existing = next(
                    (h for h in st.session_state.holdings if h["ticker"] == info["ticker"]),
                    None,
                )
                if existing:
                    existing["shares"] += shares
                    existing["value"] = existing["shares"] * existing["price"]
                    st.toast(f"Updated {info['ticker']}", icon="✅")
                else:
                    st.session_state.holdings.append(
                        {**info, "shares": shares, "value": value}
                    )
                    st.toast(f"Added {info['name']}", icon="✅")
                st.rerun()

    # ── Holdings list ─────────────────────────────────────────────────────────
    if st.session_state.holdings:
        st.divider()
        st.markdown("### Current Holdings")
        total_sidebar = sum(h["value"] for h in st.session_state.holdings)

        for i, h in enumerate(st.session_state.holdings):
            pct = h["value"] / total_sidebar * 100
            dy = h.get("div_yield", 0.0)
            c1, c2 = st.columns([5, 1])
            with c1:
                # Use st.markdown + st.caption instead of mixing HTML entities
                # with Markdown bold/code spans.  Streamlit's markdown renderer
                # outputs the literal entity text rather than a space character.
                dy_str = f"  ·  yield {dy:.2%}" if dy > 0 else ""
                er = h.get("expense_ratio", 0.0)
                er_str = f"  ·  ER {er:.2%}" if er > 0 else ""
                st.markdown(f"**{h['ticker']}** `{pct:.1f}%`")
                st.caption(
                    f"${h['value']:,.2f}  ·  {h['shares']:.4f} units"
                    f"  @  ${h['price']:.2f}{dy_str}{er_str}"
                )
            with c2:
                if st.button("x", key=f"rm_{i}", help=f"Remove {h['ticker']}"):
                    st.session_state.holdings.pop(i)
                    st.session_state.sim_data = None
                    st.rerun()

        st.metric("Portfolio Total", f"${total_sidebar:,.2f}")

        if st.button("Clear All Holdings", use_container_width=True):
            st.session_state.holdings.clear()
            st.session_state.sim_data = None
            st.rerun()

    # ── Contributions ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Periodic Contributions")
    contrib_enabled = st.toggle("Enable contributions", value=True)
    contrib_freq = st.selectbox("Frequency", list(FREQ.keys()), index=1)
    contrib_amt = st.number_input(
        "Amount per period ($)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        disabled=not contrib_enabled,
    )

    # ── DRIP ──────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Dividend Reinvestment (DRIP)")
    drip_enabled = st.toggle(
        "Reinvest dividends (DRIP)",
        value=True,
        help=(
            "ON — dividends are automatically reinvested (total-return basis). "
            "OFF — dividends are paid out as cash and tracked separately; "
            "the portfolio grows on price return only."
        ),
    )

    # Show estimated portfolio yield when holdings are present.
    if st.session_state.holdings:
        p_yield = portfolio_div_yield(st.session_state.holdings)
        if p_yield > 0:
            label = "Est. portfolio yield"
            if drip_enabled:
                st.caption(f"{label}: **{p_yield:.2%}** (reinvested)")
            else:
                st.caption(f"{label}: **{p_yield:.2%}** (paid as cash)")
        else:
            st.caption("No dividend yield detected in current holdings.")

    # Invalidate stale results when the DRIP toggle flips.
    if drip_enabled != st.session_state.last_drip:
        st.session_state.last_drip = drip_enabled
        st.session_state.sim_data = None

    # ── Expense Ratio ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Expense Ratio")

    p_er = portfolio_expense_ratio(st.session_state.holdings) if st.session_state.holdings else 0.0

    if st.session_state.holdings:
        if p_er > 0:
            st.caption(
                f"Portfolio-weighted ER: **{p_er:.2%}**. "
                "Historical prices already reflect this cost — "
                "no extra drag is applied by default."
            )
        else:
            st.caption(
                "No ER data found for current holdings. "
                "Individual stocks have no expense ratio."
            )

    custom_er_toggle = st.toggle(
        "Override ER (what-if)",
        value=False,
        help=(
            "Model a different expense ratio than the one embedded in "
            "historical prices.  Only the *difference* between your custom ER "
            "and the historical ER is applied as extra drag (positive) "
            "or boost (negative) in the simulation."
        ),
    )

    if custom_er_toggle and st.session_state.holdings:
        custom_er_pct = st.number_input(
            "Custom annual ER (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(round(p_er * 100, 3)),
            step=0.01,
            format="%.3f",
            help="Enter as a percentage, e.g. 0.090 for 0.09 %.",
        )
        custom_er = custom_er_pct / 100.0
        er_delta = custom_er - p_er
        if er_delta > 1e-6:
            st.caption(f"Extra drag vs historical: **+{er_delta:.3%}** per year.")
        elif er_delta < -1e-6:
            st.caption(f"ER reduction vs historical: **{er_delta:.3%}** per year.")
        else:
            st.caption("Custom ER matches historical — no adjustment applied.")
    else:
        er_delta = 0.0

    # Invalidate stale results when the ER delta changes.
    if er_delta != st.session_state.last_er_delta:
        st.session_state.last_er_delta = er_delta
        st.session_state.sim_data = None

    # ── Inflation / Real Value ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Inflation Adjustment")
    show_real = st.toggle(
        "Show inflation-adjusted (real) values",
        value=False,
        help=(
            "Discount projected nominal values back to today's purchasing power. "
            "This is a display-only adjustment — simulation returns are unchanged."
        ),
    )
    if show_real:
        inflation_rate = st.number_input(
            "Annual inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.5,
            step=0.1,
            format="%.1f",
            help="Historical US CPI inflation has averaged ~3 % per year over the long run.",
        ) / 100.0
        if st.session_state.sim_data:
            _sim_years = st.session_state.sim_data["years"]
            _pv = 1 / (1 + inflation_rate) ** _sim_years
            st.caption(
                f"At **{inflation_rate:.1%}** p.a., $1 today ≈ "
                f"**${_pv:,.3f}** purchasing power in {_sim_years} years."
            )
        else:
            st.caption("Real values will appear once a simulation is run.")
    else:
        inflation_rate = 0.0

    # ── Simulation settings ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### Simulation Settings")
    sim_years = st.slider("Time Horizon (years)", 1, 40, 20)
    sim_n = st.select_slider(
        "Number of Simulations", options=[250, 500, 1_000, 2_000, 5_000], value=1_000
    )
    hist_period = st.selectbox(
        "Historical period for return statistics",
        HIST_PERIODS,
        index=3,
        help=(
            "Longer periods give more stable statistics but may include "
            "very different market regimes."
        ),
    )

    st.divider()
    run_btn = st.button(
        "Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=len(st.session_state.holdings) == 0,
    )


# ─── Main panel ───────────────────────────────────────────────────────────────
st.title("📈 Monte Carlo Portfolio Simulator")
st.caption(
    "Add any ETF or stock holdings, configure periodic contributions, "
    "and project future growth using correlated Monte Carlo simulation with real market data."
)

if not st.session_state.holdings:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1** — Type any ticker (SPY, AAPL, QQQ...) in the sidebar and add it to your portfolio.")
    c2.info("**Step 2** — Optionally enable periodic contributions and choose your time horizon.")
    c3.info("**Step 3** — Click **Run Simulation** to generate your Monte Carlo projection.")
    st.stop()

# ── Fetch prices ──────────────────────────────────────────────────────────────
tickers_key = tuple(h["ticker"] for h in st.session_state.holdings)
with st.spinner("Fetching live market data..."):
    prices_df = fetch_prices(tickers_key, hist_period)

if prices_df is None or prices_df.empty:
    st.error("Failed to fetch price data. Try a shorter historical period or verify your tickers.")
    st.stop()

holdings_df = pd.DataFrame(st.session_state.holdings)
total_portfolio = holdings_df["value"].sum()

# Warn about completely missing tickers.
available_tickers = [t for t in tickers_key if t in prices_df.columns]
if len(available_tickers) < len(tickers_key):
    missing = set(tickers_key) - set(available_tickers)
    st.warning(
        f"No price data found for: {', '.join(sorted(missing))}. "
        "These will be excluded from the simulation."
    )

# Info when the common date range was trimmed by a shorter-history ticker.
_expected_rows = _PERIOD_TRADING_DAYS.get(hist_period, 252)
if len(prices_df) < _expected_rows * 0.80:
    _oldest = prices_df.index[0].strftime("%b %d, %Y")
    st.info(
        f"Common date range starts **{_oldest}** because one or more holdings "
        "have a shorter history. All tickers are plotted from their shared start "
        "date to keep the chart comparable."
    )

# ── Portfolio overview metrics ────────────────────────────────────────────────
st.subheader("Portfolio Overview")
n_cols = min(len(st.session_state.holdings) + 1, 7)
metric_cols = st.columns(n_cols)

for i, h in enumerate(st.session_state.holdings[: n_cols - 1]):
    wt = h["value"] / total_portfolio * 100
    metric_cols[i].metric(
        h["ticker"],
        f"${h['value']:,.2f}",
        f"{wt:.1f}%  ·  {h['shares']:.4f} units",
    )
metric_cols[-1].metric("Total Portfolio", f"${total_portfolio:,.2f}")

# ── Historical performance + allocation charts ────────────────────────────────
prices_avail = prices_df[available_tickers].copy()
chart_l, chart_r = st.columns([3, 2])

with chart_l:
    st.subheader("Historical Performance (Normalized to 100)")
    # bfill() fills any leading NaN so iloc[0] is always a valid baseline.
    prices_plot = prices_avail.bfill()
    norm = prices_plot / prices_plot.iloc[0] * 100
    fig_hist = go.Figure()
    for ci, col in enumerate(norm.columns):
        fig_hist.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[col],
                name=col,
                line=dict(color=COLORS[ci % len(COLORS)], width=2.5),
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}  ·  %{{y:.1f}}<extra></extra>",
            )
        )
    fig_hist.update_layout(
        template="plotly_dark",
        height=370,
        yaxis_title="Normalized Value (base 100)",
        xaxis_title=None,
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=0, r=0, t=5, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_r:
    st.subheader("Allocation")
    fig_pie = go.Figure(
        go.Pie(
            labels=holdings_df["ticker"],
            values=holdings_df["value"],
            textinfo="label+percent",
            hole=0.45,
            marker_colors=COLORS[: len(holdings_df)],
            hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )
    fig_pie.update_layout(
        template="plotly_dark",
        height=370,
        showlegend=False,
        margin=dict(l=0, r=0, t=5, b=0),
        annotations=[
            dict(
                text=f"${total_portfolio:,.0f}",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="white"),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Historical return statistics ──────────────────────────────────────────────
with st.expander("Historical Return Statistics", expanded=False):
    daily_rets = prices_avail.pct_change().dropna()
    ann_ret = (1 + daily_rets.mean()) ** 252 - 1
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol

    stats_df = pd.DataFrame(
        {
            "Annualized Return": ann_ret.map("{:.2%}".format),
            "Annualized Volatility": ann_vol.map("{:.2%}".format),
            "Sharpe Ratio (rf=0)": sharpe.map("{:.2f}".format),
            "Div Yield": pd.Series(
                {
                    h["ticker"]: f"{h.get('div_yield', 0):.2%}"
                    for h in st.session_state.holdings
                    if h["ticker"] in daily_rets.columns
                }
            ),
            "Expense Ratio": pd.Series(
                {
                    h["ticker"]: f"{h.get('expense_ratio', 0):.2%}"
                    for h in st.session_state.holdings
                    if h["ticker"] in daily_rets.columns
                }
            ),
            "Data Points": daily_rets.count().map("{:,}".format),
        }
    )
    st.dataframe(stats_df, use_container_width=True)

    if len(available_tickers) > 1:
        corr = daily_rets.corr()
        st.markdown("**Correlation Matrix**")
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
        )

# ── Run simulation ────────────────────────────────────────────────────────────
if run_btn:
    if not available_tickers:
        st.error("No price data available for any of the selected tickers.")
    else:
        prices_sim = prices_avail[available_tickers].dropna()
        holding_vals = holdings_df[holdings_df["ticker"].isin(available_tickers)]["value"].values
        w = holding_vals / holding_vals.sum()
        contrib = contrib_amt if contrib_enabled else 0.0
        p_yield = portfolio_div_yield(st.session_state.holdings)

        with st.spinner(f"Running {sim_n:,} correlated simulations over {sim_years} years..."):
            try:
                portfolio_arr, div_cash_arr = run_simulation(
                    weights=w,
                    prices=prices_sim,
                    initial=total_portfolio,
                    contribution=contrib,
                    freq=contrib_freq,
                    years=sim_years,
                    n_sims=sim_n,
                    drip=drip_enabled,
                    annual_div_yield=p_yield,
                    er_delta=er_delta,
                )
                st.session_state.sim_data = {
                    "portfolio": portfolio_arr,
                    "div_cash": div_cash_arr,
                    "years": sim_years,
                    "n_sims": sim_n,
                    "contrib": contrib,
                    "freq": contrib_freq,
                    "initial": total_portfolio,
                    "drip": drip_enabled,
                    "div_yield": p_yield,
                    "er_delta": er_delta,
                }
                st.toast("Simulation complete!", icon="🎉")
            except ValueError as e:
                st.error(str(e))

# ── Display simulation results ────────────────────────────────────────────────
if st.session_state.sim_data:
    sd = st.session_state.sim_data

    # Handle legacy sim_data that pre-dates DRIP support (key was "array").
    if "portfolio" not in sd and "array" in sd:
        sd["portfolio"] = sd["array"]
        sd["div_cash"] = np.zeros_like(sd["array"])
        sd.setdefault("drip", True)
        sd.setdefault("div_yield", 0.0)

    portfolio = sd["portfolio"]
    div_cash  = sd["div_cash"]
    is_drip   = sd.get("drip", True)

    # Total wealth = portfolio (price return) + cumulative dividends paid out.
    # When drip=True, div_cash is all-zeros so total_wealth == portfolio.
    total_wealth = portfolio + div_cash

    spy_num = FREQ[sd["freq"]]
    n_steps = total_wealth.shape[1]
    t_ax = np.linspace(0, sd["years"], n_steps)

    p10 = np.percentile(total_wealth, 10, axis=0)
    p25 = np.percentile(total_wealth, 25, axis=0)
    p50 = np.percentile(total_wealth, 50, axis=0)
    p75 = np.percentile(total_wealth, 75, axis=0)
    p90 = np.percentile(total_wealth, 90, axis=0)

    total_contrib = sd["contrib"] * spy_num * sd["years"]
    total_invested = sd["initial"] + total_contrib

    drip_label = "DRIP ON — total return" if is_drip else "DRIP OFF — dividends paid as cash"
    _er_d = sd.get("er_delta", 0.0)
    er_label = f"  ·  ER delta {_er_d:+.2%}" if abs(_er_d) > 1e-6 else ""
    st.divider()
    st.subheader(
        f"Monte Carlo Results  —  {sd['years']}-Year Projection  "
        f"({sd['n_sims']:,} simulations  ·  {sd['freq']} contributions  ·  {drip_label}{er_label})"
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_metric_cols = 6 if not is_drip and sd["div_yield"] > 0 else 5
    m_cols = st.columns(n_metric_cols)

    m_data = [
        ("Initial Portfolio",  f"${sd['initial']:,.0f}", None),
        (
            "Total Contributions",
            f"${total_contrib:,.0f}",
            f"${sd['contrib']:,.0f} {sd['freq'].lower()}" if sd["contrib"] > 0 else "None",
        ),
        ("Total Invested",     f"${total_invested:,.0f}", None),
        (
            "Median Outcome",
            f"${p50[-1]:,.0f}",
            f"{(p50[-1] / total_invested - 1) * 100:+.0f}% vs invested",
        ),
        (
            "90th Percentile",
            f"${p90[-1]:,.0f}",
            f"{(p90[-1] / total_invested - 1) * 100:+.0f}% vs invested",
        ),
    ]
    if not is_drip and sd["div_yield"] > 0:
        median_div = np.median(div_cash[:, -1])
        m_data.append(
            ("Median Dividend Income", f"${median_div:,.0f}", f"{sd['div_yield']:.2%} yield")
        )

    for col, (label, val, delta) in zip(m_cols, m_data):
        col.metric(label, val, delta)

    # ── Fan chart ─────────────────────────────────────────────────────────────
    fig_fan = go.Figure()

    # Shaded confidence bands
    for lo, hi, alpha in [(p10, p90, 0.07), (p25, p75, 0.14)]:
        fig_fan.add_trace(
            go.Scatter(
                x=np.concatenate([t_ax, t_ax[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=f"rgba(99,179,237,{alpha})",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Percentile lines
    wealth_label = "Total Wealth" if not is_drip else "Portfolio"
    for y_vals, color, pct_name, dash in [
        (p90, "#f97316", "90th %ile", "dot"),
        (p75, "#facc15", "75th %ile", "solid"),
        (p50, "#4ade80", "Median",    "solid"),
        (p25, "#60a5fa", "25th %ile", "solid"),
        (p10, "#f87171", "10th %ile", "dot"),
    ]:
        fig_fan.add_trace(
            go.Scatter(
                x=t_ax,
                y=y_vals,
                name=f"{pct_name}",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=(
                    f"<b>{pct_name}</b>  ·  Year %{{x:.1f}}  ·  ${wealth_label}: $%{{y:,.0f}}<extra></extra>"
                ),
            )
        )

    # When DRIP is OFF, overlay the median portfolio-only (ex-div) line.
    if not is_drip and sd["div_yield"] > 0:
        port_med = np.percentile(portfolio, 50, axis=0)
        fig_fan.add_trace(
            go.Scatter(
                x=t_ax,
                y=port_med,
                name="Median Portfolio (ex-div)",
                line=dict(color="#a78bfa", width=1.5, dash="dash"),
                hovertemplate="<b>Portfolio (ex-div)</b>  ·  Year %{x:.1f}  ·  $%{y:,.0f}<extra></extra>",
            )
        )

    # Total invested reference line
    invested_curve = sd["initial"] + sd["contrib"] * spy_num * t_ax
    fig_fan.add_trace(
        go.Scatter(
            x=t_ax,
            y=invested_curve,
            name="Total Invested",
            line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
            hovertemplate="<b>Total Invested</b>  ·  $%{y:,.0f}<extra></extra>",
        )
    )

    # Inflation-adjusted (real) overlays — added when the sidebar toggle is ON.
    if show_real and inflation_rate > 0:
        # Discount each time point: real_value[t] = nominal[t] / (1 + r)^t
        discount_factors = (1 + inflation_rate) ** t_ax   # shape (n_steps,)
        real_p10 = p10 / discount_factors
        real_p50 = p50 / discount_factors
        real_p90 = p90 / discount_factors
        # Shaded real band (p10–p90)
        fig_fan.add_trace(
            go.Scatter(
                x=np.concatenate([t_ax, t_ax[::-1]]),
                y=np.concatenate([real_p90, real_p10[::-1]]),
                fill="toself",
                fillcolor="rgba(232,121,249,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name="Real band",
            )
        )
        for y_real, name_real, dash_real in [
            (real_p90, f"90th %ile (Real @{inflation_rate:.1%})", "dot"),
            (real_p50, f"Median (Real @{inflation_rate:.1%})",    "longdash"),
            (real_p10, f"10th %ile (Real @{inflation_rate:.1%})", "dot"),
        ]:
            fig_fan.add_trace(
                go.Scatter(
                    x=t_ax,
                    y=y_real,
                    name=name_real,
                    line=dict(color="#e879f9", width=1.5, dash=dash_real),
                    hovertemplate=(
                        f"<b>{name_real}</b>  ·  Year %{{x:.1f}}  ·  $%{{y:,.0f}}<extra></extra>"
                    ),
                )
            )

    # Faint sample paths
    rng_sample = np.random.default_rng(42)
    sample_idx = rng_sample.choice(sd["n_sims"], size=min(60, sd["n_sims"]), replace=False)
    for idx in sample_idx:
        fig_fan.add_trace(
            go.Scatter(
                x=t_ax,
                y=total_wealth[idx],
                line=dict(color="rgba(255,255,255,0.035)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    yaxis_title = "Total Wealth ($)" if not is_drip else "Portfolio Value ($)"
    fig_fan.update_layout(
        template="plotly_dark",
        height=560,
        xaxis_title="Years from Now",
        yaxis=dict(title=yaxis_title, tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # ── Milestone table ───────────────────────────────────────────────────────
    milestones = [y for y in [1, 3, 5, 10, 15, 20, 25, 30, 40] if y <= sd["years"]]
    rows = []
    for yr in milestones:
        si = min(int(yr * spy_num), n_steps - 1)
        inv = sd["initial"] + sd["contrib"] * spy_num * yr
        tw_sl = total_wealth[:, si]
        row = {
            "Year": yr,
            "Total Invested": f"${inv:,.0f}",
            "10th %ile": f"${np.percentile(tw_sl, 10):,.0f}",
            "25th %ile": f"${np.percentile(tw_sl, 25):,.0f}",
            "Median": f"${np.percentile(tw_sl, 50):,.0f}",
            "75th %ile": f"${np.percentile(tw_sl, 75):,.0f}",
            "90th %ile": f"${np.percentile(tw_sl, 90):,.0f}",
            "Mean": f"${np.mean(tw_sl):,.0f}",
        }
        if not is_drip and sd["div_yield"] > 0:
            row["Div Income (Med)"] = f"${np.median(div_cash[:, si]):,.0f}"
        if show_real and inflation_rate > 0:
            pv_factor = (1 + inflation_rate) ** yr
            row["Median (Real $)"] = f"${np.percentile(tw_sl, 50) / pv_factor:,.0f}"
            row["10th (Real $)"]   = f"${np.percentile(tw_sl, 10) / pv_factor:,.0f}"
            row["90th (Real $)"]   = f"${np.percentile(tw_sl, 90) / pv_factor:,.0f}"
        rows.append(row)

    milestone_caption = (
        f"Nominal projected values.  Real columns discounted at {inflation_rate:.1%} p.a."
        if show_real and inflation_rate > 0 else
        "All values nominal (not inflation-adjusted)."
    )
    st.subheader("Projected Values at Milestones")
    st.caption(milestone_caption)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Drawdown Analysis ──────────────────────────────────────────────────────
    with st.expander("📉 Drawdown Analysis", expanded=False):
        # Compute running drawdown (fraction) for every path × every time step.
        dd_arr = compute_path_drawdowns(total_wealth) * 100.0   # → percent

        # Maximum drawdown experienced by each path over the whole horizon.
        max_dd_per_path = dd_arr.max(axis=1)                    # shape (n_sims,)

        # Percentile bands across paths at each time step.
        dd_p10 = np.percentile(dd_arr, 10, axis=0)
        dd_p25 = np.percentile(dd_arr, 25, axis=0)
        dd_p50 = np.percentile(dd_arr, 50, axis=0)
        dd_p75 = np.percentile(dd_arr, 75, axis=0)
        dd_p90 = np.percentile(dd_arr, 90, axis=0)

        # Y-axis ceiling: 95th percentile of max drawdowns + 10 % buffer.
        dd_ymax = min(100.0, float(np.percentile(max_dd_per_path, 95)) * 1.10 + 0.5)

        dd_l, dd_r = st.columns([3, 2])

        with dd_l:
            st.subheader("Drawdown Over Time")
            st.caption(
                "Shows how far each simulated portfolio sits below its all-time "
                "high at every point in time.  0 % = at peak; 20 % = 20 % below peak."
            )
            fig_dd = go.Figure()

            # Shaded confidence bands (p10–p90 outer, p25–p75 inner).
            for lo, hi, alpha in [(dd_p10, dd_p90, 0.08), (dd_p25, dd_p75, 0.16)]:
                fig_dd.add_trace(
                    go.Scatter(
                        x=np.concatenate([t_ax, t_ax[::-1]]),
                        y=np.concatenate([hi, lo[::-1]]),
                        fill="toself",
                        fillcolor=f"rgba(248,113,113,{alpha})",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            for y_vals, color, dd_name, dash in [
                (dd_p90, "#f97316", "90th %ile", "dot"),
                (dd_p75, "#facc15", "75th %ile", "solid"),
                (dd_p50, "#4ade80", "Median",    "solid"),
                (dd_p25, "#60a5fa", "25th %ile", "solid"),
                (dd_p10, "#f87171", "10th %ile", "dot"),
            ]:
                fig_dd.add_trace(
                    go.Scatter(
                        x=t_ax,
                        y=y_vals,
                        name=dd_name,
                        line=dict(color=color, width=2, dash=dash),
                        hovertemplate=(
                            f"<b>{dd_name}</b>  ·  Year %{{x:.1f}}  ·  "
                            f"Drawdown: %{{y:.1f}}%<extra></extra>"
                        ),
                    )
                )

            fig_dd.update_layout(
                template="plotly_dark",
                height=370,
                xaxis_title="Years from Now",
                yaxis=dict(
                    title="Drawdown from Peak (%)",
                    tickformat=".0f",
                    ticksuffix="%",
                    # 0 at top (no drawdown = at peak), larger drawdown goes down.
                    range=[dd_ymax, 0],
                ),
                legend=dict(orientation="h", y=1.08, x=0),
                margin=dict(l=0, r=0, t=5, b=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with dd_r:
            st.subheader("Max Drawdown Distribution")
            st.caption(
                "Histogram of the single worst drawdown each simulation path "
                "experiences over the entire projection horizon."
            )
            fig_mdd = go.Figure(
                go.Histogram(
                    x=max_dd_per_path,
                    nbinsx=40,
                    marker_color="rgba(248,113,113,0.65)",
                    marker_line=dict(color="rgba(248,113,113,1.0)", width=0.5),
                    hovertemplate="Drawdown: %{x:.1f}%  ·  Paths: %{y}<extra></extra>",
                )
            )
            for pct_dd, col_dd, lbl_dd in [
                (50, "#4ade80", "Median"),
                (90, "#f97316", "90th %ile"),
            ]:
                v_dd = float(np.percentile(max_dd_per_path, pct_dd))
                fig_mdd.add_vline(
                    x=v_dd,
                    line=dict(color=col_dd, width=2, dash="dash"),
                    annotation_text=f"{lbl_dd}: {v_dd:.1f}%",
                    annotation_font=dict(color=col_dd, size=11),
                    annotation_position="top right",
                )
            fig_mdd.update_layout(
                template="plotly_dark",
                height=370,
                xaxis_title="Maximum Drawdown Experienced (%)",
                xaxis_ticksuffix="%",
                yaxis_title="# Simulations",
                margin=dict(l=0, r=0, t=5, b=0),
            )
            st.plotly_chart(fig_mdd, use_container_width=True)

        # Summary drawdown stats
        dd_stat_cols = st.columns(4)
        dd_stat_cols[0].metric(
            "Median Max Drawdown",
            f"{np.percentile(max_dd_per_path, 50):.1f}%",
        )
        dd_stat_cols[1].metric(
            "90th %ile Max Drawdown",
            f"{np.percentile(max_dd_per_path, 90):.1f}%",
            help="90 % of paths stayed within this max drawdown.",
        )
        dd_stat_cols[2].metric(
            "Paths > 20 % Drawdown",
            f"{np.mean(max_dd_per_path > 20) * 100:.1f}%",
            help="Fraction of simulations that experienced at least a 20 % peak-to-trough decline.",
        )
        dd_stat_cols[3].metric(
            "Worst Drawdown (all paths)",
            f"{max_dd_per_path.max():.1f}%",
        )

    # ── Distribution + probability stats ──────────────────────────────────────
    finals = total_wealth[:, -1]
    dist_l, dist_r = st.columns([3, 2])

    with dist_l:
        st.subheader(f"Distribution of Final Outcomes (Year {sd['years']})")
        fig_dist = go.Figure(
            go.Histogram(
                x=finals,
                nbinsx=60,
                marker_color="rgba(99,179,237,0.65)",
                marker_line=dict(color="rgba(99,179,237,1)", width=0.5),
            )
        )
        for pct, col, label in [
            (10, "#f87171", "10th"),
            (50, "#4ade80", "Median"),
            (90, "#f97316", "90th"),
        ]:
            v = np.percentile(finals, pct)
            fig_dist.add_vline(
                x=v,
                line=dict(color=col, width=2, dash="dash"),
                annotation_text=f"{label}: ${v:,.0f}",
                annotation_font=dict(color=col, size=11),
                annotation_position="top right",
            )
        fig_dist.update_layout(
            template="plotly_dark",
            height=330,
            xaxis_title=f"{'Total Wealth' if not is_drip else 'Portfolio Value'} at Year {sd['years']} ($)",
            xaxis_tickformat="$,.0f",
            yaxis_title="# Simulations",
            margin=dict(l=0, r=0, t=5, b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with dist_r:
        st.subheader("Probability Analysis")
        prob_data = [
            ("Beat Total Invested",        f"{np.mean(finals > total_invested) * 100:.1f}%"),
            ("2x Return on Invested",      f"{np.mean(finals > total_invested * 2) * 100:.1f}%"),
            ("3x Return on Invested",      f"{np.mean(finals > total_invested * 3) * 100:.1f}%"),
            ("Loss vs Initial Capital",    f"{np.mean(finals < sd['initial']) * 100:.1f}%"),
            ("Expected (Mean) Outcome",    f"${np.mean(finals):,.0f}"),
            ("Std Deviation of Outcomes",  f"${np.std(finals):,.0f}"),
        ]
        for label, val in prob_data:
            st.metric(label, val)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Disclaimer:** This tool is for educational and informational purposes only. "
        "Monte Carlo simulations use historical return distributions and do not guarantee "
        "future performance. Past performance is not indicative of future results. "
        "This is not financial advice — consult a qualified financial advisor before "
        "making investment decisions."
    )
