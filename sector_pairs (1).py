"""
Sector Pairs - weekly relative-value dashboard and pair backtester.

    pip install streamlit yfinance pandas numpy plotly lxml
    streamlit run sector_pairs.py

Structure
    One tab per GICS sector, each loading on demand. Inside a tab you tab
    through that sector's S&P 500 constituents and see weekly price, RSI,
    MACD and rolling z-score against the sector SPDR.

    A backtest tab runs the long/short pair study for any ticker, entering
    at whatever z-state that pair sits in right now.

    A sector-vs-market tab does the same with the eleven SPDRs measured
    against SPY, QQQ, VBR or IWM.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
from constituents import (CONSTITUENTS, SECTOR_NAMES, TICKER_TO_ETF, TICKER_TO_NAME)

SECTOR_ETFS = list(SECTOR_NAMES.keys())
BROAD = ["SPY", "QQQ", "VBR", "IWM"]
Z_GRID = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

# The download is daily either way; "weekly" just resamples afterwards. Daily
# gives ~5x the observations and checks the stop far more precisely, at the
# cost of noisier crossings.
FREQ = {
    "Daily":  {"rule": None,    "ann": 252, "bar": "day",
               "z_default": 252, "z_max": 756, "z_step": 21,
               "hold_default": 252, "hold_max": 504, "hold_step": 21},
    "Weekly": {"rule": "W-FRI", "ann": 52,  "bar": "week",
               "z_default": 52,  "z_max": 156, "z_step": 4,
               "hold_default": 52,  "hold_max": 104, "hold_step": 4},
}


def label_for(t: str) -> str:
    """XLY -> 'XLY · Consumer Discretionary';  NVDA -> 'NVDA · NVIDIA'."""
    if t in SECTOR_NAMES:
        return f"{t} · {SECTOR_NAMES[t]}"
    if t in TICKER_TO_NAME:
        return f"{t} · {TICKER_TO_NAME[t]}"
    return t

INK, PANE, GRID = "#12151A", "#171B21", "#232830"
TEXT, MUTED, LINE = "#C8D0DA", "#6C7683", "#E8EDF3"
CHEAP, RICH, UP, DOWN = "#4CC9F0", "#F0A202", "#3DD68C", "#F2555A"
MONO = "ui-monospace, SFMono-Regular, 'JetBrains Mono', Menlo, monospace"


# ─────────────────────────────────────────────────────────────────────────────
# Constituents
# ─────────────────────────────────────────────────────────────────────────────
def constituents_for(etf: str) -> pd.DataFrame:
    rows = CONSTITUENTS.get(etf, [])
    return pd.DataFrame(rows, columns=["symbol", "name"])


# ─────────────────────────────────────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────────────────────────────────────
def _stooq(ticker: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    df = pd.read_csv(url, parse_dates=["Date"])
    return df.set_index("Date")["Close"].sort_index().rename(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_closes(tickers: tuple[str, ...], years: int = 12,
                freq: str = "Weekly") -> pd.DataFrame:
    """Adjusted closes at the chosen frequency. Yahoo first, Stooq for the rest.

    The network call is identical either way - always daily bars - so switching
    to daily costs nothing on the download."""
    tickers = tuple(dict.fromkeys(tickers))
    close = pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=f"{years}y", interval="1d",
                          auto_adjust=True, progress=False, threads=True)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw[["Close"]].copy()
                close.columns = [tickers[0]]
    except Exception:
        pass

    close = close.dropna(axis=1, how="all") if not close.empty else pd.DataFrame()
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        got = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_stooq, t): t for t in missing}
            for f in as_completed(futs):
                try:
                    got.append(f.result())
                except Exception:
                    pass
        if got:
            close = pd.concat([close] + got, axis=1) if not close.empty else pd.concat(got, axis=1)

    if close.empty:
        return close
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    rule = FREQ[freq]["rule"]
    out = close.resample(rule).last() if rule else close
    cutoff = out.index.max() - pd.Timedelta(days=365 * years)
    return out[out.index >= cutoff].dropna(axis=1, how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Indicators  (cross-validated against the daily build)
# ─────────────────────────────────────────────────────────────────────────────
def log_returns(px: pd.Series) -> pd.Series:
    return np.log(px / px.shift(1))


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = close.dropna()
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=c.index).reindex(close.index)
    d = c.diff()
    gain, loss = d.clip(lower=0).to_numpy(), (-d).clip(lower=0).to_numpy()
    ag, al = gain[1:n + 1].mean(), loss[1:n + 1].mean()
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(c)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return pd.Series(out, index=c.index).reindex(close.index)


def macd(close: pd.Series, f: int = 12, s: int = 26, g: int = 9):
    line = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
    sig = line.ewm(span=g, adjust=False).mean()
    return line, sig, line - sig


def rel_z(px: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    """Rolling z of the log price ratio. Index is the intersection of both."""
    j = pd.concat([px, bench], axis=1).dropna()
    if j.empty:
        return pd.Series(dtype=float)
    r = np.log(j.iloc[:, 0] / j.iloc[:, 1])
    mu = r.rolling(window, min_periods=window).mean()
    sd = r.rolling(window, min_periods=window).std(ddof=0)
    return (r - mu) / sd.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Pair backtest
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BTResult:
    trades: pd.DataFrame
    equity: pd.Series
    weekly: pd.Series
    stats: dict
    threshold: float
    direction: int


def run_pair_backtest(stock: pd.Series, bench: pd.Series, z: pd.Series,
                      threshold: float, direction: int,
                      stop: float = 0.10, exit_z: float = 0.0,
                      max_hold: int = 52, ann: int = 52) -> BTResult:
    """Dollar-neutral pair, entered whenever z reaches `threshold`.

    direction +1 = long stock / short bench   (used when z is stretched low)
    direction -1 = short stock / long bench   (z stretched high)

    Bar t's z uses data through t; the position is taken at t's close and the
    first return accrues over t -> t+1. No overlapping trades.
    Exits: z reverts through exit_z, cumulative spread <= -stop, or max_hold.
    """
    idx = z.dropna().index
    rs = log_returns(stock).reindex(idx)
    rb = log_returns(bench).reindex(idx)
    spread = direction * (rs - rb)          # log return of the pair
    zz = z.reindex(idx)

    stop_log = np.log(1.0 - stop)
    n = len(idx)
    strat = pd.Series(0.0, index=idx)
    trades, i = [], 1

    while i < n - 1:
        prev, cur = zz.iloc[i - 1], zz.iloc[i]
        if direction > 0:
            triggered = prev > threshold >= cur
        else:
            triggered = prev < threshold <= cur
        if not (triggered and np.isfinite(cur)):
            i += 1
            continue

        entry_i, cum, reason = i, 0.0, "max hold"
        j = i + 1
        while j < n:
            step = spread.iloc[j]
            if not np.isfinite(step):
                step = 0.0
            cum += step
            strat.iloc[j] = step
            if cum <= stop_log:
                reason = "stop"
                break
            if direction > 0 and zz.iloc[j] >= exit_z:
                reason = "target"
                break
            if direction < 0 and zz.iloc[j] <= exit_z:
                reason = "target"
                break
            if j - entry_i >= max_hold:
                reason = "max hold"
                break
            j += 1
        j = min(j, n - 1)

        trades.append({
            "entry": idx[entry_i], "exit": idx[j],
            "bars": j - entry_i,
            "entry_z": round(float(zz.iloc[entry_i]), 2),
            "exit_z": round(float(zz.iloc[j]), 2),
            "return_%": round((np.exp(cum) - 1) * 100, 2),
            "exit_reason": reason,
        })
        i = j + 1

    tdf = pd.DataFrame(trades)
    equity = strat.cumsum().pipe(np.exp)
    active = strat != 0

    stats: dict = {"trades": len(tdf)}
    if len(tdf):
        wins = tdf["return_%"] > 0
        stats.update({
            "win_rate_%": round(100 * wins.mean(), 1),
            "avg_trade_%": round(tdf["return_%"].mean(), 2),
            "median_trade_%": round(tdf["return_%"].median(), 2),
            "best_%": round(tdf["return_%"].max(), 2),
            "worst_%": round(tdf["return_%"].min(), 2),
            "avg_bars": round(tdf["bars"].mean(), 1),
            "stops_hit": int((tdf["exit_reason"] == "stop").sum()),
            "total_return_%": round((equity.iloc[-1] - 1) * 100, 2),
        })
        inv = strat[active]
        if len(inv) > 2 and inv.std(ddof=1) > 0:
            stats["sharpe_in_trade"] = round(
                float(inv.mean() / inv.std(ddof=1) * np.sqrt(ann)), 2)
        if strat.std(ddof=1) > 0:
            stats["sharpe_all_bars"] = round(
                float(strat.mean() / strat.std(ddof=1) * np.sqrt(ann)), 2)
        stats["time_in_market_%"] = round(100 * active.mean(), 1)
        dd = equity / equity.cummax() - 1
        stats["max_drawdown_%"] = round(float(dd.min()) * 100, 2)

    return BTResult(tdf, equity, strat, stats, threshold, direction)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def indicator_figure(px: pd.Series, bench: pd.Series, label: str, bench_label: str,
                     z_win: int, rsi_n: int, bar: str = "week") -> go.Figure:
    r = rsi(px, rsi_n)
    ml, sg, hist = macd(px)
    z = rel_z(px, bench, z_win)
    lr = log_returns(px)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.36, 0.18, 0.20, 0.26],
        subplot_titles=(f"{label} - {bar}ly close",
                        f"{bar.capitalize()}ly log return  ·  RSI({rsi_n}) overlaid right",
                        f"MACD 12/26/9 ({bar}ly)",
                        f"Rolling {z_win}-{bar} z-score vs {bench_label}"),
    )
    fig.add_trace(go.Scatter(x=px.index, y=px, mode="lines", line=dict(color=LINE, width=1.5),
                             name="Close"), row=1, col=1)
    fig.add_trace(go.Bar(x=lr.index, y=lr * 100, name="log ret %",
                         marker_color=np.where(lr >= 0, UP, DOWN),
                         marker_line_width=0, opacity=0.55), row=2, col=1)
    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", line=dict(color=CHEAP, width=1.2),
                             name="RSI", yaxis="y5"), row=2, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="hist",
                         marker_color=np.where(hist >= 0, UP, DOWN),
                         marker_line_width=0, opacity=0.5), row=3, col=1)
    fig.add_trace(go.Scatter(x=ml.index, y=ml, mode="lines", line=dict(color=LINE, width=1.2),
                             name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=sg.index, y=sg, mode="lines", line=dict(color=RICH, width=1.1),
                             name="signal"), row=3, col=1)
    if not z.dropna().empty:
        fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines", line=dict(color=TEXT, width=1.6),
                                 fill="tozeroy", fillcolor="rgba(200,208,218,0.10)",
                                 name="z"), row=4, col=1)
        for lvl, c in ((2, RICH), (-2, CHEAP)):
            fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dash"), row=4, col=1)
        fig.add_hline(y=0, line=dict(color=GRID, width=1), row=4, col=1)

    fig.update_layout(
        height=860, template="plotly_dark", paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(family=MONO, size=11, color=TEXT), showlegend=False,
        margin=dict(l=8, r=8, t=44, b=8), bargap=0, hovermode="x unified",
        yaxis5=dict(overlaying="y2", side="right", range=[0, 100],
                    showgrid=False, tickfont=dict(color=CHEAP, size=9)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    for a in fig.layout.annotations:
        a.font.update(size=11, color=MUTED)
    return fig


def equity_figure(res: BTResult, label: str = "") -> go.Figure:
    eq = res.equity
    dd = eq / eq.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.68, 0.32],
                        subplot_titles=(f"Pair equity curve{' - ' + label if label else ''}"
                                        "   (1.0 = flat, compounding only while in a trade)",
                                        "Drawdown"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", line=dict(color=LINE, width=1.7),
                             name="equity"), row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color=GRID, width=1), row=1, col=1)

    if not res.trades.empty:
        ent = eq.reindex(pd.DatetimeIndex(res.trades["entry"])).dropna()
        fig.add_trace(go.Scatter(
            x=ent.index, y=ent.values, mode="markers", name="entry",
            marker=dict(color=TEXT, size=7, symbol="circle-open", line=dict(width=1.4)),
            hovertemplate="entry %{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)
        for reason, colour, sym in (("stop", DOWN, "x"), ("target", UP, "triangle-up"),
                                    ("max hold", MUTED, "square")):
            sub = res.trades[res.trades["exit_reason"] == reason]
            if sub.empty:
                continue
            pts = eq.reindex(pd.DatetimeIndex(sub["exit"])).dropna()
            fig.add_trace(go.Scatter(
                x=pts.index, y=pts.values, mode="markers", name=reason,
                marker=dict(color=colour, size=8, symbol=sym),
                hovertemplate=reason + " %{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, mode="lines", line=dict(color=DOWN, width=1),
                             fill="tozeroy", fillcolor="rgba(242,85,90,0.18)",
                             name="drawdown", showlegend=False), row=2, col=1)
    fig.update_layout(height=470, template="plotly_dark", paper_bgcolor=INK,
                      plot_bgcolor=INK, font=dict(family=MONO, size=11, color=TEXT),
                      margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    for a in fig.layout.annotations:
        a.font.update(size=11, color=MUTED)
    return fig


def history_figure(runs: list[dict]) -> go.Figure:
    """Every backtest run this session, overlaid for comparison."""
    fig = go.Figure()
    palette = [LINE, CHEAP, RICH, UP, "#B084F5", "#F26BA2", "#7FD1B9", "#D9C77E"]
    for i, r in enumerate(runs):
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name=r["label"],
                                 line=dict(width=1.6, color=palette[i % len(palette)])))
    fig.add_hline(y=1.0, line=dict(color=GRID, width=1))
    fig.update_layout(height=420, template="plotly_dark", paper_bgcolor=INK,
                      plot_bgcolor=INK, font=dict(family=MONO, size=11, color=TEXT),
                      margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
                      title=dict(text="Run history - equity curves from this session",
                                 font=dict(size=12, color=MUTED)),
                      legend=dict(orientation="h", y=-0.16, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig




# ─────────────────────────────────────────────────────────────────────────────
# Shared UI pieces
# ─────────────────────────────────────────────────────────────────────────────
def direction_for(z_now: float) -> tuple[int, str]:
    """Mean reversion: negative z -> long the name, positive z -> short it."""
    if not np.isfinite(z_now):
        return 0, "no z available"
    if z_now < 0:
        return +1, f"z {z_now:+.2f} — cheap vs benchmark → **LONG stock / SHORT benchmark**"
    return -1, f"z {z_now:+.2f} — rich vs benchmark → **SHORT stock / LONG benchmark**"


def benchmark_picker(key: str, default: str = "SPY") -> str:
    """Sector ETFs and broad indexes by name, plus a free-text 'Other' box."""
    opts = SECTOR_ETFS + BROAD + ["Other…"]
    idx = opts.index(default) if default in opts else 0
    pick = st.selectbox("Benchmark", opts, index=idx, key=f"{key}_bench",
                        format_func=label_for)
    if pick == "Other…":
        typed = st.text_input("Benchmark ticker", value="", key=f"{key}_bench_other",
                              placeholder="any ticker, e.g. SMH, GLD, MSFT").strip().upper()
        return typed or ""
    return pick


def z_grid_table(stock: pd.Series, bench: pd.Series, z: pd.Series,
                 stop: float, max_hold: int, ann: int, bar: str) -> pd.DataFrame:
    """Run the study at each z level on the grid. Sign sets the direction."""
    rows = []
    for thr in Z_GRID:
        d = +1 if thr < 0 else -1
        res = run_pair_backtest(stock, bench, z, thr, d, stop=stop,
                                max_hold=max_hold, ann=ann)
        s = res.stats
        rows.append({
            "z entry": f"{thr:+.0f}",
            "side": "long stock" if d > 0 else "short stock",
            "trades": s.get("trades", 0),
            "win %": s.get("win_rate_%"),
            "avg trade %": s.get("avg_trade_%"),
            "median %": s.get("median_%", s.get("median_trade_%")),
            "total %": s.get("total_return_%"),
            "max DD %": s.get("max_drawdown_%"),
            "Sharpe": s.get("sharpe_in_trade"),
            f"avg {bar}s": s.get("avg_bars"),
            "stops": s.get("stops_hit"),
        })
    return pd.DataFrame(rows)


def browse_panel(key: str, names: pd.DataFrame, bench_ticker: str,
                 prices: pd.DataFrame, z_win: int, rsi_n: int, bar: str) -> None:
    have = [t for t in names["symbol"] if t in prices.columns]
    if not have:
        st.warning("No price history returned for these names.")
        return
    st.session_state.setdefault(f"{key}_i", 0)
    st.session_state[f"{key}_i"] %= len(have)

    c1, c2, c3 = st.columns([1, 6, 1])
    if c1.button("←", key=f"{key}_prev", use_container_width=True):
        st.session_state[f"{key}_i"] -= 1
        st.rerun()
    tick = c2.selectbox("Constituent", have, index=st.session_state[f"{key}_i"],
                        key=f"{key}_sel", label_visibility="collapsed",
                        format_func=label_for)
    st.session_state[f"{key}_i"] = have.index(tick)
    if c3.button("→", key=f"{key}_next", use_container_width=True):
        st.session_state[f"{key}_i"] += 1
        st.rerun()

    px, bench = prices[tick].dropna(), prices[bench_ticker].dropna()
    z = rel_z(px, bench, z_win).dropna()
    zn = float(z.iloc[-1]) if len(z) else np.nan
    r = rsi(px, rsi_n).dropna()

    m = st.columns(4)
    m[0].metric("Last", f"{px.iloc[-1]:,.2f}")
    m[1].metric(f"1-{bar}", f"{(px.iloc[-1] / px.iloc[-2] - 1) * 100:+.2f}%" if len(px) > 1 else "-")
    m[2].metric(f"RSI({rsi_n})", f"{r.iloc[-1]:.1f}" if len(r) else "-")
    m[3].metric(f"Z vs {bench_ticker}", f"{zn:+.2f}" if np.isfinite(zn) else "-")
    st.caption(f"{len(have)} of {len(names)} names have usable history · "
               f"position {have.index(tick) + 1}/{len(have)}")
    st.plotly_chart(indicator_figure(px, bench, label_for(tick), label_for(bench_ticker),
                                     z_win, rsi_n, bar),
                    use_container_width=True, key=f"{key}_fig")


def show_backtest(stock_t: str, bench_t: str, prices: pd.DataFrame,
                  z_win: int, stop: float, max_hold: int, key: str,
                  ann: int = 52, bar: str = "week") -> None:
    if stock_t not in prices.columns or bench_t not in prices.columns:
        st.error(f"Missing price history for {stock_t} or {bench_t}.")
        return
    px, bench = prices[stock_t].dropna(), prices[bench_t].dropna()
    z = rel_z(px, bench, z_win)
    zc = z.dropna()
    if zc.empty:
        st.error("Not enough overlapping history to compute a z-score.")
        return

    z_now = float(zc.iloc[-1])
    direction, msg = direction_for(z_now)
    st.markdown(f"**{label_for(stock_t)}** vs **{label_for(bench_t)}** — {msg}")

    res = run_pair_backtest(px, bench, z, z_now, direction, stop=stop,
                            max_hold=max_hold, ann=ann)
    if res.trades.empty:
        st.warning(f"No historical entries at z {'≤' if direction > 0 else '≥'} {z_now:+.2f}. "
                   "The grid below still covers the standard levels.")
    else:
        s = res.stats
        a = st.columns(5)
        a[0].metric("Trades", s["trades"])
        a[1].metric("Win rate", f"{s['win_rate_%']}%")
        a[2].metric("Avg trade", f"{s['avg_trade_%']:+.2f}%")
        a[3].metric("Sharpe", s.get("sharpe_in_trade", "-"))
        a[4].metric("Max DD", f"{s['max_drawdown_%']:.2f}%")
        b = st.columns(5)
        b[0].metric("Median", f"{s['median_%']:+.2f}%")
        b[1].metric("Best / worst", f"{s['best_%']:+.1f} / {s['worst_%']:+.1f}%")
        b[2].metric("Avg hold", f"{s['avg_bars']:.0f} {bar}s")
        b[3].metric("Stops hit", f"{s['stops_hit']} / {s['trades']}")
        b[4].metric("Total", f"{s['total_return_%']:+.1f}%")

        label = f"{stock_t}/{bench_t} @ z{z_now:+.2f}"
        st.plotly_chart(equity_figure(res, label), use_container_width=True, key=f"{key}_eq")
        runs = st.session_state.setdefault("bt_runs", [])
        if not any(r["label"] == label for r in runs):
            runs.append({"label": label, "equity": res.equity, "stats": s})
        st.dataframe(res.trades, use_container_width=True, hide_index=True, height=260)

    st.divider()
    st.markdown("##### Same pair across the z grid")
    st.caption("Each row is an independent study: enter every time z reached that level, "
               "long the name below zero and short it above. Direction follows the sign.")
    grid = z_grid_table(px, bench, z, stop, max_hold, ann, bar)
    st.dataframe(
        grid.style.background_gradient(cmap="RdYlGn", subset=["avg trade %"])
                  .format(precision=2, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(
        f"Exit on reversion through 0, a {stop:.0%} adverse spread move, or {max_hold} {bar}s. "
        f"The stop is evaluated on {bar}ly closes, so a violent {bar} overshoots it. No costs, "
        "borrow or slippage. Rows with few trades are noise — treat anything under ~30 "
        "trades as unmeasured, however good the win rate looks."
    )

    runs = st.session_state.get("bt_runs", [])
    if len(runs) > 1:
        st.divider()
        st.plotly_chart(history_figure(runs), use_container_width=True, key=f"{key}_hist")
        st.dataframe(pd.DataFrame([
            {"pair": r["label"], **{k: r["stats"].get(k) for k in
             ("trades", "win_rate_%", "avg_trade_%", "total_return_%",
              "max_drawdown_%", "sharpe_in_trade")}} for r in runs]),
            use_container_width=True, hide_index=True)
        if st.button("Clear run history", key=f"{key}_clear"):
            st.session_state["bt_runs"] = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sector Pairs", layout="wide")
st.markdown(f"""<style>
 .stApp {{ background:{INK}; }}
 html, body, [class*="css"] {{ font-family:{MONO}; }}
 [data-testid="stMetricValue"] {{ font-size:1.1rem; }}
 button[data-baseweb="tab"] {{ font-size:0.78rem; }}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Sector Pairs")
    freq = st.radio("Bar frequency", list(FREQ), index=1, horizontal=True,
                    help="The download is daily either way. Weekly resamples to "
                         "Friday closes; daily keeps every bar.")
    F = FREQ[freq]
    bar, ann = F["bar"], F["ann"]

    years = st.slider("History (years)", 3, 20, 12)
    z_win = st.slider(f"Z window ({bar}s)", F["z_step"] * 2, F["z_max"],
                      F["z_default"], step=F["z_step"])
    rsi_n = st.slider(f"RSI length ({bar}s)", 5, 30, 14)
    st.divider()
    st.markdown("**Backtest rules**")
    stop = st.slider("Adverse-spread stop", 0.02, 0.30, 0.10, step=0.01, format="%.2f")
    max_hold = st.slider(f"Max hold ({bar}s)", F["hold_step"], F["hold_max"],
                         F["hold_default"], step=F["hold_step"])
    st.caption(f"{z_win} {bar}s ≈ {z_win / (252 if bar == 'day' else 52):.1f} years of lookback.")
    st.divider()
    st.caption("Constituents are a static snapshot in constituents.py — no external "
               "lookup, so nothing to fail. Edit that file to change the universe.")

tabs = st.tabs([f"{e} {SECTOR_NAMES[e]}" for e in SECTOR_ETFS]
               + ["◆ Backtest", "◆ Sectors vs market"])

for etf, tab in zip(SECTOR_ETFS, tabs[:len(SECTOR_ETFS)]):
    with tab:
        st.markdown(f"#### {SECTOR_NAMES[etf]} — constituents vs {etf}")
        flag = f"load_{etf}"
        if not st.session_state.get(flag) and not st.button(f"Load {SECTOR_NAMES[etf]}",
                                                            key=f"btn_{etf}"):
            st.caption("Not loaded. Click to fetch prices for this sector.")
            continue
        st.session_state[flag] = True
        names = constituents_for(etf)
        if names.empty:
            st.error(f"No constituents defined for {etf} in constituents.py.")
            continue
        prices = load_closes(tuple(names["symbol"]) + (etf,), years, freq)
        if prices.empty or etf not in prices.columns:
            st.error(f"Price download failed for {etf}.")
            continue
        browse_panel(f"sec_{etf}", names, etf, prices, z_win, rsi_n, bar)

with tabs[len(SECTOR_ETFS)]:
    st.markdown("#### Pair backtest — entry at the current z-state")
    c1, c2 = st.columns([2, 3])
    with c1:
        tk = st.text_input("Ticker", value="NVDA", key="bt_tk").strip().upper()
    with c2:
        bench_t = benchmark_picker("bt", default=TICKER_TO_ETF.get(tk, "SPY"))
    if st.button("Run backtest", key="bt_run", type="primary"):
        if not bench_t:
            st.error("Enter a benchmark ticker.")
        else:
            show_backtest(tk, bench_t, load_closes((tk, bench_t), years, freq),
                          z_win, stop, max_hold, key="bt", ann=ann, bar=bar)

with tabs[len(SECTOR_ETFS) + 1]:
    st.markdown("#### All eleven sectors vs a broad index")
    c1, c2 = st.columns([2, 3])
    idx = c1.selectbox("Index", BROAD + ["Other…"], key="bm_idx")
    if idx == "Other…":
        idx = c2.text_input("Index ticker", value="", key="bm_other",
                            placeholder="any ticker").strip().upper()
    if st.session_state.get("bm_loaded") or st.button("Run all sectors", key="bm_btn",
                                                      type="primary"):
        if not idx:
            st.error("Enter an index ticker.")
            st.stop()
        st.session_state["bm_loaded"] = True
        prices = load_closes(tuple(SECTOR_ETFS) + (idx,), years, freq)
        if prices.empty or idx not in prices.columns:
            st.error("Price download failed.")
            st.stop()

        bench = prices[idx].dropna()
        rows = []
        for e in SECTOR_ETFS:
            if e not in prices.columns:
                continue
            zz = rel_z(prices[e].dropna(), bench, z_win).dropna()
            rr = rsi(prices[e].dropna(), rsi_n).dropna()
            rows.append({"sector": SECTOR_NAMES[e], "etf": e,
                         f"z vs {idx}": round(float(zz.iloc[-1]), 2) if len(zz) else np.nan,
                         "RSI": round(float(rr.iloc[-1]), 1) if len(rr) else np.nan})
        summary = pd.DataFrame(rows).sort_values(f"z vs {idx}", ascending=False)
        st.dataframe(summary.style.background_gradient(cmap="RdYlBu_r",
                     subset=[f"z vs {idx}"], vmin=-3, vmax=3).format(precision=2),
                     use_container_width=True, hide_index=True)

        st.divider()
        for e in SECTOR_ETFS:
            if e not in prices.columns:
                continue
            st.markdown(f"##### {SECTOR_NAMES[e]} ({e}) vs {idx}")
            st.plotly_chart(
                indicator_figure(prices[e].dropna(), bench, label_for(e), idx, z_win, rsi_n, bar),
                use_container_width=True, key=f"bm_fig_{e}")

        st.divider()
        st.markdown("##### Backtest one of these pairs")
        pick = st.selectbox("Sector", SECTOR_ETFS, key="bm_bt_pick", format_func=label_for)
        if st.button("Run backtest", key="bm_bt_run", type="primary"):
            show_backtest(pick, idx, prices, z_win, stop, max_hold, key="bm_bt",
                          ann=ann, bar=bar)
    else:
        st.caption("Not loaded. Click to fetch all eleven sector ETFs and the index.")
