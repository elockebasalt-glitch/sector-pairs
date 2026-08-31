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
               "z_default": 60,  "z_max": 504, "z_step": 5,
               "hold_default": 126, "hold_max": 504, "hold_step": 21},
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
            "median_%": round(tdf["return_%"].median(), 2),
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
def adjective(bar: str) -> str:
    """'day' -> 'daily', 'week' -> 'weekly'. Naive f"{bar}ly" gives 'dayly'."""
    return "daily" if bar == "day" else f"{bar}ly"


def view_tail(s: pd.Series, years: float | None) -> pd.Series:
    """Trim a series to the last `years` for display only.

    Indicators are always computed on the full history first — slicing after
    the fact keeps the warmup intact (a 252-bar z-score is still valid at the
    left edge) and lets each y-axis rescale to what is actually on screen,
    which is what makes MACD legible.
    """
    if s is None or len(s) == 0 or not years:
        return s
    return s[s.index >= s.index.max() - pd.Timedelta(days=int(365.25 * years))]


def indicator_figure(px: pd.Series, bench: pd.Series, label: str, bench_label: str,
                     z_win: int, rsi_n: int, bar: str = "week",
                     rsi_lo: int = 30, rsi_hi: int = 70,
                     view_years: float | None = 3.0,
                     trades: pd.DataFrame | None = None,
                     entry_z: float | None = None) -> go.Figure:
    """Pass `trades` to mark backtest entries and exits on the z pane, so the
    equity curve and the z path can be read against each other."""
    # computed on everything, then trimmed for the eye
    r = rsi(px, rsi_n)
    ml, sg, hist = macd(px)
    z = rel_z(px, bench, z_win)
    px, r, ml, sg, hist, z = (view_tail(x, view_years)
                              for x in (px, r, ml, sg, hist, z))

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.32, 0.19, 0.21, 0.28],
        subplot_titles=(f"{label} - {adjective(bar)} close"
                        + (f"   (last {view_years:g}y shown)" if view_years else "  (full history)"),
                        f"RSI({rsi_n})   ·   overbought {rsi_hi} / oversold {rsi_lo}",
                        f"MACD 12/26/9 ({adjective(bar)})",
                        f"Rolling {z_win}-{bar} z-score vs {bench_label}"),
    )
    fig.add_trace(go.Scatter(x=px.index, y=px, mode="lines",
                             line=dict(color=LINE, width=1.5), name="Close"), row=1, col=1)

    # RSI. The trace has to exist before add_hrect, which skips empty subplots.
    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines",
                             line=dict(color=CHEAP, width=1.3), name="RSI"), row=2, col=1)
    fig.add_hrect(y0=rsi_lo, y1=rsi_hi, fillcolor="rgba(122,136,153,0.09)",
                  line_width=0, layer="below", row=2, col=1)
    fig.add_hline(y=rsi_hi, line=dict(color=RICH, width=1, dash="dash"),
                  annotation_text=f"{rsi_hi} overbought", annotation_position="top left",
                  annotation_font=dict(size=9, color=RICH), row=2, col=1)
    fig.add_hline(y=rsi_lo, line=dict(color=CHEAP, width=1, dash="dash"),
                  annotation_text=f"{rsi_lo} oversold", annotation_position="bottom left",
                  annotation_font=dict(size=9, color=CHEAP), row=2, col=1)
    fig.add_hline(y=50, line=dict(color=GRID, width=1), row=2, col=1)

    fig.add_trace(go.Bar(x=hist.index, y=hist, name="hist",
                         marker_color=np.where(hist >= 0, UP, DOWN),
                         marker_line_width=0, opacity=0.5), row=3, col=1)
    fig.add_trace(go.Scatter(x=ml.index, y=ml, mode="lines",
                             line=dict(color=LINE, width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=sg.index, y=sg, mode="lines",
                             line=dict(color=RICH, width=1.1), name="signal"), row=3, col=1)

    if not z.dropna().empty:
        fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines",
                                 line=dict(color=TEXT, width=1.6), fill="tozeroy",
                                 fillcolor="rgba(200,208,218,0.10)", name="z"), row=4, col=1)
        for lvl, c in ((2, RICH), (-2, CHEAP)):
            fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dash"), row=4, col=1)
        fig.add_hline(y=0, line=dict(color=GRID, width=1), row=4, col=1)

        if entry_z is not None:
            fig.add_hline(y=entry_z, line=dict(color=UP, width=1.2, dash="dot"),
                          annotation_text=f"entry {entry_z:+.2f}",
                          annotation_position="top right",
                          annotation_font=dict(size=9, color=UP), row=4, col=1)

        if trades is not None and not trades.empty:
            zc = z.dropna()
            ent = zc.reindex(pd.DatetimeIndex(trades["entry"])).dropna()
            if len(ent):
                fig.add_trace(go.Scatter(
                    x=ent.index, y=ent.values, mode="markers", name="entry",
                    marker=dict(color=TEXT, size=8, symbol="circle-open",
                                line=dict(width=1.6)),
                    hovertemplate="entry  z %{y:+.2f}<br>%{x|%Y-%m-%d}<extra></extra>"),
                    row=4, col=1)
            for reason, colour, sym in (("stop", DOWN, "x"), ("target", UP, "triangle-up"),
                                        ("max hold", MUTED, "square")):
                sub_t = trades[trades["exit_reason"] == reason]
                if sub_t.empty:
                    continue
                pts = zc.reindex(pd.DatetimeIndex(sub_t["exit"])).dropna()
                if not len(pts):
                    continue
                fig.add_trace(go.Scatter(
                    x=pts.index, y=pts.values, mode="markers", name=reason,
                    marker=dict(color=colour, size=8, symbol=sym),
                    hovertemplate=reason + "  z %{y:+.2f}<br>%{x|%Y-%m-%d}<extra></extra>"),
                    row=4, col=1)

    fig.update_layout(height=880, template="plotly_dark", paper_bgcolor=INK,
                      plot_bgcolor=INK, showlegend=trades is not None, bargap=0,
                      legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
                      font=dict(family=MONO, size=11, color=TEXT),
                      margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(range=[0, 100], tickvals=[rsi_lo, 50, rsi_hi], row=2, col=1)
    for a_ in fig.layout.annotations:
        if a_.text and "overbought" not in a_.text and "oversold" not in a_.text:
            a_.font.update(size=11, color=MUTED)
    return fig


def compact_figure(px: pd.Series, bench: pd.Series, z_win: int, rsi_n: int,
                   bar: str, bench_label: str,
                   rsi_lo: int = 30, rsi_hi: int = 70,
                   view_years: float | None = 3.0) -> go.Figure:
    """One name, one short row: price, RSI, MACD, z. Built for scanning many
    names down a page rather than studying one."""
    r = rsi(px, rsi_n)
    ml, sg, hist = macd(px)
    z = rel_z(px, bench, z_win)
    px, r, ml, sg, hist, z = (view_tail(x, view_years)
                              for x in (px, r, ml, sg, hist, z))

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.036,
                        subplot_titles=("price", f"RSI {rsi_n}  ({rsi_lo}/{rsi_hi})",
                                        "MACD", f"z vs {bench_label}"))
    fig.add_trace(go.Scatter(x=px.index, y=px, mode="lines",
                             line=dict(color=LINE, width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines",
                             line=dict(color=CHEAP, width=1.1)), row=1, col=2)
    fig.add_hrect(y0=rsi_lo, y1=rsi_hi, fillcolor="rgba(122,136,153,0.09)",
                  line_width=0, layer="below", row=1, col=2)
    fig.add_hline(y=rsi_hi, line=dict(color=RICH, width=1, dash="dash"), row=1, col=2)
    fig.add_hline(y=rsi_lo, line=dict(color=CHEAP, width=1, dash="dash"), row=1, col=2)
    fig.add_hline(y=50, line=dict(color=GRID, width=1), row=1, col=2)

    fig.add_trace(go.Bar(x=hist.index, y=hist, marker_line_width=0, opacity=.5,
                         marker_color=np.where(hist >= 0, UP, DOWN)), row=1, col=3)
    fig.add_trace(go.Scatter(x=ml.index, y=ml, mode="lines",
                             line=dict(color=LINE, width=1)), row=1, col=3)
    fig.add_trace(go.Scatter(x=sg.index, y=sg, mode="lines",
                             line=dict(color=RICH, width=.9)), row=1, col=3)

    if not z.dropna().empty:
        fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines",
                                 line=dict(color=TEXT, width=1.3), fill="tozeroy",
                                 fillcolor="rgba(200,208,218,0.10)"), row=1, col=4)
        for lvl, c in ((2, RICH), (-2, CHEAP)):
            fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dash"), row=1, col=4)
        fig.add_hline(y=0, line=dict(color=GRID, width=1), row=1, col=4)

    fig.update_layout(height=185, template="plotly_dark", paper_bgcolor=INK,
                      plot_bgcolor=INK, showlegend=False, bargap=0,
                      font=dict(family=MONO, size=9, color=TEXT),
                      margin=dict(l=4, r=4, t=26, b=14))
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=8))
    fig.update_yaxes(range=[0, 100], tickvals=[rsi_lo, 50, rsi_hi], row=1, col=2)
    for a_ in fig.layout.annotations:
        a_.font.update(size=9, color=MUTED)
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
@st.cache_data(ttl=1800, show_spinner="Scanning every sector…")
def scan_extremes(years: int, freq: str, z_win: int, rsi_n: int,
                  thresh: float, index_sym: str):
    """Every name whose z against its own benchmark has passed +/- `thresh`.

    z is vectorised so it runs across all 501 names cheaply; RSI and MACD are
    only computed for the handful that survive the filter.
    """
    def _row(sym, name, group, series, bench, zv):
        r = rsi(series, rsi_n).dropna()
        _, _, h = macd(series)
        h = h.dropna()
        return {"ticker": sym, "name": name, "group": group,
                "z": round(zv, 2), "side": "LONG" if zv < 0 else "SHORT",
                "last": round(float(series.iloc[-1]), 2),
                "RSI": round(float(r.iloc[-1]), 1) if len(r) else np.nan,
                "MACD hist": round(float(h.iloc[-1]), 3) if len(h) else np.nan,
                "vs": bench}

    con, missing = [], []
    for etf in SECTOR_ETFS:
        ticks = tuple(t for t, _ in CONSTITUENTS[etf]) + (etf,)
        px = load_closes(ticks, years, freq)
        if px.empty or etf not in px.columns:
            missing.append(etf)
            continue
        bench = px[etf].dropna()
        for t, nm in CONSTITUENTS[etf]:
            if t not in px.columns:
                continue
            zs = rel_z(px[t].dropna(), bench, z_win).dropna()
            if zs.empty:
                continue
            zv = float(zs.iloc[-1])
            if abs(zv) < thresh:
                continue
            con.append(_row(t, nm, SECTOR_NAMES[etf], px[t].dropna(), etf, zv))

    sec = []
    base = load_closes(tuple(SECTOR_ETFS) + tuple(BROAD), years, freq)
    if not base.empty and index_sym in base.columns:
        ib = base[index_sym].dropna()
        for etf in SECTOR_ETFS:
            if etf not in base.columns:
                continue
            zs = rel_z(base[etf].dropna(), ib, z_win).dropna()
            if zs.empty:
                continue
            zv = float(zs.iloc[-1])
            if abs(zv) < thresh:
                continue
            sec.append(_row(etf, SECTOR_NAMES[etf], "Sector ETF",
                            base[etf].dropna(), index_sym, zv))

    key = lambda d: d.reindex(d["z"].abs().sort_values(ascending=False).index) if len(d) else d
    return key(pd.DataFrame(con)), key(pd.DataFrame(sec)), missing


def shade(v, lo: float, hi: float, neg=(76, 201, 240), pos=(240, 162, 2)) -> str:
    """CSS background for one cell. Avoids Styler.background_gradient, which
    drags in matplotlib for nothing."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    span = max(abs(lo), abs(hi)) or 1.0
    t = max(-1.0, min(1.0, x / span))
    r, g, b = pos if t >= 0 else neg
    return f"background-color: rgba({r},{g},{b},{0.08 + 0.42 * abs(t):.2f})"


def shade_col(df: pd.DataFrame, col: str, lo: float, hi: float, **kw):
    return df.style.apply(lambda c: [shade(v, lo, hi, **kw) for v in c], subset=[col])


def stat(s: dict, key: str, fmt: str = "{:+.2f}%", dash: str = "—") -> str:
    """Read a stat safely. A missing or non-numeric value renders as a dash
    rather than taking down the whole page."""
    v = s.get(key)
    if v is None:
        return dash
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


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
            "median %": s.get("median_%"),
            "total %": s.get("total_return_%"),
            "max DD %": s.get("max_drawdown_%"),
            "Sharpe": s.get("sharpe_in_trade"),
            f"avg {bar}s": s.get("avg_bars"),
            "stops": s.get("stops_hit"),
        })
    return pd.DataFrame(rows)


def scan_panel(key: str, names: pd.DataFrame, bench_ticker: str,
               prices: pd.DataFrame, z_win: int, rsi_n: int, bar: str,
               rsi_lo: int = 30, rsi_hi: int = 70,
               view_years: float | None = 3.0,
               max_charts: int | None = None) -> None:
    """Every constituent stacked down one scrollable pane, one compact row each."""
    have = [t for t in names["symbol"] if t in prices.columns]
    if not have:
        st.warning("No price history returned for these names.")
        return
    bench = prices[bench_ticker].dropna()

    rows = []
    for t in have:
        px = prices[t].dropna()
        z = rel_z(px, bench, z_win).dropna()
        r = rsi(px, rsi_n).dropna()
        _, _, h = macd(px)
        rows.append({
            "ticker": t, "name": TICKER_TO_NAME.get(t, ""),
            "last": round(float(px.iloc[-1]), 2),
            f"1{bar[0]} %": round(float(px.iloc[-1] / px.iloc[-2] - 1) * 100, 2) if len(px) > 1 else np.nan,
            "RSI": round(float(r.iloc[-1]), 1) if len(r) else np.nan,
            f"z vs {bench_ticker}": round(float(z.iloc[-1]), 2) if len(z) else np.nan,
            "MACD hist": round(float(h.dropna().iloc[-1]), 3) if len(h.dropna()) else np.nan,
        })
    df = pd.DataFrame(rows)
    zcol = f"z vs {bench_ticker}"

    n_show = len(have) if max_charts is None else min(max_charts, len(have))
    c1, c2 = st.columns([2, 5])
    order = c1.selectbox("Sort by", ["|z| widest first", "z ascending (cheapest first)",
                                     "z descending (richest first)", "alphabetical"],
                         key=f"{key}_sort")
    c2.caption(f"{len(have)} of {len(names)} names have usable history · "
               f"drawing {n_show}. Change the count with **Charts per sector** in the "
               f"sidebar. Scroll down through the charts.")

    if order.startswith("|z|"):
        df = df.reindex(df[zcol].abs().sort_values(ascending=False).index)
    elif order.startswith("z ascending"):
        df = df.sort_values(zcol, ascending=True)
    elif order.startswith("z descending"):
        df = df.sort_values(zcol, ascending=False)
    else:
        df = df.sort_values("ticker")

    st.dataframe(shade_col(df.reset_index(drop=True), zcol, -3, 3).format(precision=2, na_rep="—"),
                 use_container_width=True, hide_index=True, height=300)
    st.divider()

    for t in df["ticker"].head(n_show):
        px = prices[t].dropna()
        zz = rel_z(px, bench, z_win).dropna()
        zn = float(zz.iloc[-1]) if len(zz) else np.nan
        col = RICH if zn > 2 else CHEAP if zn < -2 else TEXT
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:.8rem;margin:.1rem 0 -.4rem'>"
            f"<b style='font-size:1.05rem;color:{LINE}'>{t}</b>"
            f"<span style='font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;"
            f"color:{MUTED}'>{TICKER_TO_NAME.get(t, '')}</span>"
            f"<span style='margin-left:auto;font-size:.85rem;color:{col}'>"
            f"z {zn:+.2f}</span></div>", unsafe_allow_html=True)
        st.plotly_chart(compact_figure(px, bench, z_win, rsi_n, bar, bench_ticker,
                                       rsi_lo, rsi_hi, view_years),
                        use_container_width=True, key=f"{key}_cf_{t}")


def show_backtest(stock_t: str, bench_t: str, prices: pd.DataFrame,
                  z_win: int, stop: float, max_hold: int, key: str,
                  ann: int = 52, bar: str = "week",
                  rsi_n: int = 14, rsi_lo: int = 30, rsi_hi: int = 70) -> None:
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
        a[0].metric("Trades", stat(s, "trades", "{:.0f}"))
        a[1].metric("Win rate", stat(s, "win_rate_%", "{:.1f}%"))
        a[2].metric("Avg trade", stat(s, "avg_trade_%"))
        a[3].metric("Sharpe", stat(s, "sharpe_in_trade", "{:.2f}"))
        a[4].metric("Max DD", stat(s, "max_drawdown_%", "{:.2f}%"))
        b = st.columns(5)
        b[0].metric("Median", stat(s, "median_%"))
        b[1].metric("Best / worst",
                    f"{stat(s, 'best_%', '{:+.1f}')} / {stat(s, 'worst_%', '{:+.1f}')}%")
        b[2].metric("Avg hold", stat(s, "avg_bars", "{:.0f} " + f"{bar}s"))
        b[3].metric("Stops hit", f"{stat(s, 'stops_hit', '{:.0f}')} / {stat(s, 'trades', '{:.0f}')}")
        b[4].metric("Total", stat(s, "total_return_%", "{:+.1f}%"))

    st.plotly_chart(
        indicator_figure(px, bench, label_for(stock_t), label_for(bench_t),
                         z_win, rsi_n, bar, rsi_lo, rsi_hi,
                         view_years=None,
                         trades=None if res.trades.empty else res.trades,
                         entry_z=z_now),
        use_container_width=True, key=f"{key}_pair")
    st.caption(
        f"Same {z_win}-{bar} z-score as the sector tabs, on the full pull rather than the "
        f"sidebar chart window, so every trade is visible. The dotted line is today's z "
        f"({z_now:+.2f}) — the level entries are taken at. Hollow circles are entries; "
        f"triangles took profit at z=0, crosses stopped out, squares hit the hold limit."
    )

    if not res.trades.empty:
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
        shade_col(grid, "avg trade %", -20, 20,
                  neg=(242, 85, 90), pos=(61, 214, 140)).format(precision=2, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(
        f"Exit on reversion through 0, a {stop:.0%} adverse spread move, or {max_hold} {bar}s. "
        f"The stop is evaluated on {adjective(bar)} closes, so one violent {bar} overshoots it. No costs, "
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
    freq = st.radio("Bar frequency", list(FREQ), index=0, horizontal=True,
                    help="The download is daily either way. Weekly resamples to "
                         "Friday closes; daily keeps every bar.")
    F = FREQ[freq]
    bar, ann = F["bar"], F["ann"]

    years = st.slider("History pulled (years)", 3, 20, 12,
                      help="How much data is downloaded and used to compute "
                           "indicators. Longer gives the z-score a deeper baseline.")
    view_choice = st.select_slider("Chart window", options=[1, 2, 3, 5, 7, 10, "All"],
                                   value=3,
                                   help="How much of that history is drawn. Indicators "
                                        "still use the full pull; this only zooms the view "
                                        "so the axes rescale and MACD stays legible.")
    view_years = None if view_choice == "All" else float(view_choice)
    z_win = st.slider(f"Z window ({bar}s)", F["z_step"] * 2, F["z_max"],
                      F["z_default"], step=F["z_step"])
    rsi_n = st.slider(f"RSI length ({bar}s)", 5, 30, 14)
    rc1, rc2 = st.columns(2)
    rsi_lo = rc1.number_input("Oversold", 5, 45, 30, step=5)
    rsi_hi = rc2.number_input("Overbought", 55, 95, 70, step=5)
    st.divider()
    st.markdown("**Backtest rules**")
    stop = st.slider("Adverse-spread stop", 0.02, 0.30, 0.10, step=0.01, format="%.2f")
    max_hold = st.slider(f"Max hold ({bar}s)", F["hold_step"], F["hold_max"],
                         F["hold_default"], step=F["hold_step"])
    z_flag = st.slider("Summary |z| threshold", 1.0, 3.0, 2.0, step=0.1,
                       help="What counts as stretched on the Summary tab.")
    summary_idx = st.selectbox("Summary index", BROAD, key="sum_idx",
                               help="Benchmark for the sector-vs-index half of the Summary tab.")
    _yrs = z_win / (252 if bar == "day" else 52)
    st.caption(f"{z_win} {bar}s ≈ "
               + (f"{_yrs * 12:.0f} months" if _yrs < 1 else f"{_yrs:.1f} years")
               + " of lookback.")
    st.divider()
    st.markdown("**Rendering**")
    cap_choice = st.select_slider(
        "Charts per sector", options=[5, 10, 15, 25, 50, "All"], value="All",
        help="Streamlit executes every tab on each rerun, not just the visible one. "
             "With all sectors loaded and this on All you are drawing ~400 charts per "
             "interaction. Dial it down if the app feels sluggish.")
    max_charts = None if cap_choice == "All" else int(cap_choice)

    st.divider()
    loaded_now = sum(1 for e in SECTOR_ETFS if st.session_state.get(f"load_{e}"))
    b1, b2 = st.columns([3, 2])
    fetch_all = b1.button("⟳ Load / refresh all", use_container_width=True,
                          help="Clears the price cache and re-pulls every sector plus the "
                               "broad indexes, then opens all tabs. Takes a few minutes on "
                               "first run.")
    if b2.button("Unload", use_container_width=True,
                 help="Closes every tab so reruns stay fast. Cached prices are kept."):
        for e in SECTOR_ETFS:
            st.session_state.pop(f"load_{e}", None)
        st.session_state.pop("bm_loaded", None)
        st.rerun()
    st.caption(f"{loaded_now}/{len(SECTOR_ETFS)} sector tabs open"
               + ("  ·  reruns will be slow" if loaded_now > 3 and max_charts is None else ""))

    st.divider()
    st.caption("Constituents are a static snapshot in constituents.py — no external "
               "lookup, so nothing to fail. Edit that file to change the universe.")

if fetch_all:
    st.cache_data.clear()
    jobs = [(e, tuple(t for t, _ in CONSTITUENTS[e]) + (e,)) for e in SECTOR_ETFS]
    jobs.append(("broad indexes", tuple(SECTOR_ETFS) + tuple(BROAD)))
    bar_ui = st.progress(0.0, text="Starting…")
    failed = []
    for i, (nm, ticks) in enumerate(jobs, 1):
        bar_ui.progress(i / len(jobs),
                        text=f"[{i}/{len(jobs)}] {SECTOR_NAMES.get(nm, nm)} — {len(ticks)} tickers")
        try:
            got = load_closes(ticks, years, freq)
            if got.empty:
                failed.append(nm)
        except Exception:
            failed.append(nm)
        if nm in SECTOR_NAMES:
            st.session_state[f"load_{nm}"] = True
    st.session_state["bm_loaded"] = True
    bar_ui.empty()
    if failed:
        st.warning("Download returned nothing for: " + ", ".join(failed))
    else:
        st.success("All sectors cached. Every tab is open — set **Charts per sector** "
                   "lower in the sidebar if reruns feel slow.")

tabs = st.tabs(["◆ Summary"] + [f"{e} {SECTOR_NAMES[e]}" for e in SECTOR_ETFS]
               + ["◆ Backtest", "◆ Sectors vs market"])
sector_tabs = tabs[1:1 + len(SECTOR_ETFS)]

with tabs[0]:
    st.markdown(f"#### Everything past ±{z_flag:.1f}σ")
    st.caption(f"Constituents measured against their own sector ETF, sector ETFs against "
               f"{summary_idx}. Rolling {z_win}-{bar} window. LONG means stretched cheap "
               f"(buy the name, sell the benchmark); SHORT is the reverse.")
    if st.session_state.get("sum_done") or st.button("Scan all 501 names", key="sum_btn",
                                                     type="primary"):
        st.session_state["sum_done"] = True
        con, sec, miss = scan_extremes(years, freq, z_win, rsi_n, z_flag, summary_idx)

        st.markdown(f"##### Sector ETFs vs {summary_idx}")
        if sec.empty:
            st.info(f"No sector is past ±{z_flag:.1f} against {summary_idx} right now.")
        else:
            st.dataframe(shade_col(sec.reset_index(drop=True), "z", -3, 3)
                         .format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True)

        st.markdown("##### Constituents vs their sector")
        if con.empty:
            st.info(f"Nothing past ±{z_flag:.1f} across the 501 names.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Stretched names", len(con))
            c2.metric("Cheap (LONG)", int((con["side"] == "LONG").sum()))
            c3.metric("Rich (SHORT)", int((con["side"] == "SHORT").sum()))
            pick_sec = st.multiselect("Filter by sector", sorted(con["group"].unique()),
                                      key="sum_filt")
            view = con[con["group"].isin(pick_sec)] if pick_sec else con
            st.dataframe(shade_col(view.reset_index(drop=True), "z", -3, 3)
                         .format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True, height=460)
            st.caption("Sorted by |z|. Take a ticker to the Backtest tab to see how "
                       "entries at that level have historically resolved.")
        if miss:
            st.warning("No price data returned for: " + ", ".join(miss))
        if st.button("Rescan", key="sum_rescan"):
            scan_extremes.clear()
            st.rerun()
    else:
        st.caption("Not scanned. Uses cached prices, so run **⟳ Load / refresh all** "
                   "first and this is quick.")

for etf, tab in zip(SECTOR_ETFS, sector_tabs):
    with tab:
        st.markdown(f"#### {SECTOR_NAMES[etf]} — constituents vs {etf}")
        flag = f"load_{etf}"
        if not st.session_state.get(flag) and not st.button(f"Load {SECTOR_NAMES[etf]}",
                                                            key=f"btn_{etf}"):
            st.caption("Not loaded. Click to fetch this sector, or use "
                       "**⟳ Load / refresh all** in the sidebar to do every sector at once.")
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
        scan_panel(f"sec_{etf}", names, etf, prices, z_win, rsi_n, bar,
                   rsi_lo, rsi_hi, view_years, max_charts)

with tabs[len(SECTOR_ETFS) + 1]:
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
                          z_win, stop, max_hold, key="bt", ann=ann, bar=bar,
                          rsi_n=rsi_n, rsi_lo=rsi_lo, rsi_hi=rsi_hi)

with tabs[len(SECTOR_ETFS) + 2]:
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
        # request the same tuple the refresh button caches, otherwise the cache
        # key differs and this refetches despite the data already being warm
        prices = (load_closes(tuple(SECTOR_ETFS) + tuple(BROAD), years, freq)
                  if idx in BROAD else
                  load_closes(tuple(SECTOR_ETFS) + (idx,), years, freq))
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
        st.dataframe(shade_col(summary, f"z vs {idx}", -3, 3).format(precision=2, na_rep="—"),
                     use_container_width=True, hide_index=True)

        st.divider()
        for e in SECTOR_ETFS:
            if e not in prices.columns:
                continue
            st.markdown(f"##### {SECTOR_NAMES[e]} ({e}) vs {idx}")
            st.plotly_chart(
                indicator_figure(prices[e].dropna(), bench, label_for(e), idx,
                                 z_win, rsi_n, bar, rsi_lo, rsi_hi, view_years),
                use_container_width=True, key=f"bm_fig_{e}")

        st.divider()
        st.markdown("##### Backtest one of these pairs")
        pick = st.selectbox("Sector", SECTOR_ETFS, key="bm_bt_pick", format_func=label_for)
        if st.button("Run backtest", key="bm_bt_run", type="primary"):
            show_backtest(pick, idx, prices, z_win, stop, max_hold, key="bm_bt",
                          ann=ann, bar=bar, rsi_n=rsi_n, rsi_lo=rsi_lo, rsi_hi=rsi_hi)
    else:
        st.caption("Not loaded. Click to fetch all eleven sector ETFs and the index.")
