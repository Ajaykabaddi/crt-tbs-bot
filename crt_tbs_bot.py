"""
CRT + TBS signal bot v2 (notifications only, never places trades)

STRATEGY
  CRT (Candle Range Theory) on the higher timeframe (HTF):
    last closed HTF candle sweeps the previous candle's high/low and closes
    back INSIDE its range. Target = the opposite end of the previous range.
  TBS (Turtle Body Soup) on the lower timeframe (LTF):
    inside the next HTF candle, an LTF candle sweeps the last N candles'
    extreme and closes back inside -> entry at close, stop beyond the wick.

ADDED IN V2 (ideas borrowed from the day-trader-skill pack)
  - backtest      : replays the exact same rules on past candles (Time Machine idea)
  - Monte Carlo   : 10,000 reshuffles of the backtest trades -> honest odds
  - position size : qty from your max-loss-per-trade (Budget idea)
  - risk guard    : min R:R and max signals per day (Risk idea)
  - symbol filter : only alert on coins whose backtest expectancy is positive

USAGE
  pip install ccxt requests
  export TG_TOKEN="..."  TG_CHAT_ID="..."
  python crt_tbs_bot.py backtest     # see historical results first
  python crt_tbs_bot.py              # run the live scanner (loops forever)
  python crt_tbs_bot.py once         # one scan then exit (used by GitHub Actions)

No backtest gives "exact" results: it shows how the rules WOULD have done
on past data. Treat it as a reality check, not a promise. Not financial advice.
"""

import os
import sys
import json
import time
import random
from bisect import bisect_left
from datetime import datetime, timezone

import ccxt
import requests

# ---------------- CONFIG ----------------
EXCHANGE = os.getenv("EXCHANGE", "okx")   # ccxt id; okx/kraken work from GitHub servers
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
HTF = "4h"
LTF = "15m"
SWEEP_LOOKBACK = 20                # N candles for the turtle-soup sweep
SL_BUFFER_PCT = 0.05               # extra % beyond sweep wick for stop
MIN_RR = 1.5
POLL_SECONDS = 60

# risk / sizing
ACCOUNT_USD = 1000
MAX_LOSS_USD = 10                  # loss if stop is hit (1% of account)
MAX_LEVERAGE = 5
MAX_SIGNALS_PER_DAY = 3            # UTC day

# backtest
BACKTEST_DAYS = 180
HOLD_HTF_CANDLES = 2               # max trade life, in HTF candles after CRT
FEE_PCT = 0.06                     # per side (taker), charged on entry + exit
BACKTEST_FILTER = True             # live mode: skip coins with weak backtest
MIN_TRADES = 20                    # need at least this many trades to trust it
MONTE_CARLO_RUNS = 10000

STATE_FILE = "state.json"          # remembers sent alerts between GitHub runs

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
# ----------------------------------------

ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})


def load_state():
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        return set(d.get("seen", [])), d.get("signals_today", {})
    except Exception:
        return set(), {}


def save_state():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(STATE_FILE, "w") as f:
        json.dump({"seen": sorted(seen)[-200:],
                   "signals_today": {today: signals_today.get(today, 0)}},
                  f, indent=1)


seen, signals_today = load_state()


# ---------- helpers ----------
def notify(msg: str) -> None:
    print(msg, flush=True)
    if not (TG_TOKEN and TG_CHAT_ID):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=15,
        )
    except Exception as e:
        print("Telegram error:", e)


def closed_candles(symbol, tf, limit):
    return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)[:-1]


def fetch_history(symbol, tf, days):
    """Page through candles from `days` ago until now; drops the forming one."""
    step = ex.parse_timeframe(tf) * 1000
    since = ex.milliseconds() - days * 86400 * 1000
    rows = {}
    for _ in range(300):
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=1000)
        if not batch:
            break
        for c in batch:
            rows[c[0]] = c
        nxt = batch[-1][0] + step
        if nxt <= since or nxt > ex.milliseconds():
            break
        since = nxt
    out = [rows[k] for k in sorted(rows)]
    return out[:-1]


# ---------- strategy rules (shared by live + backtest) ----------
def crt_signal(ref, crt):
    ref_h, ref_l = ref[2], ref[3]
    c_h, c_l, c_c = crt[2], crt[3], crt[4]
    base = {"crt_ts": crt[0], "ref_h": ref_h, "ref_l": ref_l}
    if c_l < ref_l and ref_l < c_c <= ref_h:
        return {**base, "side": "LONG", "target": ref_h}
    if c_h > ref_h and ref_l <= c_c < ref_h:
        return {**base, "side": "SHORT", "target": ref_l}
    return None


def tbs_signal(side, target, prior, last):
    """prior = N candles before `last`. Returns trade plan or None."""
    h, l, cl = last[2], last[3], last[4]
    if side == "LONG":
        lowest = min(x[3] for x in prior)
        if not (l < lowest and cl > lowest):
            return None
        entry, sl, tp = cl, l * (1 - SL_BUFFER_PCT / 100), target
        if tp <= entry:
            return None
        rr = (tp - entry) / (entry - sl)
    else:
        highest = max(x[2] for x in prior)
        if not (h > highest and cl < highest):
            return None
        entry, sl, tp = cl, h * (1 + SL_BUFFER_PCT / 100), target
        if tp >= entry:
            return None
        rr = (entry - tp) / (sl - entry)
    if rr < MIN_RR:
        return None
    return {"ts": last[0], "entry": entry, "sl": sl, "tp": tp, "rr": rr}


# ---------- sizing ----------
def position_size(entry, sl):
    risk = abs(entry - sl)
    qty = MAX_LOSS_USD / risk
    notional = qty * entry
    cap = ACCOUNT_USD * MAX_LEVERAGE
    if notional > cap:
        qty, notional = cap / entry, cap
    return qty, notional, qty * risk


# ---------- backtest ----------
def simulate(side, t, ltf, k, deadline):
    """Walk candles after entry. Stop wins if both hit in one candle (conservative)."""
    entry, sl, tp = t["entry"], t["sl"], t["tp"]
    risk = abs(entry - sl)
    fee_r = 2 * FEE_PCT / 100 * entry / risk
    last_close = entry
    for c in ltf[k:]:
        if c[0] >= deadline:
            break
        h, l = c[2], c[3]
        if side == "LONG":
            if l <= sl:
                return -1 - fee_r
            if h >= tp:
                return t["rr"] - fee_r
        else:
            if h >= sl:
                return -1 - fee_r
            if l <= tp:
                return t["rr"] - fee_r
        last_close = c[4]
    pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
    return pnl / risk - fee_r


def backtest_symbol(symbol):
    htf = fetch_history(symbol, HTF, BACKTEST_DAYS)
    ltf = fetch_history(symbol, LTF, BACKTEST_DAYS)
    return run_backtest(htf, ltf)


def run_backtest(htf, ltf):
    htf_ms = ex.parse_timeframe(HTF) * 1000
    ts = [c[0] for c in ltf]
    trades = []
    for i in range(1, len(htf)):
        sig = crt_signal(htf[i - 1], htf[i])
        if not sig:
            continue
        start = htf[i][0] + htf_ms
        end = start + htf_ms
        j0, j1 = bisect_left(ts, start), bisect_left(ts, end)
        for j in range(max(j0, SWEEP_LOOKBACK), j1):
            t = tbs_signal(sig["side"], sig["target"],
                           ltf[j - SWEEP_LOOKBACK:j], ltf[j])
            if not t:
                continue
            deadline = start + HOLD_HTF_CANDLES * htf_ms
            r = simulate(sig["side"], t, ltf, j + 1, deadline)
            trades.append({"side": sig["side"], "r": r})
            break  # one trade per CRT
    return trades


def stats(rs):
    n = len(rs)
    if n == 0:
        return {"n": 0, "win": 0, "avg": 0, "total": 0, "pf": 0, "dd": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gl = abs(sum(losses))
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {"n": n, "win": len(wins) / n, "avg": sum(rs) / n,
            "total": sum(rs), "pf": (sum(wins) / gl) if gl else float("inf"),
            "dd": dd}


def monte_carlo(rs):
    n = len(rs)
    if n < 5:
        return None
    pos, dds = 0, []
    for _ in range(MONTE_CARLO_RUNS):
        eq = peak = dd = 0.0
        for r in random.choices(rs, k=n):
            eq += r
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        pos += eq > 0
        dds.append(dd)
    dds.sort()
    return pos / MONTE_CARLO_RUNS, dds[int(MONTE_CARLO_RUNS * 0.95)]


def report(symbol, trades):
    rs = [t["r"] for t in trades]
    s = stats(rs)
    if s["n"] == 0:
        return f"{symbol}: no CRT+TBS trades in {BACKTEST_DAYS} days"
    longs = sum(1 for t in trades if t["side"] == "LONG")
    mc = monte_carlo(rs)
    lines = [
        f"📊 {symbol} | {HTF} CRT + {LTF} TBS | {BACKTEST_DAYS}d",
        f"Trades: {s['n']} ({longs} long / {s['n'] - longs} short)",
        f"Win rate: {s['win'] * 100:.0f}%   Avg: {s['avg']:+.2f}R",
        f"Total: {s['total']:+.1f}R   Profit factor: {s['pf']:.2f}",
        f"Max drawdown: {s['dd']:.1f}R (fees {FEE_PCT}%/side included)",
    ]
    if mc:
        lines.append(f"Monte Carlo: profitable in {mc[0] * 100:.0f}% of "
                     f"{MONTE_CARLO_RUNS} reshuffles; bad-case drawdown "
                     f"{mc[1]:.1f}R")
    if s["n"] < MIN_TRADES:
        lines.append(f"⚠️ Only {s['n']} trades (<{MIN_TRADES}): too few to trust.")
    elif s["avg"] <= 0:
        lines.append("⚠️ Negative expectancy on this coin.")
    else:
        lines.append("✅ Positive expectancy in this sample.")
    return "\n".join(lines)


def run_backtests():
    good = []
    for symbol in SYMBOLS:
        try:
            trades = backtest_symbol(symbol)
        except Exception as e:
            notify(f"{symbol}: backtest failed ({e})")
            continue
        notify(report(symbol, trades))
        s = stats([t["r"] for t in trades])
        if s["n"] >= MIN_TRADES and s["avg"] > 0:
            good.append(symbol)
    return good


# ---------- live scan ----------
def scan(symbols):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for symbol in symbols:
        try:
            c = closed_candles(symbol, HTF, 5)
            if len(c) < 2:
                continue
            crt = crt_signal(c[-2], c[-1])
            if not crt:
                continue
            htf_ms = ex.parse_timeframe(HTF) * 1000
            ltf = closed_candles(symbol, LTF, SWEEP_LOOKBACK + 5)
            if len(ltf) < SWEEP_LOOKBACK + 1:
                continue
            last = ltf[-1]
            if last[0] < crt["crt_ts"] + htf_ms:
                continue
            t = tbs_signal(crt["side"], crt["target"],
                           ltf[-1 - SWEEP_LOOKBACK:-1], last)
            if not t:
                continue
            key = f"{symbol}|{crt['side']}|{t['ts']}"
            if key in seen:
                continue
            seen.add(key)
            if signals_today.get(today, 0) >= MAX_SIGNALS_PER_DAY:
                print(f"[{symbol}] signal skipped: daily cap reached")
                continue
            signals_today[today] = signals_today.get(today, 0) + 1
            qty, notional, loss = position_size(t["entry"], t["sl"])
            icon = "🟢" if crt["side"] == "LONG" else "🔴"
            notify(
                f"{icon} CRT+TBS {crt['side']} | {symbol}\n"
                f"HTF {HTF} CRT + LTF {LTF} turtle-soup trigger\n"
                f"Entry:  {t['entry']:.4f}\n"
                f"Stop:   {t['sl']:.4f}\n"
                f"Target: {t['tp']:.4f}\n"
                f"R:R ≈ {t['rr']:.2f}\n"
                f"Size: {qty:.4f} ({notional:.0f} USD), loss at stop ≈ {loss:.2f} USD\n"
                f"CRT range: {crt['ref_l']:.4f} – {crt['ref_h']:.4f}"
            )
        except Exception as e:
            print(f"[{symbol}] error: {e}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtests()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        scan(SYMBOLS)      # one pass, for GitHub Actions
        save_state()
        return
    symbols = SYMBOLS
    if BACKTEST_FILTER:
        notify("Running startup backtest filter...")
        symbols = run_backtests()
        if not symbols:
            notify("No coin passed the backtest filter. Edit SYMBOLS/settings "
                   "or set BACKTEST_FILTER = False.")
            return
    notify(f"✅ CRT+TBS bot live ({EXCHANGE}, {HTF}/{LTF}): {', '.join(symbols)}")
    while True:
        scan(symbols)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
