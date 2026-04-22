"""Consecutive pulls of the Binance futures OI endpoint.

Goal: confirm that a row stamped at T = HH:MM represents the exchange's
snapshot at exactly HH:MM (i.e. the same thing we'd get pulling at HH:MM),
and characterise how soon after the 5m boundary the new row appears.

We pull 7 times at 60s intervals (~6 minutes total), so we cross at least one
5m boundary, probably two. For each pull we record:
  - wall clock pull_time (UTC)
  - the latest few rows (stamp + OI + OI-value)

Then we ask:
  1. Does the latest stamp advance in 5m increments as expected?
  2. What's the lag between stamp and when the row is first visible?
  3. For a given stamp, do OI / OI-value mutate across later pulls, or stay fixed?
"""
import json
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone


URL = "https://fapi.binance.com/futures/data/openInterestHist"
SYMBOL = "BTCUSDT"
PERIOD = "5m"
LIMIT = 4
N_PULLS = 7
INTERVAL_S = 60


def pull():
    q = f"?symbol={SYMBOL}&period={PERIOD}&limit={LIMIT}"
    req = urllib.request.Request(URL + q, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fmt(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def main():
    pulls = []
    print(f"# {N_PULLS} pulls, {INTERVAL_S}s apart, symbol={SYMBOL} period={PERIOD}")
    print(f"# start: {datetime.now(timezone.utc).isoformat()}")
    print()
    for i in range(N_PULLS):
        pt = datetime.now(timezone.utc)
        try:
            rows = pull()
        except Exception as e:
            print(f"pull {i+1}  ERR {e}")
            rows = []
        pulls.append((pt, rows))
        print(f"=== pull {i+1}  at {pt.isoformat()} ===")
        for r in rows:
            ts = r["timestamp"]
            lag = (pt - datetime.fromtimestamp(ts / 1000, tz=timezone.utc)).total_seconds()
            print(f"  {fmt(ts)}  lag={lag:7.1f}s  OI={r['sumOpenInterest']}  OIV={r['sumOpenInterestValue']}")
        print(flush=True)
        if i < N_PULLS - 1:
            time.sleep(INTERVAL_S)

    # cross-pull analysis
    print("# ================ ANALYSIS ================")
    by_stamp = defaultdict(list)
    for pull_idx, (pt, rows) in enumerate(pulls):
        for r in rows:
            by_stamp[r["timestamp"]].append((pull_idx, pt, r))

    for stamp in sorted(by_stamp):
        obs = by_stamp[stamp]
        ois = {o[2]["sumOpenInterest"] for o in obs}
        oivs = {o[2]["sumOpenInterestValue"] for o in obs}
        first_pull, first_pt, _ = obs[0]
        first_lag = (first_pt - datetime.fromtimestamp(stamp / 1000, tz=timezone.utc)).total_seconds()
        verdict_oi = "STABLE" if len(ois) == 1 else f"CHANGED ({len(ois)} values)"
        verdict_oiv = "STABLE" if len(oivs) == 1 else f"CHANGED ({len(oivs)} values)"
        print(
            f"stamp {fmt(stamp)}  seen in {len(obs)} pulls  "
            f"first_seen_lag={first_lag:.1f}s  OI={verdict_oi}  OIV={verdict_oiv}"
        )
        if len(ois) > 1 or len(oivs) > 1:
            for pi, pt, r in obs:
                print(
                    f"  pull {pi+1} @ {pt.isoformat()}: "
                    f"OI={r['sumOpenInterest']} OIV={r['sumOpenInterestValue']}"
                )

    print()
    print(f"# end: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
