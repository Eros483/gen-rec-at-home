"""Verbalize a user's event history into a ranking prompt (spec 4.2).

Knobs: n_events (oldest dropped first), verbosity full/compact, low-signal filter,
optional token-budget truncation (drop oldest until it fits).
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")


def render_event(e, item, verbosity):
    title = item["title"] if verbosity == "full" else YEAR_RE.sub("", item["title"])
    line = f'- Watched "{title}"'
    if verbosity == "full":
        line += f" [{', '.join(item['genres'])}]"
    return f"{line} — rated {e['r']}"


def verbalize(events, catalog, n_events=10, verbosity="full", drop_low_signal=False,
              fit_tokens=None, max_len=1024):
    """events: chronological [{"i","r","t"}]; catalog: item_id -> {title, genres}; returns prompt text."""
    first_month = datetime.fromtimestamp(events[0]["t"], tz=timezone.utc).strftime("%Y-%m") if events else "unknown"
    tenure = f"tenure {len(events)} ratings since {first_month}."

    pool = [e for e in events if not drop_low_signal or e["r"] > 2]
    evs = pool[-n_events:]
    if not evs:  # every event was low-signal and filtered out
        hist = "(no usable history)"
    else:
        hist = "\n".join(render_event(e, catalog[e["i"]], verbosity) for e in evs)
    prompt = f"User profile: {tenure}\nContext: homepage.\n\nViewing history (most recent last):\n{hist}\n\nTask: rank the catalog for this user."

    if fit_tokens is not None:
        while len(evs) > 1 and fit_tokens(prompt) > max_len:
            evs = evs[1:]  # drop oldest first
            hist = "\n".join(render_event(e, catalog[e["i"]], verbosity) for e in evs)
            prompt = f"User profile: {tenure}\nContext: homepage.\n\nViewing history (most recent last):\n{hist}\n\nTask: rank the catalog for this user."
    return prompt


def load_data(data_dir):
    catalog = {c["item_id"]: c for c in map(json.loads, open(Path(data_dir) / "catalog.jsonl"))}
    users = {u["user_id"]: u for u in map(json.loads, open(Path(data_dir) / "events.jsonl"))}
    return catalog, users


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--verbosity", choices=["full", "compact"], default="full")
    ap.add_argument("--drop-low-signal", action="store_true")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    catalog, users = load_data(args.data_dir)
    u = users[args.user if args.user is not None else next(iter(users))]
    prompt = verbalize(u["events"], catalog, args.n, args.verbosity, args.drop_low_signal)
    print(prompt)

    # self-checks
    full = verbalize(u["events"], catalog, n_events=3, verbosity="full")
    compact = verbalize(u["events"], catalog, n_events=3, verbosity="compact")
    assert full.count("\n- Watched") == compact.count("\n- Watched") == min(3, len(u["events"]))
    assert "[Animation" not in compact and "— rated" in compact
    dropped = verbalize(u["events"], catalog, n_events=10**6, drop_low_signal=True)
    assert "rated 1\n" not in dropped and "rated 2\n" not in dropped
    truncated = verbalize(u["events"], catalog, n_events=10**6, fit_tokens=lambda p: len(p.split()), max_len=60)
    assert len(truncated.split("\n")) < len(full.split("\n")) + 2 and truncated.rstrip().endswith("Task: rank the catalog for this user.")
    print("\n[verbalize self-checks passed]", file=__import__("sys").stderr)
