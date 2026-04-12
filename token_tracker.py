#!/usr/bin/env python3
"""
Token Tracker — Parse Claude Code JSONL session logs to track token spending
and identify redundant/expensive tool calls.

Usage:
    python token_tracker.py                        # Analyze all sessions
    python token_tracker.py --project Token         # Filter by project name
    python token_tracker.py --session <uuid>        # Specific session
    python token_tracker.py --top 5                 # Top 5 costliest sessions
    python token_tracker.py --recommendations       # Show optimization tips
    python token_tracker.py --list-projects         # List all projects
    python token_tracker.py --list-sessions         # List all sessions
    python token_tracker.py --json                  # Machine-readable output
"""

import json
import os
import sys
import argparse

# Fix Windows console encoding — must happen before any print()
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# ANSI color support with Windows fallback
# ─────────────────────────────────────────────────────────────────────────────

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # Try to enable ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not (mode.value & 0x0004):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else str(text)


def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def cyan(t):    return _c("36", t)
def magenta(t): return _c("35", t)


# ─────────────────────────────────────────────────────────────────────────────
# Model pricing (per 1M tokens)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PRICING = {
    "claude-opus-4-6":              {"input": 5.00,  "output": 25.00},
    "claude-opus-4-20250918":       {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-5-20250929":   {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5":            {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":    {"input": 1.00,  "output": 5.00},
    "claude-haiku-4-5":             {"input": 1.00,  "output": 5.00},
}

# Cache pricing as multiplier of base input price
CACHE_WRITE_5M_MULT = 1.25   # 5-minute ephemeral: 25% more than input
CACHE_WRITE_1H_MULT = 2.00   # 1-hour ephemeral: 2x input price
CACHE_READ_MULT     = 0.10   # 90% less than input

# Web search cost
WEB_SEARCH_COST = 0.01  # $0.01 per search request


# ─────────────────────────────────────────────────────────────────────────────
# Subscription loader — reads subscriptions.json next to this script
# ─────────────────────────────────────────────────────────────────────────────

def _load_subscriptions():
    """Load subscriptions from subscriptions.json beside this script."""
    sub_path = Path(__file__).resolve().parent / "subscriptions.json"
    if not sub_path.exists():
        return []
    try:
        with open(sub_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("subscriptions", [])
    except (json.JSONDecodeError, OSError):
        return []


def _load_budgets():
    """Load budget thresholds from subscriptions.json."""
    sub_path = Path(__file__).resolve().parent / "subscriptions.json"
    if not sub_path.exists():
        return {}
    try:
        with open(sub_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("budgets", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _load_usage_log():
    """Load manual usage log from usage_log.json beside this script."""
    log_path = Path(__file__).resolve().parent / "usage_log.json"
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("usage", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_usage_log(entries):
    """Write usage entries back to usage_log.json."""
    log_path = Path(__file__).resolve().parent / "usage_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"usage": entries}, f, indent=2, default=str)


def _sync_usage(analyses):
    """Auto-populate Claude Code entries in usage_log.json from JSONL session data."""
    # Aggregate Claude costs by month
    monthly = defaultdict(lambda: {"cost": 0.0, "sessions": 0, "models": set()})
    for a in analyses:
        ts = a.get("start_time")
        if not ts:
            continue
        month = ts[:7]  # "YYYY-MM"
        monthly[month]["cost"] += a["total_cost"]
        monthly[month]["sessions"] += 1
        for m in a.get("models", []):
            if m and not m.startswith("<"):
                monthly[month]["models"].add(m)

    # Load existing usage log
    existing = _load_usage_log()

    # Load subscriptions to get Claude base cost
    subs = _load_subscriptions()
    claude_sub = next((s for s in subs if "anthropic" in s.get("name", "").lower()
                       or "claude" in s.get("name", "").lower()), None)
    base_cost = claude_sub["cost"] if claude_sub else 0.0

    # Build index of existing Claude entries by month
    claude_idx = {}
    for i, entry in enumerate(existing):
        svc = entry.get("service", "")
        if "claude" in svc.lower() or "anthropic" in svc.lower():
            claude_idx[entry.get("month")] = i

    updated = 0
    added = 0
    for month in sorted(monthly.keys()):
        info = monthly[month]
        models_str = ", ".join(sorted(info["models"])) if info["models"] else "unknown"
        overage = max(0, info["cost"] - base_cost)
        note = f"{info['sessions']} sessions, {models_str}. Auto-synced from JSONL."

        entry = {
            "month": month,
            "service": "Anthropic (Claude Code)",
            "category": "LLM",
            "cost_base": base_cost,
            "cost_overage": round(overage, 2),
            "note": note,
        }

        if month in claude_idx:
            existing[claude_idx[month]] = entry
            updated += 1
        else:
            existing.append(entry)
            added += 1

    _save_usage_log(existing)
    print(f"Synced Claude usage: {added} months added, {updated} months updated.", file=sys.stderr)


def _get_pricing(model_name):
    """Return pricing dict for a model, with fuzzy matching."""
    if not model_name:
        model_name = "unknown"
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # Fuzzy: strip trailing date segments
    for key, pricing in MODEL_PRICING.items():
        if model_name.startswith(key.split("-20")[0]):
            return pricing
    # Keyword fallback
    low = model_name.lower()
    if "opus" in low:
        return MODEL_PRICING["claude-opus-4-6"]
    if "haiku" in low:
        return MODEL_PRICING["claude-haiku-4-5"]
    # Default to Sonnet
    return MODEL_PRICING["claude-sonnet-4-5"]


def _calc_cost(usage, model):
    """Calculate dollar cost for one API call's usage block."""
    p = _get_pricing(model)
    inp   = usage.get("input_tokens", 0)
    out   = usage.get("output_tokens", 0)
    cr    = usage.get("cache_read_input_tokens", 0)

    # Differentiate 5-min vs 1-hour cache writes
    cd = usage.get("cache_creation", {})
    cw_5m = cd.get("ephemeral_5m_input_tokens", 0)
    cw_1h = cd.get("ephemeral_1h_input_tokens", 0)
    # Fallback: if breakdown missing, treat all as 1h
    cw_total = usage.get("cache_creation_input_tokens", 0)
    if cw_5m == 0 and cw_1h == 0 and cw_total > 0:
        cw_1h = cw_total

    # Web search cost
    stu = usage.get("server_tool_use", {})
    web_searches = stu.get("web_search_requests", 0)

    return (
        (inp / 1_000_000) * p["input"]
        + (cw_5m / 1_000_000) * p["input"] * CACHE_WRITE_5M_MULT
        + (cw_1h / 1_000_000) * p["input"] * CACHE_WRITE_1H_MULT
        + (cr / 1_000_000) * p["input"] * CACHE_READ_MULT
        + (out / 1_000_000) * p["output"]
        + web_searches * WEB_SEARCH_COST
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session scanner — discovers project dirs and JSONL files
# ─────────────────────────────────────────────────────────────────────────────

class SessionScanner:
    def __init__(self):
        self.base_dir = Path.home() / ".claude" / "projects"

    def list_projects(self):
        if not self.base_dir.exists():
            return []
        return sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

    def list_sessions(self, project_dir=None, include_subagents=False):
        dirs = [project_dir] if project_dir else self.list_projects()
        sessions = []
        for d in dirs:
            sessions.extend(d.glob("*.jsonl"))
            if include_subagents:
                sessions.extend(d.glob("*/subagents/*.jsonl"))
        return sorted(sessions, key=lambda f: f.stat().st_mtime, reverse=True)

    def find_project(self, query):
        for d in self.list_projects():
            if query.lower() in d.name.lower():
                return d
        return None

    def find_session(self, session_id):
        for d in self.list_projects():
            candidate = d / f"{session_id}.jsonl"
            if candidate.exists():
                return candidate
        return None


# ─────────────────────────────────────────────────────────────────────────────
# JSONL parser — streaming dedup by requestId / tool_use_id
# ─────────────────────────────────────────────────────────────────────────────

class SessionData:
    __slots__ = (
        "filepath", "session_id", "project_dir",
        "api_calls", "tool_calls", "tool_results",
        "user_messages", "start_time", "end_time", "models_used",
    )

    def __init__(self):
        self.filepath = None
        self.session_id = None
        self.project_dir = None
        self.api_calls = []          # deduplicated {request_id, model, usage, timestamp}
        self.tool_calls = []         # {id, name, input, timestamp, result, is_error, result_size}
        self.tool_results = {}       # tool_use_id -> result dict
        self.user_messages = []      # {content, timestamp}
        self.start_time = None
        self.end_time = None
        self.models_used = set()


def parse_session(filepath):
    """Parse a JSONL session file with streaming-chunk deduplication."""
    data = SessionData()
    data.filepath = filepath
    data.project_dir = filepath.parent.name

    request_map = {}        # requestId  -> {request_id, model, usage, timestamp}
    tool_call_map = {}      # tool_use_id -> tool-call dict
    tool_result_map = {}    # tool_use_id -> result dict
    timestamps = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            etype = entry.get("type")
            ts = entry.get("timestamp")
            if ts:
                timestamps.append(ts)

            if not data.session_id:
                sid = entry.get("sessionId")
                if sid:
                    data.session_id = sid

            # ── assistant entries (contain usage + tool_use blocks) ───────
            if etype == "assistant":
                req_id = entry.get("requestId")
                msg = entry.get("message", {})
                usage = msg.get("usage")
                model = msg.get("model")

                if model:
                    data.models_used.add(model)

                # Dedup: always overwrite with the latest chunk for this requestId
                if req_id and usage:
                    request_map[req_id] = {
                        "request_id": req_id,
                        "model": model,
                        "usage": usage,
                        "timestamp": ts,
                    }

                # Collect tool_use blocks (first occurrence per id)
                for block in (msg.get("content") or []):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tid = block.get("id")
                        if tid and tid not in tool_call_map:
                            tool_call_map[tid] = {
                                "id": tid,
                                "name": block.get("name", "unknown"),
                                "input": block.get("input", {}),
                                "timestamp": ts,
                                "request_id": req_id,
                            }

            # ── user entries (contain tool_result blocks + plain text) ────
            elif etype == "user":
                msg = entry.get("message", {})
                content = msg.get("content")

                if isinstance(content, str) and content.strip():
                    data.user_messages.append({"content": content, "timestamp": ts})

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        # Plain text in list form
                        if block.get("type") == "text":
                            txt = block.get("text", "")
                            if txt.strip():
                                data.user_messages.append({"content": txt, "timestamp": ts})

                        # Tool result
                        if block.get("type") == "tool_result":
                            tid = block.get("tool_use_id")
                            if not tid:
                                continue
                            res_content = block.get("content", "")
                            is_error = block.get("is_error", False)
                            detail = entry.get("toolUseResult", {})

                            result_size = len(str(res_content))
                            if isinstance(detail, dict):
                                stdout = detail.get("stdout", "")
                                stderr = detail.get("stderr", "")
                                result_size = max(result_size, len(str(stdout)) + len(str(stderr)))
                                # Check for Write/Edit results
                                fc = detail.get("content", "")
                                if fc:
                                    result_size = max(result_size, len(str(fc)))

                            tool_result_map[tid] = {
                                "tool_use_id": tid,
                                "content": res_content,
                                "is_error": is_error,
                                "result_size": result_size,
                            }

    # ── Finalize ──────────────────────────────────────────────────────────
    data.api_calls = list(request_map.values())

    for tid, tc in tool_call_map.items():
        res = tool_result_map.get(tid)
        tc["result"] = res
        tc["is_error"] = res.get("is_error", False) if res else False
        tc["result_size"] = res.get("result_size", 0) if res else 0
        data.tool_calls.append(tc)

    data.tool_results = tool_result_map

    if timestamps:
        data.start_time = min(timestamps)
        data.end_time = max(timestamps)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_session(data):
    """Compute costs, token totals, tool stats, and flag issues."""
    a = {
        "session_id":              data.session_id,
        "project":                 data.project_dir,
        "filepath":                str(data.filepath) if data.filepath else None,
        "models":                  sorted(data.models_used),
        "start_time":              data.start_time,
        "end_time":                data.end_time,
        "num_api_calls":           len(data.api_calls),
        "num_tool_calls":          len(data.tool_calls),
        "num_user_messages":       len(data.user_messages),
        "total_cost":              0.0,
        "total_input_tokens":      0,
        "total_output_tokens":     0,
        "total_cache_write_tokens": 0,
        "total_cache_read_tokens": 0,
        "cost_by_model":           defaultdict(float),
        "tokens_by_model":         defaultdict(lambda: {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}),
        "tool_call_counts":        Counter(),
        "tool_errors":             [],
        "large_results":           [],
        "duplicate_tools":         [],
        "cache_ephemeral_5m":      0,
        "cache_ephemeral_1h":      0,
        "web_search_requests":     0,
        # ── New fields for enhanced recommendations ──────────────────────
        "session_duration_min":    0.0,      # duration in minutes
        "output_cost_pct":         0.0,      # % of cost from output tokens
        "bash_grep_count":         0,        # Bash calls containing grep/rg/cat/find
        "read_large_count":        0,        # Read calls with result > 5K chars (no offset)
        "edit_file_targets":       Counter(),  # file -> count of edits
        "tool_result_total_chars": 0,        # total chars across all tool results
    }

    # ── Token & cost aggregation ──────────────────────────────────────────
    for call in data.api_calls:
        usage = call.get("usage", {})
        model = call.get("model", "unknown")

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cr  = usage.get("cache_read_input_tokens", 0)

        # Use nested cache_creation breakdown when available (matches _calc_cost)
        cd = usage.get("cache_creation", {})
        cw_5m = cd.get("ephemeral_5m_input_tokens", 0)
        cw_1h = cd.get("ephemeral_1h_input_tokens", 0)
        cw = cw_5m + cw_1h or usage.get("cache_creation_input_tokens", 0)

        a["total_input_tokens"]       += inp
        a["total_output_tokens"]      += out
        a["total_cache_write_tokens"] += cw
        a["total_cache_read_tokens"]  += cr

        cost = _calc_cost(usage, model)
        a["total_cost"] += cost
        a["cost_by_model"][model] += cost

        tbm = a["tokens_by_model"][model]
        tbm["input"]       += inp
        tbm["output"]      += out
        tbm["cache_write"] += cw
        tbm["cache_read"]  += cr

        a["cache_ephemeral_5m"] += cw_5m
        a["cache_ephemeral_1h"] += cw_1h

        stu = usage.get("server_tool_use", {})
        a["web_search_requests"] += stu.get("web_search_requests", 0)

    # ── Session duration ──────────────────────────────────────────────────
    if data.start_time and data.end_time:
        try:
            t0 = datetime.fromisoformat(data.start_time.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(data.end_time.replace("Z", "+00:00"))
            a["session_duration_min"] = max(0, (t1 - t0).total_seconds() / 60)
        except Exception:
            pass

    # ── Output cost percentage ───────────────────────────────────────────
    if a["total_cost"] > 0:
        out_cost = 0.0
        for call in data.api_calls:
            usage = call.get("usage", {})
            model = call.get("model", "unknown")
            p = _get_pricing(model)
            out_cost += (usage.get("output_tokens", 0) / 1_000_000) * p["output"]
        a["output_cost_pct"] = (out_cost / a["total_cost"]) * 100

    # ── Tool-call analysis ────────────────────────────────────────────────
    inputs_by_name = defaultdict(list)  # name -> [(input_json, tc), ...]

    for tc in data.tool_calls:
        name = tc["name"]
        a["tool_call_counts"][name] += 1

        if tc["is_error"]:
            a["tool_errors"].append(tc)

        if tc["result_size"] > 10_000:
            a["large_results"].append(tc)

        a["tool_result_total_chars"] += tc.get("result_size", 0)

        input_key = json.dumps(tc.get("input", {}), sort_keys=True)
        inputs_by_name[name].append((input_key, tc))

        # ── Detect anti-patterns in tool inputs ──────────────────────
        tc_input = tc.get("input", {})

        # Bash calls that should use dedicated tools
        if name in ("Bash", "bash"):
            cmd = str(tc_input.get("command", ""))
            if any(kw in cmd for kw in ["grep ", "rg ", " cat ", "find ", "head ", "tail "]):
                a["bash_grep_count"] += 1

        # Read calls returning large results without offset
        if name in ("Read", "read"):
            if tc.get("result_size", 0) > 5000 and not tc_input.get("offset") and not tc_input.get("limit"):
                a["read_large_count"] += 1

        # Edit calls — track target files
        if name in ("Edit", "edit"):
            fp = tc_input.get("file_path", "")
            if fp:
                a["edit_file_targets"][fp] += 1

    # Duplicate detection: same tool + identical input
    for name, entries in inputs_by_name.items():
        seen = defaultdict(list)
        for key, tc in entries:
            seen[key].append(tc)
        for key, tcs in seen.items():
            if len(tcs) > 1:
                a["duplicate_tools"].append({
                    "tool": name,
                    "count": len(tcs),
                    "input_preview": tcs[0].get("input", {}),
                })

    return a


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(a):
    """Return a list of actionable optimization suggestions."""
    recs = []

    total_out = a["total_output_tokens"]
    total_in  = (a["total_input_tokens"]
                 + a["total_cache_write_tokens"]
                 + a["total_cache_read_tokens"])

    # ── Duplicate tool calls ──────────────────────────────────────────────
    for dup in a["duplicate_tools"]:
        recs.append({
            "severity": "high",
            "title":  f"Duplicate {dup['tool']} calls ({dup['count']}x identical input)",
            "detail": (f"The same input was sent to {dup['tool']} {dup['count']} times. "
                       "Each duplicate wastes output tokens and an API round-trip."),
        })

    # ── High error rate ───────────────────────────────────────────────────
    n_err   = len(a.get("tool_errors", []))
    n_tools = a.get("num_tool_calls", 0) or a.get("tool_call_counts", Counter()).total() if hasattr(Counter, 'total') else sum(a.get("tool_call_counts", {}).values())
    if n_tools > 2 and n_err / n_tools > 0.10:
        recs.append({
            "severity": "high",
            "title":  f"High tool error rate ({n_err}/{n_tools} = {n_err/n_tools:.0%})",
            "detail": "Errors force the model to retry, doubling token spend per failed call.",
        })

    # ── Large tool results ────────────────────────────────────────────────
    big = sorted(a.get("large_results", []), key=lambda t: -t["result_size"])
    for lr in big[:3]:
        recs.append({
            "severity": "medium",
            "title":  f"Large {lr['name']} result ({lr['result_size']:,} chars)",
            "detail": "Oversized tool results inflate input tokens on every subsequent turn.",
        })

    # ── High output-to-input ratio ────────────────────────────────────────
    if total_in > 1000 and total_out > 1000:
        ratio = total_out / total_in
        if ratio > 0.25:
            recs.append({
                "severity": "medium",
                "title":  f"High output/input ratio ({ratio:.0%})",
                "detail": (f"Output tokens ({total_out:,}) are expensive — "
                           "consider requesting concise responses or using Haiku for exploratory steps."),
            })

    # ── Low cache-read utilization ────────────────────────────────────────
    cw = a.get("total_cache_write_tokens", 0)
    cr = a.get("total_cache_read_tokens", 0)
    n_api = a.get("num_api_calls", 0)
    if cw > 5000 and n_api > 3 and cr < cw * 0.5:
        recs.append({
            "severity": "low",
            "title":  "Low prompt-cache hit rate",
            "detail": (f"Cache reads ({cr:,}) are low vs. cache writes ({cw:,}). "
                       "Frequent context changes reduce cache reuse."),
        })

    # ── Many API round-trips ─────────────────────────────────────────────
    if n_api > 25:
        recs.append({
            "severity": "low",
            "title":  f"Many API round-trips ({n_api})",
            "detail": "Provide more upfront context or batch instructions to reduce calls.",
        })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: severity_order.get(r["severity"], 9))
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fc(cost):
    """Format a dollar cost."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _ft(n):
    """Format a token count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _ftime(iso):
    """Format an ISO timestamp for display."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16]


def _project_label(name):
    """Make project dir names more readable.

    Converts encoded paths like
      C--Users-tdamy-OneDrive---CLS-CRE-CLS-CRE-Brokerage-AI---ChatGPT-Claude-Code-Lender-Scraper
    into clean names like "Lender Scraper".
    """
    if not name:
        return "Unknown"

    import re

    # Claude Code encodes paths: / → -- and space → -
    # Split on --- (original triple-hyphen or deeper separators) first
    segments = name.split("---")
    # Take the last segment (the project folder name)
    last = segments[-1].strip("-")

    # If the last segment still has -- separators (subfolder encoding), take last part
    if "--" in last:
        sub_parts = last.split("--")
        last = sub_parts[-1].strip("-")

    # Strip common path prefixes that aren't meaningful project names
    # e.g. "ChatGPT-Claude-Code-Lender-Scraper" → "Lender-Scraper"
    last = re.sub(r'^(?:ChatGPT-)?Claude-Code-', '', last)

    # Convert remaining single hyphens to spaces for readability
    label = last.replace("-", " ").strip()

    # Title-case it for display, then fix known acronyms (whole words only)
    if label:
        result = label.title()
        for acronym in ("Hbu", "La", "Ai", "Api", "Cre", "Cls"):
            result = re.sub(r'\b' + acronym + r'\b', acronym.upper(), result)
        return result
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Display functions
# ─────────────────────────────────────────────────────────────────────────────

def display_session(a, show_recs=False):
    """Pretty-print one session analysis."""
    sid = a["session_id"] or "unknown"
    bar = "═" * min(len(sid) + 14, 70)

    print()
    print(bold(f"═══ Session: {sid} ═══"))
    print(f"  Project : {dim(_project_label(a['project'] or '?'))}")
    print(f"  Time    : {_ftime(a['start_time'])}  →  {_ftime(a['end_time'])}")
    print(f"  Models  : {', '.join(a['models']) or '?'}")
    print(f"  Messages: {a['num_user_messages']}  API calls: {a['num_api_calls']}  Tool calls: {a['num_tool_calls']}")
    print()

    # ── Token / cost table ────────────────────────────────────────────────
    inp = a["total_input_tokens"]
    out = a["total_output_tokens"]
    cw  = a["total_cache_write_tokens"]
    cr  = a["total_cache_read_tokens"]

    print(bold("  Token Breakdown"))
    print(f"  {'─' * 45}")
    print(f"  {'Input (non-cached):':<30} {_ft(inp):>12}")
    print(f"  {'Cache write:':<30} {_ft(cw):>12}")
    print(f"  {'Cache read:':<30} {_ft(cr):>12}")
    print(f"  {'Output:':<30} {_ft(out):>12}")

    e5 = a.get("cache_ephemeral_5m", 0)
    e1 = a.get("cache_ephemeral_1h", 0)
    if e5 or e1:
        print(f"  {'─' * 45}")
        print(f"  {'  5-min ephemeral cache:':<30} {_ft(e5):>12}")
        print(f"  {'  1-hour ephemeral cache:':<30} {_ft(e1):>12}")

    ws = a.get("web_search_requests", 0)
    if ws:
        print(f"  {'─' * 45}")
        print(f"  {'Web searches:':<30} {str(ws):>12}  ({_fc(ws * WEB_SEARCH_COST)})")

    print(f"  {'─' * 45}")
    print(f"  {'Total estimated cost:':<30} {green(bold(_fc(a['total_cost']))):>12}")
    print()

    # ── Per-model breakdown (if multiple) ─────────────────────────────────
    if len(a["cost_by_model"]) > 1:
        print(bold("  Cost by Model"))
        for model in sorted(a["cost_by_model"], key=lambda m: -a["cost_by_model"][m]):
            c = a["cost_by_model"][model]
            t = a["tokens_by_model"][model]
            print(f"    {model}: {_fc(c)}  "
                  f"(in:{_ft(t['input'])} cw:{_ft(t['cache_write'])} cr:{_ft(t['cache_read'])} out:{_ft(t['output'])})")
        print()

    # ── Tool usage table ──────────────────────────────────────────────────
    if a["tool_call_counts"]:
        err_by_tool = Counter(tc["name"] for tc in a["tool_errors"])

        print(bold("  Tool Usage"))
        print(f"  {'─' * 45}")
        print(f"  {'Tool':<28} {'Calls':>6}  {'Errors':>6}")
        print(f"  {'─' * 45}")
        for tool, cnt in a["tool_call_counts"].most_common():
            errs = err_by_tool.get(tool, 0)
            e_str = red(str(errs)) if errs else dim("0")
            print(f"  {tool:<28} {cnt:>6}  {e_str:>6}")
        total_tc = sum(a["tool_call_counts"].values())
        total_err = len(a["tool_errors"])
        te_str = red(str(total_err)) if total_err else dim("0")
        print(f"  {'─' * 45}")
        print(f"  {'TOTAL':<28} {total_tc:>6}  {te_str:>6}")
        print()

    # ── Flagged issues ────────────────────────────────────────────────────
    if a["duplicate_tools"]:
        print(bold(yellow("  Duplicate Tool Calls")))
        for dup in a["duplicate_tools"]:
            preview = str(dup["input_preview"])
            if len(preview) > 80:
                preview = preview[:77] + "..."
            print(f"    {dup['tool']} x{dup['count']}: {dim(preview)}")
        print()

    if a["large_results"]:
        print(bold(yellow("  Large Tool Results (>10K chars)")))
        for lr in sorted(a["large_results"], key=lambda t: -t["result_size"])[:5]:
            print(f"    {lr['name']}: {lr['result_size']:,} chars")
        print()

    # ── Recommendations ───────────────────────────────────────────────────
    if show_recs:
        recs = generate_recommendations(a)
        _display_recs(recs)


def _display_recs(recs):
    if not recs:
        print(f"  {green('No recommendations — session looks efficient!')}")
        print()
        return
    print(bold("  Recommendations"))
    print(f"  {'─' * 45}")
    icons = {"high": red("!"), "medium": yellow("~"), "low": dim("-")}
    for r in recs:
        icon = icons.get(r["severity"], "?")
        print(f"  [{icon}] {bold(r['title'])}")
        print(f"      {dim(r['detail'])}")
    print()


def display_project_summary(analyses, top_n=5):
    """Show aggregate stats across multiple sessions."""
    print()
    print(bold("═══ Aggregate Summary ═══"))
    print()

    total_cost = sum(a["total_cost"] for a in analyses)
    total_api  = sum(a["num_api_calls"] for a in analyses)
    total_tc   = sum(a["num_tool_calls"] for a in analyses)
    total_in   = sum(a["total_input_tokens"] for a in analyses)
    total_out  = sum(a["total_output_tokens"] for a in analyses)
    total_cw   = sum(a["total_cache_write_tokens"] for a in analyses)
    total_cr   = sum(a["total_cache_read_tokens"] for a in analyses)

    print(f"  Sessions analyzed : {len(analyses)}")
    print(f"  Total API calls   : {total_api:,}")
    print(f"  Total tool calls  : {total_tc:,}")
    print()
    print(f"  Input tokens      : {_ft(total_in)}")
    print(f"  Cache write tokens: {_ft(total_cw)}")
    print(f"  Cache read tokens : {_ft(total_cr)}")
    print(f"  Output tokens     : {_ft(total_out)}")
    print()
    print(f"  {bold('Total estimated cost')}: {green(bold(_fc(total_cost)))}")
    print()

    # ── Cost by model across all sessions ─────────────────────────────────
    model_cost = defaultdict(float)
    for a in analyses:
        for m, c in a["cost_by_model"].items():
            model_cost[m] += c
    if model_cost:
        print(bold("  Cost by Model (all sessions)"))
        for m in sorted(model_cost, key=lambda x: -model_cost[x]):
            print(f"    {m}: {_fc(model_cost[m])}")
        print()

    # ── Top sessions table ────────────────────────────────────────────────
    ranked = sorted(analyses, key=lambda x: -x["total_cost"])
    show = ranked[:top_n]
    if show:
        print(bold(f"  Top {len(show)} Sessions by Cost"))
        print(f"  {'─' * 72}")
        print(f"  {'Session ID':<38} {'Cost':>8}  {'API':>4}  {'Tools':>5}  {'Time':>16}")
        print(f"  {'─' * 72}")
        for s in show:
            sid = (s["session_id"] or "?")[:36]
            print(f"  {sid:<38} {_fc(s['total_cost']):>8}  "
                  f"{s['num_api_calls']:>4}  {s['num_tool_calls']:>5}  {_ftime(s['start_time']):>16}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track Claude Code token spending from JSONL session logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                          Summarize all sessions
  %(prog)s -p Token                 Sessions whose project contains "Token"
  %(prog)s -s <session-uuid>        Analyze one session in detail
  %(prog)s --top 5                  Show top 5 costliest sessions
  %(prog)s -r                       Include optimization recommendations
  %(prog)s --list-projects          List discovered projects
  %(prog)s --list-sessions          List all session files
  %(prog)s --json                   Output machine-readable JSON
  %(prog)s --html                   Open interactive dashboard in browser
  %(prog)s --watch                  Live dashboard (auto-refreshes every 30s)
  %(prog)s --publish                Generate docs/index.html for GitHub Pages
  %(prog)s --sync-usage             Auto-populate Claude entries in usage_log.json
""",
    )
    parser.add_argument("-s", "--session",       help="Analyze a specific session by UUID")
    parser.add_argument("-p", "--project",       help="Filter by project name (substring match)")
    parser.add_argument("-t", "--top",           type=int, default=5, help="Number of top sessions to show (default: 5)")
    parser.add_argument("-r", "--recommendations", action="store_true", help="Show optimization recommendations")
    parser.add_argument("--list-projects",       action="store_true", help="List all project directories")
    parser.add_argument("--list-sessions",       action="store_true", help="List all session log files")
    parser.add_argument("--json",                action="store_true", help="Output as JSON")
    parser.add_argument("--html",                action="store_true", help="Open interactive dashboard in browser")
    parser.add_argument("--watch",               action="store_true", help="Live dashboard with auto-refresh (implies --html)")
    parser.add_argument("--publish",             action="store_true", help="Generate docs/index.html for GitHub Pages")
    parser.add_argument("--sync-usage",           action="store_true", help="Auto-populate Claude entries in usage_log.json from JSONL data")
    parser.add_argument("--no-color",            action="store_true", help="Disable colored output")
    args = parser.parse_args()

    if args.no_color:
        global USE_COLOR
        USE_COLOR = False

    scanner = SessionScanner()

    if not scanner.base_dir.exists():
        print(red(f"Error: {scanner.base_dir} not found. Is Claude Code installed?"), file=sys.stderr)
        sys.exit(1)

    # ── Sync usage log ────────────────────────────────────────────────────
    if args.sync_usage:
        sessions = scanner.list_sessions()
        print(dim(f"Scanning {len(sessions)} session(s)..."), file=sys.stderr)
        analyses = []
        for fp in sessions:
            try:
                d = parse_session(fp)
                a = analyze_session(d)
                if a["num_api_calls"] > 0:
                    analyses.append(a)
            except Exception:
                pass
        _sync_usage(analyses)
        return

    # ── Watch mode (live server) ──────────────────────────────────────────
    if args.watch:
        proj = scanner.find_project(args.project) if args.project else None
        if args.project and not proj:
            print(red(f"No project matching '{args.project}'"), file=sys.stderr)
            sys.exit(1)
        sid = args.session if args.session else None
        _run_watch_server(scanner, project_dir=proj, session_id=sid)
        return

    # ── List projects ─────────────────────────────────────────────────────
    if args.list_projects:
        projects = scanner.list_projects()
        if not projects:
            print("No projects found.")
            return
        print(bold("Projects:"))
        for p in projects:
            n_sess = len(list(p.glob("*.jsonl")))
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
            label = _project_label(p.name)
            print(f"  {label:<50} {n_sess:>3} sessions   {dim(mtime)}")
        return

    # ── List sessions ─────────────────────────────────────────────────────
    if args.list_sessions:
        proj = scanner.find_project(args.project) if args.project else None
        if args.project and not proj:
            print(red(f"No project matching '{args.project}'"), file=sys.stderr)
            sys.exit(1)
        sessions = scanner.list_sessions(proj)
        if not sessions:
            print("No sessions found.")
            return
        hdr = f"Sessions"
        if proj:
            hdr += f" ({_project_label(proj.name)})"
        print(bold(hdr + ":"))
        for s in sessions:
            sz = s.stat().st_size
            sz_str = f"{sz/1024:.0f}KB" if sz < 1_048_576 else f"{sz/1_048_576:.1f}MB"
            mt = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {s.stem}  {sz_str:>8}  {mt}  {dim(_project_label(s.parent.name))}")
        return

    # ── Single session ────────────────────────────────────────────────────
    if args.session:
        fp = scanner.find_session(args.session)
        if not fp:
            print(red(f"Session '{args.session}' not found."), file=sys.stderr)
            sys.exit(1)
        data = parse_session(fp)
        analysis = analyze_session(data)

        if args.publish:
            _publish_html([analysis])
        elif args.html:
            _generate_html([analysis])
        elif args.json:
            _print_json([analysis])
        else:
            display_session(analysis, show_recs=args.recommendations)
        return

    # ── All sessions (optionally project-filtered) ────────────────────────
    proj = None
    if args.project:
        proj = scanner.find_project(args.project)
        if not proj:
            print(red(f"No project matching '{args.project}'"), file=sys.stderr)
            sys.exit(1)

    sessions = scanner.list_sessions(proj)
    if not sessions:
        print("No sessions found.")
        return

    print(dim(f"Scanning {len(sessions)} session(s)..."), file=sys.stderr)

    analyses = []
    for fp in sessions:
        try:
            d = parse_session(fp)
            a = analyze_session(d)
            if a["num_api_calls"] > 0:  # skip empty / metadata-only sessions
                analyses.append(a)
        except Exception as exc:
            print(dim(f"  skip {fp.name}: {exc}"), file=sys.stderr)

    if not analyses:
        print("No sessions with API calls found.")
        return

    if args.publish:
        _publish_html(analyses)
        return

    if args.html:
        _generate_html(analyses)
        return

    if args.json:
        _print_json(analyses)
        return

    display_project_summary(analyses, top_n=args.top)

    # Show detailed view for top N
    ranked = sorted(analyses, key=lambda x: -x["total_cost"])
    for a in ranked[: args.top]:
        display_session(a, show_recs=args.recommendations)

    # Overall recommendations
    if args.recommendations and len(analyses) > 1:
        print(bold("═══ Overall Recommendations ═══"))
        combined = {
            "total_cost":              sum(a["total_cost"] for a in analyses),
            "total_input_tokens":      sum(a["total_input_tokens"] for a in analyses),
            "total_output_tokens":     sum(a["total_output_tokens"] for a in analyses),
            "total_cache_write_tokens": sum(a["total_cache_write_tokens"] for a in analyses),
            "total_cache_read_tokens": sum(a["total_cache_read_tokens"] for a in analyses),
            "num_api_calls":           sum(a["num_api_calls"] for a in analyses),
            "num_tool_calls":          sum(a["num_tool_calls"] for a in analyses),
            "tool_errors":             [e for a in analyses for e in a["tool_errors"]],
            "large_results":           [r for a in analyses for r in a["large_results"]],
            "duplicate_tools":         [d for a in analyses for d in a["duplicate_tools"]],
            "tool_call_counts":        Counter(),
        }
        for a in analyses:
            combined["tool_call_counts"] += Counter(a["tool_call_counts"])
        recs = generate_recommendations(combined)
        _display_recs(recs)


# ─────────────────────────────────────────────────────────────────────────────
# HTML dashboard template
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Token Tracker</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; --blue: #58a6ff; --purple: #bc8cff; --cyan: #39d2c0;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; overflow: hidden; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; display: flex; flex-direction: column; }

  /* Top bar */
  .topbar { padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; background: var(--bg); }
  .topbar-left { display: flex; align-items: center; gap: 20px; }
  .topbar-left h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
  .topbar-meta { color: var(--dim); font-size: 12px; }
  .topbar-spend { font-size: 28px; font-weight: 700; color: var(--green); }

  /* Summary strip */
  .summary-strip { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-shrink: 0; background: var(--bg2); overflow-x: auto; }
  .strip-card { padding: 10px 20px; border-right: 1px solid var(--border); min-width: 140px; }
  .strip-card:last-child { border-right: none; }
  .strip-label { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  .strip-value { font-size: 18px; font-weight: 700; margin-top: 2px; }
  .strip-detail { color: var(--dim); font-size: 11px; }

  /* Budget bars */
  .budget-strip { display: flex; gap: 16px; padding: 8px 24px; background: var(--bg2); border-bottom: 1px solid var(--border); flex-shrink: 0; overflow-x: auto; }
  .budget-item { flex: 1; min-width: 200px; }
  .budget-label { font-size: 11px; color: var(--dim); margin-bottom: 3px; display: flex; justify-content: space-between; }
  .budget-track { height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; position: relative; }
  .budget-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
  .budget-alert { font-size: 10px; color: var(--red); font-weight: 600; margin-top: 2px; }

  /* Main nav tabs */
  .nav-bar { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-shrink: 0; background: var(--bg); padding: 0 24px; }
  .nav-tab { padding: 10px 20px; cursor: pointer; font-size: 13px; font-weight: 500; color: var(--dim); border-bottom: 2px solid transparent; transition: all 0.15s; }
  .nav-tab:hover { color: var(--text); }
  .nav-tab.active { color: var(--blue); border-bottom-color: var(--blue); }
  .nav-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
  .nav-right label { color: var(--dim); font-size: 12px; display: flex; align-items: center; gap: 4px; }
  .nav-right input[type="date"] { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 3px 8px; border-radius: 4px; font-size: 12px; color-scheme: dark; }
  .nav-right .filter-btn { background: none; border: none; color: var(--dim); cursor: pointer; font-size: 11px; text-decoration: underline; }
  .nav-right .filter-btn:hover { color: var(--text); }

  /* Tab panels */
  .tab-panel { flex: 1; overflow-y: auto; padding: 20px 24px; display: none; }
  .tab-panel.active { display: block; }

  /* Tables — compact */
  table { width: 100%; border-collapse: collapse; background: var(--bg2); border-radius: 6px; overflow: hidden; border: 1px solid var(--border); font-size: 13px; }
  th { background: var(--bg3); color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding: 7px 12px; text-align: left; font-weight: 600; }
  th.right, td.right { text-align: right; }
  td { padding: 7px 12px; border-top: 1px solid var(--border); }
  tr:hover td { background: rgba(56, 139, 253, 0.04); }
  tfoot td { background: var(--bg3); font-weight: 600; border-top: 2px solid var(--border); }

  /* Section inside panels */
  .section { margin-bottom: 20px; }
  .section-title { font-size: 14px; font-weight: 600; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  .section-subtitle { font-size: 12px; color: var(--dim); font-weight: 400; margin-left: 6px; }

  /* Bar chart cells */
  .bar-cell { width: 25%; }
  .bar-wrap { display: flex; align-items: center; gap: 6px; }
  .bar { height: 8px; border-radius: 4px; min-width: 2px; transition: width 0.3s; }
  .bar-pct { font-size: 10px; color: var(--dim); white-space: nowrap; }

  /* Colors */
  .cost-green { color: var(--green); }
  .cost-yellow { color: var(--yellow); }
  .cost-red { color: var(--red); }
  .cost { color: var(--green); }

  /* Two-column */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

  /* Recommendations */
  .rec-item { padding: 10px 12px; border-left: 3px solid; margin-bottom: 6px; background: var(--bg3); border-radius: 0 6px 6px 0; font-size: 13px; }
  .rec-item.high { border-color: var(--red); }
  .rec-item.medium { border-color: var(--yellow); }
  .rec-item.low { border-color: var(--dim); }
  .rec-item.info { border-color: var(--blue); }
  .rec-title { font-weight: 600; margin-bottom: 3px; }
  .rec-detail { font-size: 12px; color: var(--dim); }
  .rec-stat { display: inline-block; background: var(--bg); padding: 1px 6px; border-radius: 3px; font-family: 'SFMono-Regular', Consolas, monospace; font-size: 11px; color: var(--cyan); }
  .no-recs { color: var(--green); font-size: 13px; padding: 8px 0; }

  /* Category badge */
  .cat-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
  .cat-llm { background: rgba(188,140,255,0.15); color: var(--purple); }
  .cat-search { background: rgba(57,210,192,0.15); color: var(--cyan); }
  .cat-tooling { background: rgba(88,166,255,0.15); color: var(--blue); }
  .cat-automation { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .cat-video { background: rgba(248,81,73,0.15); color: var(--red); }
  .cat-other { background: rgba(139,148,158,0.15); color: var(--dim); }

  /* Projection row */
  .projected { font-style: italic; opacity: 0.7; }
  .projected td { border-top: 2px dashed var(--border); }

  /* Timeline sub-tabs */
  .sub-tabs { display: flex; gap: 4px; margin-bottom: 10px; }
  .sub-tab { background: var(--bg3); border: 1px solid var(--border); color: var(--dim); padding: 4px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 500; }
  .sub-tab:hover { color: var(--text); border-color: var(--blue); }
  .sub-tab.active { background: var(--blue); color: #fff; border-color: var(--blue); }

  .timestamp { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 12px; }

  /* Savings guide */
  .guide-item { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; display: flex; gap: 14px; align-items: flex-start; }
  .guide-score { min-width: 56px; text-align: center; padding: 4px 0; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; flex-shrink: 0; }
  .guide-score.critical { background: rgba(248,81,73,0.15); color: var(--red); }
  .guide-score.high-rel { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .guide-score.moderate { background: rgba(88,166,255,0.15); color: var(--blue); }
  .guide-score.low-rel { background: rgba(139,148,158,0.1); color: var(--dim); }
  .guide-score.applied { background: rgba(63,185,80,0.15); color: var(--green); }
  .guide-body { flex: 1; }
  .guide-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
  .guide-how { font-size: 12px; color: var(--dim); line-height: 1.5; }
  .guide-how code { background: var(--bg3); padding: 1px 5px; border-radius: 3px; font-size: 11px; }
  .guide-how strong { color: var(--text); }
  .guide-savings { display: inline-block; margin-top: 4px; background: rgba(63,185,80,0.1); color: var(--green); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }

  @media (max-width: 900px) {
    .two-col { grid-template-columns: 1fr; }
    .summary-strip { flex-wrap: wrap; }
  }
</style>
</head>
<body>

<!-- Top bar: title + total spend -->
<div class="topbar">
  <div class="topbar-left">
    <h1>Token Tracker</h1>
    <span class="topbar-meta" id="date-range"></span>
    <span class="topbar-meta" id="last-refreshed"></span>
    <span class="topbar-meta" id="filter-info"></span>
  </div>
  <div class="topbar-spend" id="total-spend"></div>
</div>

<!-- Summary strip -->
<div class="summary-strip" id="summary-cards"></div>
<div class="budget-strip" id="budget-bars"></div>

<!-- Navigation tabs + date filter -->
<div class="nav-bar">
  <div class="nav-tab active" onclick="switchTab('overview')">Overview</div>
  <div class="nav-tab" onclick="switchTab('projects')">Projects & Models</div>
  <div class="nav-tab" onclick="switchTab('timeline')">Timeline</div>
  <div class="nav-tab" onclick="switchTab('subscriptions')">Subscriptions</div>
  <div class="nav-tab" onclick="switchTab('sessions')">Sessions</div>
  <div class="nav-tab" onclick="switchTab('savings')">Savings Guide</div>
  <div class="nav-right">
    <label>From <input type="date" id="filter-start"></label>
    <label>To <input type="date" id="filter-end"></label>
    <button class="filter-btn" id="filter-reset">Reset</button>
  </div>
</div>

<!-- ═══ Tab: Overview ═══ -->
<div class="tab-panel active" id="panel-overview">
  <div class="two-col">
    <div class="section">
      <div class="section-title">All AI Spend — Current Month</div>
      <table>
        <thead><tr><th>Service</th><th>Cat</th><th class="right">Base</th><th class="right">Usage</th><th class="right">Total</th><th>Note</th></tr></thead>
        <tbody id="all-ai-table"></tbody>
        <tfoot id="all-ai-footer"></tfoot>
      </table>
    </div>
    <div class="section">
      <div class="section-title">Cost-Saving Insights</div>
      <div id="recommendations"></div>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Monthly Report</div>
    <table>
      <thead><tr><th>Month</th><th class="right">Claude Tokens</th><th class="right">Other AI</th><th class="right">Subs</th><th class="right">Total</th><th class="bar-cell"></th></tr></thead>
      <tbody id="monthly-report-table"></tbody>
      <tfoot id="monthly-report-footer"></tfoot>
    </table>
  </div>
</div>

<!-- ═══ Tab: Projects & Models ═══ -->
<div class="tab-panel" id="panel-projects">
  <div class="two-col">
    <div class="section">
      <div class="section-title">Spending by Project</div>
      <table>
        <thead><tr><th>Project</th><th class="right">Spend</th><th class="right">Sessions</th><th class="right">Msgs</th><th class="bar-cell"></th></tr></thead>
        <tbody id="project-table"></tbody>
      </table>
    </div>
    <div class="section">
      <div class="section-title">Breakdown by Model</div>
      <table>
        <thead><tr><th>Model</th><th class="right">Spend</th><th class="right">Input</th><th class="right">Output</th><th class="right">Cache W</th><th class="right">Cache R</th></tr></thead>
        <tbody id="model-table"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══ Tab: Timeline ═══ -->
<div class="tab-panel" id="panel-timeline">
  <div class="section">
    <div class="sub-tabs" id="timeline-tabs">
      <button class="sub-tab active" onclick="setTimelineMode('daily')">Daily</button>
      <button class="sub-tab" onclick="setTimelineMode('weekly')">Weekly</button>
      <button class="sub-tab" onclick="setTimelineMode('monthly')">Monthly</button>
    </div>
    <table>
      <thead><tr><th id="timeline-col-header">Date</th><th class="right">Spend</th><th class="right">Sessions</th><th class="bar-cell"></th></tr></thead>
      <tbody id="timeline-table"></tbody>
    </table>
  </div>
</div>

<!-- ═══ Tab: Subscriptions ═══ -->
<div class="tab-panel" id="panel-subscriptions">
  <div class="two-col">
    <div class="section">
      <div class="section-title">Active Subscriptions <span class="section-subtitle">Edit subscriptions.json</span></div>
      <table>
        <thead><tr><th>Service</th><th class="right">Monthly</th><th>Category</th><th>Since</th></tr></thead>
        <tbody id="subscriptions-table"></tbody>
        <tfoot id="subscriptions-footer"></tfoot>
      </table>
    </div>
    <div class="section">
      <div class="section-title">Usage Log <span class="section-subtitle">Edit usage_log.json</span></div>
      <table>
        <thead><tr><th>Month</th><th>Service</th><th class="right">Base</th><th class="right">Overage</th><th>Note</th></tr></thead>
        <tbody id="usage-log-table"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══ Tab: Sessions ═══ -->
<div class="tab-panel" id="panel-sessions">
  <div class="section">
    <table>
      <thead><tr><th>Project</th><th>Time</th><th class="right">Cost</th><th>Model</th><th class="right">Msgs</th><th class="right">Tools</th><th class="right">Errors</th></tr></thead>
      <tbody id="sessions-table"></tbody>
    </table>
  </div>
</div>

<!-- ═══ Tab: Savings Guide ═══ -->
<div class="tab-panel" id="panel-savings">
  <div class="section">
    <div class="section-title">Token Savings Guide <span class="section-subtitle">Tips scored by relevance to your usage</span></div>
    <p style="color:var(--dim);font-size:13px;margin-bottom:16px;line-height:1.5">Each tip is scored from your actual session data. Higher relevance means this tip would save you the most money based on your current patterns. Tips marked <span style="color:var(--green)">APPLIED</span> mean you're already doing well in that area.</p>
    <div id="savings-guide"></div>
  </div>
  <div class="section" style="margin-top:24px">
    <div class="section-title">Quick Reference: Token Cost Cheat Sheet</div>
    <table>
      <thead><tr><th>What</th><th class="right">Cost (Opus)</th><th class="right">Cost (Sonnet)</th><th>Tip</th></tr></thead>
      <tbody>
        <tr><td>1K input tokens (~750 words)</td><td class="right">$0.005</td><td class="right">$0.003</td><td style="color:var(--dim);font-size:12px">Every tool result re-sent = this cost per turn</td></tr>
        <tr><td>1K output tokens (~750 words)</td><td class="right" style="color:var(--red)">$0.025</td><td class="right" style="color:var(--yellow)">$0.015</td><td style="color:var(--dim);font-size:12px">Output is 5x more expensive than input</td></tr>
        <tr><td>1K cache read tokens</td><td class="right" style="color:var(--green)">$0.0005</td><td class="right" style="color:var(--green)">$0.0003</td><td style="color:var(--dim);font-size:12px">90% cheaper than fresh input &mdash; keep sessions focused</td></tr>
        <tr><td>1K cache write tokens</td><td class="right">$0.010</td><td class="right">$0.006</td><td style="color:var(--dim);font-size:12px">1-hour cache: 2x input price; amortized over reads</td></tr>
        <tr><td>1 web search</td><td class="right">$0.01</td><td class="right">$0.01</td><td style="color:var(--dim);font-size:12px">Flat rate per search request</td></tr>
        <tr><td>1 API round-trip (avg)</td><td class="right">~$0.15</td><td class="right">~$0.08</td><td style="color:var(--dim);font-size:12px">Fewer turns = fewer round-trips = less spend</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const DATA = %%DATA%%;
const SUBS = %%SUBS%%;
const USAGE_LOG = %%USAGE%%;
const BUDGETS = %%BUDGETS%%;

/* ── Helpers ── */
function fc(c){ if(c<0) return '-$'+(-c).toFixed(2); return c<0.01&&c>0 ? '<$0.01' : '$'+c.toFixed(2); }
function esc(s){ const d=document.createElement('div'); d.textContent=String(s??''); return d.innerHTML; }
function ft(n){ if(n>=1e9) return (n/1e9).toFixed(1)+'B'; if(n>=1e6) return (n/1e6).toFixed(1)+'M'; if(n>=1e3) return (n/1e3).toFixed(1)+'K'; return String(n); }
function ftime(iso){ if(!iso) return '\u2014'; const d=new Date(iso); return d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
function fdate(iso){ if(!iso) return '\u2014'; return iso.slice(0,10); }
function fmonth(ym){ if(!ym||!ym.includes('-')) return ym||'\u2014'; const [y,m]=ym.split('-'); const names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']; return (names[parseInt(m)-1]||m)+' '+y; }
function costColor(c){ return c>20?'cost-red':c>5?'cost-yellow':'cost-green'; }
function barColor(c){ return c>20?'var(--red)':c>5?'var(--yellow)':'var(--green)'; }
function catClass(c){ const k=(c||'').toLowerCase(); if(k==='llm') return 'cat-llm'; if(k==='search') return 'cat-search'; if(k==='tooling') return 'cat-tooling'; if(k==='automation') return 'cat-automation'; if(k==='video') return 'cat-video'; return 'cat-other'; }

function projectLabel(name){
  if(!name) return 'Unknown';
  const segs = name.split('---');
  let last = segs[segs.length-1].replace(/^-+|-+$/g,'');
  if(last.includes('--')){
    const sub = last.split('--');
    last = sub[sub.length-1].replace(/^-+|-+$/g,'');
  }
  last = last.replace(/^(?:ChatGPT-)?Claude-Code-/,'');
  let label = last.replace(/-/g,' ').trim();
  if(!label) return name;
  label = label.replace(/\b\w/g, c=>c.toUpperCase());
  ['Hbu','La','Ai','Api','Cre','Cls','Seo','Sb'].forEach(a=>{
    label = label.replace(new RegExp('\\b'+a+'\\b','g'), a.toUpperCase());
  });
  return label;
}

/* ── Date filter state ── */
let filterStart = null;
let filterEnd = null;

function filteredDATA(){
  return DATA.filter(s=>{
    if(!s.start_time) return true;
    const d = s.start_time.slice(0,10);
    if(filterStart && d < filterStart) return false;
    if(filterEnd && d > filterEnd) return false;
    return true;
  });
}

/* ── Subscription helpers ── */
function totalMonthlySub(){
  return SUBS.reduce((s,sub)=>s+(sub.cost||0), 0);
}

function subsActiveInMonth(ym){
  return SUBS.filter(sub=>{
    if(!sub.start_date) return true;
    return sub.start_date.slice(0,7) <= ym;
  }).reduce((s,sub)=>s+(sub.cost||0), 0);
}

/* ── Timeline aggregation ── */
let timelineMode = 'daily';

function aggregateTimeline(data, mode){
  const map = {};
  data.forEach(s => {
    if(!s.start_time) return;
    let key;
    if(mode==='monthly'){ key = s.start_time.slice(0,7); }
    else if(mode==='weekly'){
      const d = new Date(s.start_time);
      const tmp = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
      tmp.setUTCDate(tmp.getUTCDate() + 4 - (tmp.getUTCDay() || 7));
      const yearStart = new Date(Date.UTC(tmp.getUTCFullYear(), 0, 1));
      const weekNum = Math.ceil((((tmp - yearStart) / 86400000) + 1) / 7);
      key = tmp.getUTCFullYear() + '-W' + String(weekNum).padStart(2,'0');
    } else { key = fdate(s.start_time); }
    if(!key || key==='\u2014') return;
    if(!map[key]) map[key] = { label:key, cost:0, sessions:0 };
    map[key].cost += s.total_cost;
    map[key].sessions += 1;
  });
  return Object.values(map).sort((a,b)=>a.label.localeCompare(b.label));
}

/* ── Render functions ── */

function renderHeader(data){
  const totalCost = data.reduce((s,d)=>s+d.total_cost,0);
  const dates = data.map(s=>s.start_time).filter(Boolean).sort();
  const dateRange = dates.length ? fdate(dates[0]) + ' to ' + fdate(dates[dates.length-1]) : '';
  document.getElementById('total-spend').textContent = fc(totalCost);
  document.getElementById('date-range').textContent = dateRange;
  document.getElementById('last-refreshed').textContent = 'Last refreshed: ' + new Date().toLocaleString();
  document.getElementById('filter-info').textContent = data.length + ' of ' + DATA.length + ' sessions';
}

function renderSummaryCards(data){
  const totalCost = data.reduce((s,d)=>s+d.total_cost,0);
  const totalSessions = data.length;
  const totalMessages = data.reduce((s,d)=>s+(d.num_user_messages||0),0);
  const avgCost = totalSessions ? totalCost/totalSessions : 0;
  const today = new Date().toISOString().slice(0,10);
  const todaySpend = data.filter(s=>(s.start_time||'').startsWith(today)).reduce((s,d)=>s+d.total_cost,0);
  const monthlySub = totalMonthlySub();
  // Other AI from usage log for current month
  const now = new Date();
  const cm = now.getFullYear()+'-'+String(now.getMonth()+1).padStart(2,'0');
  const otherAI = USAGE_LOG.filter(u=>u.month===cm && !(u.service||'').includes('Claude')).reduce((s,u)=>s+(u.cost_base||0)+(u.cost_overage||0),0);
  const cmTokenCost = data.filter(s=>(s.start_time||'').startsWith(cm)).reduce((s,d)=>s+d.total_cost,0);
  const allIn = cmTokenCost + otherAI + monthlySub;

  document.getElementById('summary-cards').innerHTML = [
    { label:'All-In This Month', value:fc(allIn), cls:'cost-red', detail:'tokens + subs + other' },
    { label:'Claude Tokens', value:fc(totalCost), cls:'cost', detail:totalSessions+' sessions' },
    { label:'Other AI', value:fc(otherAI), cls:'cost-yellow', detail:'from usage_log.json' },
    { label:'Subscriptions', value:fc(monthlySub), cls:'cost-yellow', detail:SUBS.length+' services' },
    { label:"Today", value:fc(todaySpend), cls:todaySpend>10?'cost-red':todaySpend>3?'cost-yellow':'cost-green', detail:today },
    { label:'Avg/Session', value:fc(avgCost), cls:avgCost>10?'cost-red':avgCost>5?'cost-yellow':'cost-green', detail:totalSessions+' sessions' },
  ].map(c=>`<div class="strip-card"><div class="strip-label">${c.label}</div><div class="strip-value ${c.cls||''}">${c.value}</div><div class="strip-detail">${c.detail}</div></div>`).join('');
}

function renderBudgetBars(data){
  const el = document.getElementById('budget-bars');
  if(!BUDGETS || !Object.keys(BUDGETS).length){ el.style.display='none'; return; }
  el.style.display='';
  const now = new Date();
  const cm = now.getFullYear()+'-'+String(now.getMonth()+1).padStart(2,'0');
  const today = now.toISOString().slice(0,10);
  const dayOfMonth = now.getDate();
  const daysInMonth = new Date(now.getFullYear(), now.getMonth()+1, 0).getDate();
  const alertPct = BUDGETS.alert_threshold_pct || 80;

  const cmTokenCost = data.filter(s=>(s.start_time||'').startsWith(cm)).reduce((s,d)=>s+d.total_cost,0);
  const todayCost = data.filter(s=>(s.start_time||'').startsWith(today)).reduce((s,d)=>s+d.total_cost,0);
  const monthlySub = totalMonthlySub();
  const otherAI = USAGE_LOG.filter(u=>u.month===cm && !(u.service||'').includes('Claude')).reduce((s,u)=>s+(u.cost_base||0)+(u.cost_overage||0),0);
  const allIn = cmTokenCost + otherAI + monthlySub;

  const bars = [];

  if(BUDGETS.monthly_total){
    const pct = Math.min(100, (allIn/BUDGETS.monthly_total)*100);
    const color = pct>=100?'var(--red)':pct>=alertPct?'var(--yellow)':'var(--green)';
    const projected = (allIn / dayOfMonth) * daysInMonth;
    let alert = '';
    if(pct>=100) alert = 'OVER BUDGET';
    else if(pct>=alertPct) alert = 'Approaching limit';
    else if(projected > BUDGETS.monthly_total) alert = 'Projected: '+fc(projected)+' (over)';
    bars.push({label:'Monthly Total', spent:allIn, budget:BUDGETS.monthly_total, pct, color, alert});
  }
  if(BUDGETS.monthly_claude){
    const pct = Math.min(100, (cmTokenCost/BUDGETS.monthly_claude)*100);
    const color = pct>=100?'var(--red)':pct>=alertPct?'var(--yellow)':'var(--green)';
    let alert = '';
    if(pct>=100) alert = 'OVER BUDGET';
    else if(pct>=alertPct) alert = 'Approaching limit';
    bars.push({label:'Claude Tokens', spent:cmTokenCost, budget:BUDGETS.monthly_claude, pct, color, alert});
  }
  if(BUDGETS.daily_claude){
    const pct = Math.min(100, (todayCost/BUDGETS.daily_claude)*100);
    const color = pct>=100?'var(--red)':pct>=alertPct?'var(--yellow)':'var(--green)';
    let alert = '';
    if(pct>=100) alert = 'DAILY LIMIT HIT';
    else if(pct>=alertPct) alert = 'Approaching daily limit';
    bars.push({label:'Today (Claude)', spent:todayCost, budget:BUDGETS.daily_claude, pct, color, alert});
  }

  el.innerHTML = bars.map(b=>`<div class="budget-item"><div class="budget-label"><span>${b.label}</span><span>${fc(b.spent)} / ${fc(b.budget)} (${b.pct.toFixed(0)}%)</span></div><div class="budget-track"><div class="budget-fill" style="width:${b.pct}%;background:${b.color}"></div></div>${b.alert?'<div class="budget-alert">'+b.alert+'</div>':''}</div>`).join('');
}

function renderProjectTable(data){
  const projMap = {};
  data.forEach(s => {
    const name = projectLabel(s.project||'');
    if(!projMap[name]) projMap[name] = { cost:0, sessions:0, messages:0 };
    projMap[name].cost += s.total_cost;
    projMap[name].sessions += 1;
    projMap[name].messages += s.num_user_messages || 0;
  });
  const projects = Object.entries(projMap).map(([name,d])=>({name,...d})).sort((a,b)=>b.cost-a.cost);
  const maxCost = projects[0]?.cost || 1;
  const totalCost = data.reduce((s,d)=>s+d.total_cost,0) || 1;

  document.getElementById('project-table').innerHTML = projects.map(p=>{
    const pct = ((p.cost/maxCost)*100).toFixed(0);
    return `<tr><td><strong>${esc(p.name)}</strong></td><td class="right ${costColor(p.cost)}" style="font-weight:600">${fc(p.cost)}</td><td class="right">${p.sessions}</td><td class="right">${p.messages}</td><td class="bar-cell"><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${barColor(p.cost)}"></div><span class="bar-pct">${((p.cost/totalCost)*100).toFixed(0)}%</span></div></td></tr>`;
  }).join('');
}

function renderModelTable(data){
  const modelMap = {};
  data.forEach(s => {
    const cbm = s.cost_by_model || {};
    const tbm = s.tokens_by_model || {};
    for(const [model, cost] of Object.entries(cbm)){
      if(!modelMap[model]) modelMap[model] = { cost:0, input:0, output:0, cache_write:0, cache_read:0 };
      modelMap[model].cost += cost;
    }
    for(const [model, tok] of Object.entries(tbm)){
      if(!modelMap[model]) modelMap[model] = { cost:0, input:0, output:0, cache_write:0, cache_read:0 };
      modelMap[model].input += tok.input||0;
      modelMap[model].output += tok.output||0;
      modelMap[model].cache_write += tok.cache_write||0;
      modelMap[model].cache_read += tok.cache_read||0;
    }
  });
  const models = Object.entries(modelMap).map(([name,d])=>({name,...d})).sort((a,b)=>b.cost-a.cost);

  document.getElementById('model-table').innerHTML = models.map(m=>{
    return `<tr><td style="font-family:monospace;font-size:13px">${esc(m.name)}</td><td class="right ${costColor(m.cost)}" style="font-weight:600">${fc(m.cost)}</td><td class="right">${ft(m.input)}</td><td class="right">${ft(m.output)}</td><td class="right">${ft(m.cache_write)}</td><td class="right">${ft(m.cache_read)}</td></tr>`;
  }).join('');
}

function renderTimeline(data){
  const headers = { daily:'Date', weekly:'Week', monthly:'Month' };
  document.getElementById('timeline-col-header').textContent = headers[timelineMode] || 'Date';
  const rows = aggregateTimeline(data, timelineMode);
  const maxCost = Math.max(...rows.map(r=>r.cost), 1);
  document.getElementById('timeline-table').innerHTML = rows.map(r=>{
    const pct = ((r.cost/maxCost)*100).toFixed(0);
    const label = timelineMode==='monthly' ? fmonth(r.label) : r.label;
    return `<tr><td class="timestamp">${label}</td><td class="right ${costColor(r.cost)}" style="font-weight:600">${fc(r.cost)}</td><td class="right">${r.sessions}</td><td class="bar-cell"><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${barColor(r.cost)}"></div></div></td></tr>`;
  }).join('');
}

function renderSubscriptions(){
  const total = totalMonthlySub();
  document.getElementById('subscriptions-table').innerHTML = SUBS.map(sub=>{
    const cc = catClass(sub.category);
    return `<tr><td><strong>${esc(sub.name)}</strong><br><span style="font-size:12px;color:var(--dim)">${esc(sub.note||'')}</span></td><td class="right" style="font-weight:600">${fc(sub.cost)}</td><td><span class="cat-badge ${cc}">${esc(sub.category||'Other')}</span></td><td class="timestamp">${sub.start_date||'\u2014'}</td></tr>`;
  }).join('');
  document.getElementById('subscriptions-footer').innerHTML = `<tr><td colspan="1"><strong>Total Monthly</strong></td><td class="right cost-yellow" style="font-weight:700">${fc(total)}</td><td colspan="2"></td></tr>`;
}

function renderAllAISpend(data){
  const now = new Date();
  const currentMonth = now.getFullYear()+'-'+String(now.getMonth()+1).padStart(2,'0');

  // Claude token costs from session data
  const claudeTokenCost = data.reduce((s,d)=>s+d.total_cost,0);
  const claudeSessions = data.length;

  // Usage log entries for current month
  const monthUsage = USAGE_LOG.filter(u=>u.month===currentMonth);

  // Build rows: start with usage log entries, then add Claude from live data
  const rows = [];
  let hasClaudeInLog = false;

  monthUsage.forEach(u=>{
    if(u.service && u.service.includes('Claude')) hasClaudeInLog = true;
    const base = u.cost_base||0;
    const overage = u.cost_overage||0;
    const total = base + overage;
    const note = u.note||'';
    rows.push({ service:u.service, category:u.category||'Other', base, overage, total, note, source:'log' });
  });

  // Always add live Claude data (override log if present)
  if(!hasClaudeInLog){
    rows.unshift({ service:'Claude Code (live)', category:'LLM', base:30, overage:claudeTokenCost, total:30+claudeTokenCost, note:claudeSessions+' sessions this period (auto-calculated from logs)', source:'live' });
  } else {
    // Update the Claude entry with live data
    const idx = rows.findIndex(r=>r.service && r.service.includes('Claude'));
    if(idx>=0){
      rows[idx].overage = claudeTokenCost;
      rows[idx].total = rows[idx].base + claudeTokenCost;
      rows[idx].note = claudeSessions+' sessions (live data) — ' + (rows[idx].note||'');
    }
  }

  // Add subscription-only services not in usage log
  const loggedServices = new Set(rows.map(r=>(r.service||'').toLowerCase()));
  SUBS.forEach(sub=>{
    const name = sub.name||'';
    if(!loggedServices.has(name.toLowerCase()) && !rows.some(r=>(r.service||'').toLowerCase().includes(name.split(' ')[0].toLowerCase()))){
      if(sub.cost > 0){
        rows.push({ service:name, category:sub.category||'Other', base:sub.cost, overage:0, total:sub.cost, note:sub.note||'From subscriptions.json', source:'sub' });
      }
    }
  });

  rows.sort((a,b)=>b.total-a.total);
  const grandTotal = rows.reduce((s,r)=>s+r.total,0);

  document.getElementById('all-ai-table').innerHTML = rows.map(r=>{
    const cc = catClass(r.category);
    const sourceTag = r.source==='live' ? ' <span style="color:var(--green);font-size:11px">[LIVE]</span>' : r.source==='log' ? ' <span style="color:var(--dim);font-size:11px">[LOG]</span>' : '';
    return `<tr><td><strong>${esc(r.service)}</strong>${sourceTag}</td><td><span class="cat-badge ${cc}">${esc(r.category)}</span></td><td class="right">${fc(r.base)}</td><td class="right ${r.overage>100?'cost-red':r.overage>0?'cost-yellow':''}">${r.overage>0?fc(r.overage):'\u2014'}</td><td class="right" style="font-weight:700">${fc(r.total)}</td><td style="font-size:12px;color:var(--dim);max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.note)}</td></tr>`;
  }).join('');

  document.getElementById('all-ai-footer').innerHTML = `<tr><td colspan="4"><strong>Total AI Spend (${fmonth(currentMonth)})</strong></td><td class="right cost-red" style="font-weight:700;font-size:16px">${fc(grandTotal)}</td><td></td></tr>`;
}

function renderMonthlyReport(data){
  const allDates = data.map(s=>s.start_time).filter(Boolean).sort();
  if(allDates.length === 0){ document.getElementById('monthly-report-table').innerHTML=''; document.getElementById('monthly-report-footer').innerHTML=''; return; }

  const startMonth = allDates[0].slice(0,7);
  const now = new Date();
  const endMonth = now.getFullYear()+'-'+String(now.getMonth()+1).padStart(2,'0');

  const months = [];
  let [y,m] = startMonth.split('-').map(Number);
  while(true){
    const ym = y+'-'+String(m).padStart(2,'0');
    months.push(ym);
    if(ym >= endMonth) break;
    m++; if(m>12){ m=1; y++; }
  }

  // Claude token costs per month
  const tokenByMonth = {};
  data.forEach(s=>{
    if(!s.start_time) return;
    const ym = s.start_time.slice(0,7);
    if(!tokenByMonth[ym]) tokenByMonth[ym] = { cost:0, sessions:0 };
    tokenByMonth[ym].cost += s.total_cost;
    tokenByMonth[ym].sessions += 1;
  });

  // Other AI costs from usage log (non-Claude entries)
  const otherByMonth = {};
  USAGE_LOG.forEach(u=>{
    if(u.service && u.service.includes('Claude')) return; // skip Claude, we have live data
    const ym = u.month;
    if(!otherByMonth[ym]) otherByMonth[ym] = 0;
    otherByMonth[ym] += (u.cost_base||0) + (u.cost_overage||0);
  });

  const rows = months.map(ym=>{
    const tok = tokenByMonth[ym] || { cost:0, sessions:0 };
    const other = otherByMonth[ym] || 0;
    const sub = subsActiveInMonth(ym);
    return { month:ym, tokenCost:tok.cost, otherAI:other, subCost:sub, total:tok.cost+other+sub, sessions:tok.sessions };
  });

  // Projection
  const currentMonth = endMonth;
  const completed = rows.filter(r=>r.month < currentMonth && (r.tokenCost > 0 || r.otherAI > 0));
  let projNextTotal = null;
  let nextMonth = null;
  if(completed.length >= 2){
    const [ny,nm] = currentMonth.split('-').map(Number);
    const nm2 = nm===12 ? 1 : nm+1;
    const ny2 = nm===12 ? ny+1 : ny;
    nextMonth = ny2+'-'+String(nm2).padStart(2,'0');
    const last = completed.slice(-3);
    const avgTotal = last.reduce((s,r)=>s+r.total,0) / last.length;
    const slope = last.length>=2 ? (last[last.length-1].total - last[0].total)/(last.length-1) : 0;
    projNextTotal = Math.max(0, avgTotal + slope);
  }

  const maxTotal = Math.max(...rows.map(r=>r.total), projNextTotal||0, 1);

  let html = rows.map(r=>{
    const pct = ((r.total/maxTotal)*100).toFixed(0);
    const isCurrent = r.month === currentMonth;
    const label = fmonth(r.month) + (isCurrent ? ' (current)' : '');
    return `<tr><td class="timestamp">${label}</td><td class="right ${costColor(r.tokenCost)}" style="font-weight:600">${fc(r.tokenCost)}</td><td class="right" style="color:var(--purple)">${r.otherAI>0?fc(r.otherAI):'\u2014'}</td><td class="right" style="color:var(--yellow)">${fc(r.subCost)}</td><td class="right" style="font-weight:700">${fc(r.total)}</td><td class="bar-cell"><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:${r.total>500?'var(--red)':r.total>200?'var(--yellow)':'var(--green)'}"></div></div></td></tr>`;
  }).join('');

  // Projection row
  if(nextMonth && projNextTotal !== null){
    const pct = ((projNextTotal/maxTotal)*100).toFixed(0);
    html += `<tr class="projected"><td class="timestamp">${fmonth(nextMonth)} (projected)</td><td class="right" style="color:var(--dim)">\u2014</td><td class="right" style="color:var(--dim)">\u2014</td><td class="right" style="color:var(--dim)">\u2014</td><td class="right" style="font-weight:600;color:var(--purple)">${fc(projNextTotal)}</td><td class="bar-cell"><div class="bar-wrap"><div class="bar" style="width:${pct}%;background:var(--purple);opacity:0.5"></div></div></td></tr>`;
  }

  document.getElementById('monthly-report-table').innerHTML = html;

  // Footer totals
  const grandTokens = rows.reduce((s,r)=>s+r.tokenCost,0);
  const grandOther = rows.reduce((s,r)=>s+r.otherAI,0);
  const grandSubs = rows.reduce((s,r)=>s+r.subCost,0);
  const grandTotal = rows.reduce((s,r)=>s+r.total,0);
  const grandSessions = rows.reduce((s,r)=>s+r.sessions,0);
  document.getElementById('monthly-report-footer').innerHTML = `<tr><td><strong>Grand Total</strong></td><td class="right cost-green" style="font-weight:700">${fc(grandTokens)}</td><td class="right" style="font-weight:700;color:var(--purple)">${grandOther>0?fc(grandOther):'\u2014'}</td><td class="right cost-yellow" style="font-weight:700">${fc(grandSubs)}</td><td class="right" style="font-weight:700">${fc(grandTotal)}</td><td></td></tr>`;
}

function renderInsights(data){
  const recs = [];
  const totalCost = data.reduce((s,d)=>s+d.total_cost,0);
  const totalIn = data.reduce((s,d)=>s+d.total_input_tokens,0);
  const totalOut = data.reduce((s,d)=>s+d.total_output_tokens,0);
  const totalCW = data.reduce((s,d)=>s+d.total_cache_write_tokens,0);
  const totalCR = data.reduce((s,d)=>s+d.total_cache_read_tokens,0);
  const totalTools = data.reduce((s,d)=>s+d.num_tool_calls,0);
  const totalErrors = data.reduce((s,d)=>s+(d.tool_errors||[]).length,0);
  const totalSessions = data.length;

  /* ════════════════════════════════════════════════════════════════════
     HIGH SEVERITY — biggest cost drivers
     ════════════════════════════════════════════════════════════════════ */

  // 1. Duplicate tool calls — wasted round-trips
  const dupMap = {};
  data.forEach(s=>{ (s.duplicate_tools||[]).forEach(d=>{ if(!dupMap[d.tool]) dupMap[d.tool]=0; dupMap[d.tool]+=d.count; }); });
  const dupTotal = Object.values(dupMap).reduce((s,c)=>s+c,0);
  if(dupTotal > 0){
    const estWaste = dupTotal * 0.02; // ~$0.02 per duplicate round-trip (conservative)
    Object.entries(dupMap).sort((a,b)=>b[1]-a[1]).slice(0,3).forEach(([tool,count])=>{
      recs.push({ severity:'high', title:'Repeated '+tool+' calls (<span class="rec-stat">'+count+'x</span> identical)', detail:'Same input sent multiple times. Each duplicate wastes an API round-trip. <strong>Fix:</strong> Provide clearer instructions so Claude doesn\'t re-read the same files. Use CLAUDE.md to document project structure.' });
    });
  }

  // 2. Error rate — retries double cost
  if(totalTools>10 && totalErrors/totalTools>0.08){
    const errCostEst = totalErrors * 0.03;
    recs.push({ severity:'high', title:'High tool error rate: <span class="rec-stat">'+(100*totalErrors/totalTools).toFixed(0)+'%</span> ('+totalErrors+'/'+totalTools+') ~<span class="rec-stat">'+fc(errCostEst)+'</span> wasted', detail:'Each error forces a retry, roughly doubling token spend. <strong>Fix:</strong> Check common errors \u2014 bad file paths, permission denials, missing dependencies. Add a CLAUDE.md with correct paths.' });
  }

  // 3. Output token dominance — output costs 5x input on Opus
  const avgOutputPct = totalSessions > 0 ? data.reduce((s,d)=>s+(d.output_cost_pct||0),0) / totalSessions : 0;
  if(avgOutputPct > 40 && totalCost > 5){
    const outCostEst = totalCost * (avgOutputPct/100);
    recs.push({ severity:'high', title:'Output tokens dominate cost: <span class="rec-stat">'+avgOutputPct.toFixed(0)+'%</span> of spend (~<span class="rec-stat">'+fc(outCostEst)+'</span>)', detail:'Output tokens cost 5x input on Opus ($25/M vs $5/M). <strong>Fix:</strong> Ask for concise responses, skip trailing summaries, request "no explanation" for code-only tasks. Use <code>/fast</code> mode for exploratory work.' });
  }

  // 4. Monthly burn rate projection
  const now = new Date();
  const cm = now.getFullYear()+'-'+String(now.getMonth()+1).padStart(2,'0');
  const cmData = data.filter(s=>(s.start_time||'').slice(0,7)===cm);
  if(cmData.length > 3){
    const cmCost = cmData.reduce((s,d)=>s+d.total_cost,0);
    const dayOfMonth = now.getDate();
    const daysInMonth = new Date(now.getFullYear(), now.getMonth()+1, 0).getDate();
    const projected = (cmCost / dayOfMonth) * daysInMonth;
    if(projected > cmCost * 1.2){
      recs.push({ severity: projected>500?'high':'medium', title:'Month pace: <span class="rec-stat">'+fc(cmCost)+'</span> spent \u2192 <span class="rec-stat">'+fc(projected)+'</span> projected for '+fmonth(cm), detail:'Based on '+dayOfMonth+' days of data ('+cmData.length+' sessions). '+(projected>1000?'On track for a very high month. ':'')+'<strong>Fix:</strong> Set a daily budget target of <span class="rec-stat">'+fc(projected/daysInMonth)+'</span>/day and track it.' });
    }
  }

  /* ════════════════════════════════════════════════════════════════════
     MEDIUM SEVERITY — tool usage anti-patterns
     ════════════════════════════════════════════════════════════════════ */

  // 5. Bash used instead of dedicated tools
  const bashGrepTotal = data.reduce((s,d)=>s+(d.bash_grep_count||0),0);
  if(bashGrepTotal > 5){
    recs.push({ severity:'medium', title:'Bash used for search/read <span class="rec-stat">'+bashGrepTotal+'x</span> instead of dedicated tools', detail:'Bash grep/cat/find outputs flood the context. Dedicated tools (Grep, Read, Glob) return structured, trimmed results. <strong>Fix:</strong> Add to CLAUDE.md: "Always use Grep instead of bash grep, Read instead of cat, Glob instead of find."' });
  }

  // 6. Reading full large files
  const readLargeTotal = data.reduce((s,d)=>s+(d.read_large_count||0),0);
  if(readLargeTotal > 3){
    recs.push({ severity:'medium', title:'Full-file reads on large files: <span class="rec-stat">'+readLargeTotal+'x</span>', detail:'Reading entire large files dumps thousands of tokens into context. <strong>Fix:</strong> Use <code>offset</code> and <code>limit</code> parameters on Read, or Grep to find the relevant section first. Add to CLAUDE.md: "For files >200 lines, read only the relevant section."' });
  }

  // 7. Many edits to same file — could consolidate
  const editTargets = {};
  data.forEach(s=>{ const et = s.edit_file_targets||{}; Object.entries(et).forEach(([f,c])=>{ if(!editTargets[f]) editTargets[f]=0; editTargets[f]+=c; }); });
  const multiEditFiles = Object.entries(editTargets).filter(([,c])=>c>5).sort((a,b)=>b[1]-a[1]);
  if(multiEditFiles.length > 0){
    const top = multiEditFiles.slice(0,2);
    const names = top.map(([f,c])=>f.split(/[/\\]/).pop()+' ('+c+'x)').join(', ');
    recs.push({ severity:'medium', title:'Frequent re-edits: <span class="rec-stat">'+names+'</span>', detail:'Multiple small edits to the same file consume tokens per round-trip. <strong>Fix:</strong> Describe the full change upfront so Claude can make it in fewer passes. For large rewrites, ask Claude to use Write instead of incremental Edits.' });
  }

  // 8. Large tool results inflating context
  const bigResults = [];
  data.forEach(s=>{ (s.large_results||[]).forEach(r=>bigResults.push(r)); });
  bigResults.sort((a,b)=>b.result_size-a.result_size).slice(0,2).forEach(r=>{
    recs.push({ severity:'medium', title:'Large '+r.name+' result: <span class="rec-stat">'+(r.result_size||0).toLocaleString()+' chars</span>', detail:'Oversized results get re-sent in context on every subsequent turn. <strong>Fix:</strong> For Read, use offset/limit. For Bash, pipe through <code>| head -50</code>. For Grep, use <code>head_limit</code> parameter.' });
  });

  // 9. Cost per message outliers by project
  const projMap = {};
  data.forEach(s=>{
    const name = projectLabel(s.project||'');
    if(!projMap[name]) projMap[name] = { cost:0, messages:0 };
    projMap[name].cost += s.total_cost;
    projMap[name].messages += s.num_user_messages || 0;
  });
  const projArr = Object.entries(projMap).filter(([,d])=>d.messages>3).map(([name,d])=>({name, cpm:d.cost/d.messages, ...d}));
  if(projArr.length > 2){
    const avgCPM = projArr.reduce((s,p)=>s+p.cpm,0)/projArr.length;
    const expensive = projArr.filter(p=>p.cpm > avgCPM*2).sort((a,b)=>b.cpm-a.cpm);
    expensive.slice(0,2).forEach(p=>{
      recs.push({ severity:'medium', title:esc(p.name)+': <span class="rec-stat">'+fc(p.cpm)+'/msg</span> vs <span class="rec-stat">'+fc(avgCPM)+'</span> avg', detail:'This project costs '+(p.cpm/avgCPM).toFixed(1)+'x average per message. <strong>Fix:</strong> Add a detailed CLAUDE.md to this project with file structure, key paths, and conventions to reduce exploration overhead.' });
    });
  }

  // 10. High cost sessions — break up big tasks
  const highCost = data.filter(s=>s.total_cost>10);
  if(highCost.length>0){
    recs.push({ severity:'medium', title:highCost.length+' session'+(highCost.length>1?'s':'')+' over $10 (<span class="rec-stat">'+fc(highCost.reduce((s,d)=>s+d.total_cost,0))+'</span> total)', detail:'Long-running sessions accumulate context. <strong>Fix:</strong> Break complex tasks into focused sub-tasks. Start fresh sessions for distinct features. Use <code>/clear</code> to reset context mid-session.' });
  }

  // 11. Long session duration
  const longSessions = data.filter(s=>(s.session_duration_min||0)>120);
  if(longSessions.length > 2){
    const avgDur = longSessions.reduce((s,d)=>s+(d.session_duration_min||0),0) / longSessions.length;
    recs.push({ severity:'medium', title:longSessions.length+' sessions over 2 hours (avg <span class="rec-stat">'+avgDur.toFixed(0)+' min</span>)', detail:'Context grows with every message. After ~30 turns, you\'re paying mostly for cache writes. <strong>Fix:</strong> Start fresh sessions for new tasks. Summarize progress in CLAUDE.md before ending a session so the next one starts informed.' });
  }

  /* ════════════════════════════════════════════════════════════════════
     INFO — efficiency metrics + structural insights
     ════════════════════════════════════════════════════════════════════ */

  // 12. Cache efficiency
  if(totalCW > 0){
    const ratio = totalCR / totalCW;
    const cwCostEst = (totalCW/1e6)*10;
    const crSavings = (totalCR/1e6)*4.5; // cache reads cost $0.50/M vs $5/M input = $4.50 saved per M
    recs.push({ severity:'info', title:'Cache ratio: <span class="rec-stat">'+ratio.toFixed(1)+'x</span> read/write \u2014 saved ~<span class="rec-stat">'+fc(crSavings)+'</span>', detail:'Cache reads cost 90% less than fresh input. '+(ratio<5?'Low ratio \u2014 sessions may be too short to amortize writes. Longer, focused sessions improve cache reuse.':'Healthy cache reuse!') });
  }

  // 13. Model mix analysis
  const modelCosts = {};
  data.forEach(s=>{ Object.entries(s.cost_by_model||{}).forEach(([m,c])=>{ if(!modelCosts[m]) modelCosts[m]=0; modelCosts[m]+=c; }); });
  const opusCost = Object.entries(modelCosts).filter(([m])=>m.includes('opus')).reduce((s,[,c])=>s+c,0);
  const sonnetCost = Object.entries(modelCosts).filter(([m])=>m.includes('sonnet')).reduce((s,[,c])=>s+c,0);
  const haikuCost = Object.entries(modelCosts).filter(([m])=>m.includes('haiku')).reduce((s,[,c])=>s+c,0);
  if(opusCost > 0 && totalCost > 10){
    const opusPct = (opusCost/totalCost*100).toFixed(0);
    const potentialSavings = opusCost * 0.40; // Sonnet is ~60% of Opus cost
    recs.push({ severity:'info', title:'Model mix: <span class="rec-stat">'+opusPct+'%</span> Opus (<span class="rec-stat">'+fc(opusCost)+'</span>)', detail:'Opus is the most capable but costs ~1.7x Sonnet. <strong>Savings tip:</strong> Use Sonnet (<code>claude-sonnet-4-6</code>) for exploration, debugging, and file reads \u2014 potential savings of ~<span class="rec-stat">'+fc(potentialSavings)+'</span>. Toggle with <code>/model sonnet</code> or use <code>/fast</code> for faster, equally-capable output.' });
  }

  // 14. Subscription vs token split
  const monthlySub = totalMonthlySub();
  if(monthlySub > 0 && totalCost > 0){
    const months = new Set();
    data.forEach(s=>{ if(s.start_time) months.add(s.start_time.slice(0,7)); });
    const numMonths = months.size || 1;
    const avgMonthlyTokens = totalCost / numMonths;
    const subPct = (monthlySub / (monthlySub + avgMonthlyTokens) * 100).toFixed(0);
    recs.push({ severity:'info', title:'Monthly split: <span class="rec-stat">'+fc(monthlySub)+'</span> fixed + <span class="rec-stat">'+fc(avgMonthlyTokens)+'</span> variable', detail:'Subscriptions are '+subPct+'% of average monthly AI spend. '+(avgMonthlyTokens > monthlySub*3 ? 'Token costs dominate \u2014 the recommendations above can save the most.' : 'Balanced split.') });
  }

  // 15. Total tool result size — context bloat indicator
  const totalResultChars = data.reduce((s,d)=>s+(d.tool_result_total_chars||0),0);
  if(totalResultChars > 5_000_000){
    const resultMB = (totalResultChars/1_000_000).toFixed(1);
    recs.push({ severity:'info', title:'Total tool output: <span class="rec-stat">'+resultMB+'M chars</span> across all sessions', detail:'Large tool results are the #1 context inflator. Each char becomes ~0.3 tokens re-sent on subsequent turns. Trimming tool results is the highest-leverage optimization.' });
  }

  /* ════════════════════════════════════════════════════════════════════
     Render
     ════════════════════════════════════════════════════════════════════ */
  const sevOrder = {high:0, medium:1, info:2, low:3};
  recs.sort((a,b)=>(sevOrder[a.severity]||9)-(sevOrder[b.severity]||9));

  document.getElementById('recommendations').innerHTML = recs.length
    ? recs.map(r=>`<div class="rec-item ${r.severity}"><div class="rec-title">${r.title}</div><div class="rec-detail">${r.detail}</div></div>`).join('')
    : '<div class="no-recs">Looking good! No optimization issues found.</div>';
}

function renderSessionTable(data){
  const sorted = [...data].sort((a,b)=>b.total_cost-a.total_cost);
  document.getElementById('sessions-table').innerHTML = sorted.map(s=>{
    const errs = (s.tool_errors||[]).length;
    const models = (s.models||[]).join(', ') || '?';
    return `<tr><td>${esc(projectLabel(s.project||''))}</td><td class="timestamp">${ftime(s.start_time)}</td><td class="right ${costColor(s.total_cost)}" style="font-weight:600">${fc(s.total_cost)}</td><td style="font-size:13px;color:var(--dim)">${esc(models)}</td><td class="right">${s.num_user_messages||0}</td><td class="right">${s.num_tool_calls}</td><td class="right" style="color:${errs?'var(--red)':'var(--dim)'}">${errs}</td></tr>`;
  }).join('');
}

/* ── Usage log table ── */
function renderUsageLog(){
  document.getElementById('usage-log-table').innerHTML = USAGE_LOG.map(u=>{
    return `<tr><td class="timestamp">${u.month?fmonth(u.month):'\u2014'}</td><td>${u.service||''}</td><td class="right">${fc(u.cost_base||0)}</td><td class="right ${(u.cost_overage||0)>0?'cost-red':''}">${(u.cost_overage||0)>0?fc(u.cost_overage):'\u2014'}</td><td style="font-size:11px;color:var(--dim)">${u.note||''}</td></tr>`;
  }).join('') || '<tr><td colspan="5" style="color:var(--dim);text-align:center">No entries yet. Add to usage_log.json.</td></tr>';
}

/* ── Tab switching ── */
function switchTab(name){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  const panel = document.getElementById('panel-'+name);
  if(panel) panel.classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(t=>{
    if(t.textContent.toLowerCase().includes(name.split('-')[0]) || t.getAttribute('onclick')?.includes(name)) t.classList.add('active');
  });
  // More reliable: match by onclick
  document.querySelectorAll('.nav-tab').forEach(t=>{
    const oc = t.getAttribute('onclick')||'';
    t.classList.toggle('active', oc.includes("'"+name+"'"));
  });
}

/* ── Savings Guide — static tips dynamically scored ── */
function renderSavingsGuide(data){
  const totalCost = data.reduce((s,d)=>s+d.total_cost,0);
  const totalOut = data.reduce((s,d)=>s+d.total_output_tokens,0);
  const totalIn = data.reduce((s,d)=>s+d.total_input_tokens,0);
  const totalCW = data.reduce((s,d)=>s+d.total_cache_write_tokens,0);
  const totalCR = data.reduce((s,d)=>s+d.total_cache_read_tokens,0);
  const totalTools = data.reduce((s,d)=>s+d.num_tool_calls,0);
  const totalErrors = data.reduce((s,d)=>s+(d.tool_errors||[]).length,0);
  const totalDups = data.reduce((s,d)=>s+(d.duplicate_tools||[]).reduce((a,x)=>a+x.count,0),0);
  const totalBashGrep = data.reduce((s,d)=>s+(d.bash_grep_count||0),0);
  const totalReadLarge = data.reduce((s,d)=>s+(d.read_large_count||0),0);
  const avgOutputPct = data.length>0 ? data.reduce((s,d)=>s+(d.output_cost_pct||0),0)/data.length : 0;
  const longSessions = data.filter(s=>(s.session_duration_min||0)>90).length;
  const totalSessions = data.length;
  const avgCostPerSession = totalSessions>0 ? totalCost/totalSessions : 0;

  // Model mix
  const modelCosts = {};
  data.forEach(s=>{ Object.entries(s.cost_by_model||{}).forEach(([m,c])=>{ if(!modelCosts[m]) modelCosts[m]=0; modelCosts[m]+=c; }); });
  const opusCost = Object.entries(modelCosts).filter(([m])=>m.includes('opus')).reduce((s,[,c])=>s+c,0);
  const opusPct = totalCost>0 ? opusCost/totalCost : 0;

  // Edit consolidation
  const editTargets = {};
  data.forEach(s=>{ Object.entries(s.edit_file_targets||{}).forEach(([f,c])=>{ if(!editTargets[f]) editTargets[f]=0; editTargets[f]+=c; }); });
  const multiEdits = Object.values(editTargets).filter(c=>c>5).length;

  const tips = [
    {
      id: 'concise-output',
      title: 'Request Concise Responses',
      how: 'Output tokens cost <strong>5x more</strong> than input on Opus ($25/M vs $5/M). Tell Claude to be brief: <code>"Be concise, skip explanations"</code>, <code>"Code only, no comments"</code>, <code>"Answer in under 50 words"</code>. Avoid asking "explain" unless you need it. Use <code>/fast</code> for exploratory work.',
      score: avgOutputPct > 50 ? 95 : avgOutputPct > 35 ? 70 : avgOutputPct > 20 ? 40 : 10,
      savings: totalCost > 5 ? totalCost * (avgOutputPct/100) * 0.3 : 0,
    },
    {
      id: 'claude-md',
      title: 'Maintain a CLAUDE.md File',
      how: 'A well-structured CLAUDE.md at the project root eliminates exploration overhead. Include: <strong>file structure</strong>, <strong>key entry points</strong>, <strong>build/test commands</strong>, <strong>naming conventions</strong>, and <strong>common pitfalls</strong>. This alone can cut 20-40% of tool calls per session by reducing aimless file reads and greps.',
      score: totalDups > 10 ? 90 : avgCostPerSession > 15 ? 80 : avgCostPerSession > 5 ? 60 : 30,
      savings: totalCost * 0.15,
    },
    {
      id: 'model-selection',
      title: 'Use the Right Model for the Task',
      how: 'Opus ($5/$25 per M) is best for complex architecture and multi-step implementation. <strong>Sonnet</strong> ($3/$15 per M) handles most coding tasks well. Use <code>/model sonnet</code> for debugging, exploration, and simple edits. Use <code>/model opus</code> for complex features. <code>/fast</code> mode uses the same model with faster output for iterative work.',
      score: opusPct > 0.9 && totalCost > 20 ? 85 : opusPct > 0.7 ? 60 : 20,
      savings: opusCost * 0.35,
    },
    {
      id: 'session-hygiene',
      title: 'Keep Sessions Short and Focused',
      how: 'Context accumulates with every message. After ~30 turns, you\'re paying mostly for cache writes (2x input cost). <strong>Start a new session</strong> for each distinct task. Use <code>/clear</code> to reset context mid-session. Before ending a long session, save progress in CLAUDE.md so the next session starts informed, not from scratch.',
      score: longSessions > 5 ? 90 : longSessions > 2 ? 65 : avgCostPerSession > 20 ? 70 : 25,
      savings: longSessions > 0 ? longSessions * avgCostPerSession * 0.25 : 0,
    },
    {
      id: 'trim-tool-results',
      title: 'Trim Large Tool Results',
      how: 'Every tool result is re-sent as context on subsequent turns. For <strong>Read</strong>: use <code>offset</code> and <code>limit</code> params (e.g., "read lines 50-100"). For <strong>Bash</strong>: pipe through <code>| head -50</code> or <code>| tail -20</code>. For <strong>Grep</strong>: use <code>head_limit</code> parameter. Add to CLAUDE.md: "For files >200 lines, read only the relevant section."',
      score: totalReadLarge > 10 ? 90 : totalReadLarge > 3 ? 65 : 20,
      savings: totalReadLarge * 0.05,
    },
    {
      id: 'dedicated-tools',
      title: 'Use Dedicated Tools, Not Bash',
      how: 'Bash <code>grep</code>/<code>cat</code>/<code>find</code> produce raw, unstructured output that floods context. Dedicated tools (Grep, Read, Glob) return trimmed, structured results. Add to CLAUDE.md: <code>"Always use Grep instead of bash grep, Read instead of cat, Glob instead of find."</code>',
      score: totalBashGrep > 20 ? 85 : totalBashGrep > 5 ? 55 : 5,
      savings: totalBashGrep * 0.02,
    },
    {
      id: 'reduce-errors',
      title: 'Reduce Tool Errors',
      how: 'Each tool error triggers a retry, roughly <strong>doubling</strong> the token cost for that step. Common causes: wrong file paths, missing dependencies, permission issues. <strong>Fix:</strong> Document correct paths in CLAUDE.md. Run <code>npm install</code>/<code>pip install</code> before starting. Provide explicit paths instead of letting Claude guess.',
      score: totalTools > 10 && totalErrors/totalTools > 0.1 ? 85 : totalErrors > 5 ? 50 : 5,
      savings: totalErrors * 0.04,
    },
    {
      id: 'avoid-duplicates',
      title: 'Eliminate Duplicate Tool Calls',
      how: 'Identical tool calls waste full round-trips. Common pattern: Claude reads the same file multiple times because it forgot the contents after context compression. <strong>Fix:</strong> For large tasks, provide a summary of what Claude has already read. Use CLAUDE.md to list key files so Claude reads them once and right.',
      score: totalDups > 20 ? 85 : totalDups > 5 ? 55 : 5,
      savings: totalDups * 0.02,
    },
    {
      id: 'batch-instructions',
      title: 'Batch Instructions in One Message',
      how: 'Each user message triggers a new API call. Instead of: <code>"read file A"</code> then <code>"now edit it"</code> \u2014 say: <code>"Read file A, then change X to Y on line 50"</code>. Fewer turns = fewer round-trips = less spend. Give all requirements upfront when possible.',
      score: totalSessions > 5 && avgCostPerSession > 10 ? 70 : avgCostPerSession > 5 ? 45 : 15,
      savings: totalCost * 0.08,
    },
    {
      id: 'consolidate-edits',
      title: 'Consolidate Multiple Small Edits',
      how: 'Many small edits to the same file cost a round-trip each. <strong>Fix:</strong> Describe the full change upfront: <code>"Change all X to Y, update the import, and fix the return type"</code>. For large rewrites, ask Claude to use Write (full file replacement) instead of many incremental Edits.',
      score: multiEdits > 3 ? 75 : multiEdits > 0 ? 40 : 5,
      savings: multiEdits * avgCostPerSession * 0.05,
    },
    {
      id: 'cache-reuse',
      title: 'Maximize Cache Reads',
      how: 'Cache reads cost <strong>90% less</strong> than fresh input ($0.50/M vs $5/M on Opus). Caches last 5 min (ephemeral) or 1 hour. <strong>Maximize reuse:</strong> Keep conversations focused on one topic. Avoid switching between unrelated projects mid-session. Prompt caching is automatic \u2014 your job is to keep the conversation context stable.',
      score: totalCW > 0 && totalCR/totalCW < 5 ? 60 : totalCW > 0 && totalCR/totalCW < 10 ? 35 : 10,
      savings: totalCW > 0 ? (totalCW/1e6) * 4.5 * 0.2 : 0,
    },
  ];

  // Score and sort
  tips.sort((a,b) => b.score - a.score);

  const html = tips.map(tip => {
    let scoreLabel, scoreClass;
    if(tip.score >= 80){ scoreLabel='Critical'; scoreClass='critical'; }
    else if(tip.score >= 55){ scoreLabel='High'; scoreClass='high-rel'; }
    else if(tip.score >= 25){ scoreLabel='Moderate'; scoreClass='moderate'; }
    else if(tip.score >= 10){ scoreLabel='Low'; scoreClass='low-rel'; }
    else { scoreLabel='Applied'; scoreClass='applied'; }

    const savingsHtml = tip.savings > 0.50 ? '<div class="guide-savings">Potential savings: ~'+fc(tip.savings)+'</div>' : '';

    return '<div class="guide-item"><div class="guide-score '+scoreClass+'">'+scoreLabel+'<br><span style="font-size:18px">'+tip.score+'</span></div><div class="guide-body"><div class="guide-title">'+tip.title+'</div><div class="guide-how">'+tip.how+'</div>'+savingsHtml+'</div></div>';
  }).join('');

  document.getElementById('savings-guide').innerHTML = html;
}

/* ── Master render ── */
function renderAll(){
  const data = filteredDATA();
  renderHeader(data);
  renderSummaryCards(data);
  renderBudgetBars(data);
  renderAllAISpend(data);
  renderMonthlyReport(data);
  renderProjectTable(data);
  renderModelTable(data);
  renderTimeline(data);
  renderSubscriptions();
  renderUsageLog();
  renderInsights(data);
  renderSessionTable(data);
  renderSavingsGuide(data);
}

function setTimelineMode(mode){
  timelineMode = mode;
  document.querySelectorAll('#timeline-tabs .sub-tab').forEach(btn=>{
    btn.classList.toggle('active', btn.textContent.toLowerCase() === mode);
  });
  renderTimeline(filteredDATA());
}

/* ── Initialize date filter ── */
(function(){
  const dates = DATA.map(s=>s.start_time).filter(Boolean).sort();
  const startEl = document.getElementById('filter-start');
  const endEl = document.getElementById('filter-end');
  if(dates.length){
    startEl.value = fdate(dates[0]);
    endEl.value = fdate(dates[dates.length-1]);
    filterStart = startEl.value;
    filterEnd = endEl.value;
  }
  startEl.addEventListener('change', ()=>{ filterStart = startEl.value; renderAll(); });
  endEl.addEventListener('change', ()=>{ filterEnd = endEl.value; renderAll(); });
  document.getElementById('filter-reset').addEventListener('click', ()=>{
    filterStart = dates.length ? fdate(dates[0]) : null;
    filterEnd = dates.length ? fdate(dates[dates.length-1]) : null;
    startEl.value = filterStart||'';
    endEl.value = filterEnd||'';
    renderAll();
  });
})();

/* ── Initial render ── */
renderAll();
</script>
</body>
</html>'''


def _build_html(analyses, watch_mode=False):
    """Build the HTML string with data injected."""
    serialized = _serialize_analyses(analyses)
    data_json = json.dumps(serialized, default=str)
    subs_json = json.dumps(_load_subscriptions(), default=str)
    usage_json = json.dumps(_load_usage_log(), default=str)
    html = _HTML_TEMPLATE.replace("%%DATA%%", data_json)
    html = html.replace("%%SUBS%%", subs_json)
    html = html.replace("%%USAGE%%", usage_json)
    budgets_json = json.dumps(_load_budgets(), default=str)
    html = html.replace("%%BUDGETS%%", budgets_json)
    if watch_mode:
        # Inject live-refresh script: fetches /data every 30s, re-renders all sections
        refresh_js = """
<script>
(function(){
  const INTERVAL = 30000;
  async function refresh(){
    try {
      const resp = await fetch('/data');
      if (!resp.ok) return;
      const fresh = await resp.json();
      DATA.length = 0;
      fresh.forEach(d => DATA.push(d));
      renderAll();
    } catch(e) { console.warn('refresh failed', e); }
  }
  setInterval(refresh, INTERVAL);
})();
</script>"""
        html = html.replace("</body>", refresh_js + "\n</body>")
    return html


def _publish_html(analyses):
    """Generate docs/index.html for GitHub Pages deployment."""
    html = _build_html(analyses, watch_mode=False)
    # Find the repo root (where .git lives)
    repo_root = Path(__file__).resolve().parent
    while repo_root != repo_root.parent:
        if (repo_root / ".git").exists():
            break
        repo_root = repo_root.parent
    if not (repo_root / ".git").exists():
        print(red("Error: no git repository found. Run from inside your repo."), file=sys.stderr)
        sys.exit(1)
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Published dashboard to {out_path}", file=sys.stderr)
    print(f"Commit & push, then enable GitHub Pages (source: /docs on main).", file=sys.stderr)


def _generate_html(analyses):
    """Generate a static HTML dashboard, write to temp file, and open in browser."""
    import tempfile
    import webbrowser

    html = _build_html(analyses, watch_mode=False)
    fd, path = tempfile.mkstemp(suffix=".html", prefix="token_tracker_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(html)

    url = "file:///" + path.replace("\\", "/")
    webbrowser.open(url)
    print(f"Report opened in browser: {path}", file=sys.stderr)


def _run_watch_server(scanner, project_dir=None, session_id=None, port=7865):
    """Start a local HTTP server that serves a live-updating dashboard."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import webbrowser
    import threading

    def _scan():
        """Re-scan and analyze sessions, return analyses list."""
        if session_id:
            fp = scanner.find_session(session_id)
            if not fp:
                return []
            return [analyze_session(parse_session(fp))]
        sessions = scanner.list_sessions(project_dir)
        out = []
        for fp in sessions:
            try:
                d = parse_session(fp)
                a = analyze_session(d)
                if a["num_api_calls"] > 0:
                    out.append(a)
            except Exception:
                pass
        return out

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/data":
                # JSON endpoint — re-scan every time
                analyses = _scan()
                payload = json.dumps(_serialize_analyses(analyses), default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            else:
                # Serve the dashboard HTML (initial load)
                analyses = _scan()
                html = _build_html(analyses, watch_mode=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)

        def log_message(self, fmt, *a):
            pass  # suppress request logs

    # Find an available port
    server = None
    for p in range(port, port + 20):
        try:
            server = HTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue

    if not server:
        print(red("Error: could not find an open port."), file=sys.stderr)
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    print(f"Live dashboard at {bold(cyan(url))}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        server.server_close()


def _serialize_analyses(analyses):
    """Prepare analyses list for JSON serialization."""
    out = []
    for a in analyses:
        item = dict(a)
        item["cost_by_model"] = dict(item["cost_by_model"])
        item["tokens_by_model"] = {k: dict(v) for k, v in item["tokens_by_model"].items()}
        item["tool_call_counts"] = dict(item["tool_call_counts"])
        for key in ("tool_errors", "large_results"):
            cleaned = []
            for tc in item.get(key, []):
                cleaned.append({
                    "name": tc.get("name"),
                    "id": tc.get("id"),
                    "is_error": tc.get("is_error"),
                    "result_size": tc.get("result_size"),
                })
            item[key] = cleaned
        item["duplicate_tools"] = [
            {"tool": d["tool"], "count": d["count"]}
            for d in item.get("duplicate_tools", [])
        ]
        # Serialize new fields
        item["edit_file_targets"] = dict(item.get("edit_file_targets", {}))
        out.append(item)
    return out


def _print_json(analyses):
    """Serialize analyses list to JSON on stdout."""
    out = _serialize_analyses(analyses)
    print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))


if __name__ == "__main__":
    main()
