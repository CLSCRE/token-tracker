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
    }

    # ── Token & cost aggregation ──────────────────────────────────────────
    for call in data.api_calls:
        usage = call.get("usage", {})
        model = call.get("model", "unknown")

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cw  = usage.get("cache_creation_input_tokens", 0)
        cr  = usage.get("cache_read_input_tokens", 0)

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

        cd = usage.get("cache_creation", {})
        a["cache_ephemeral_5m"] += cd.get("ephemeral_5m_input_tokens", 0)
        a["cache_ephemeral_1h"] += cd.get("ephemeral_1h_input_tokens", 0)

        stu = usage.get("server_tool_use", {})
        a["web_search_requests"] += stu.get("web_search_requests", 0)

    # ── Tool-call analysis ────────────────────────────────────────────────
    inputs_by_name = defaultdict(list)  # name -> [(input_json, tc), ...]

    for tc in data.tool_calls:
        name = tc["name"]
        a["tool_call_counts"][name] += 1

        if tc["is_error"]:
            a["tool_errors"].append(tc)

        if tc["result_size"] > 10_000:
            a["large_results"].append(tc)

        input_key = json.dumps(tc.get("input", {}), sort_keys=True)
        inputs_by_name[name].append((input_key, tc))

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
    """Make project dir names more readable."""
    # C--Users-tdamy-OneDrive---CLS-CRE-... → last meaningful segment
    parts = name.replace("---", "/").replace("--", "/").split("/")
    # Take last 2-3 non-empty parts
    parts = [p.strip("-") for p in parts if p.strip("-")]
    if len(parts) > 2:
        return "/".join(parts[-2:])
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
    parser.add_argument("--no-color",            action="store_true", help="Disable colored output")
    args = parser.parse_args()

    if args.no_color:
        global USE_COLOR
        USE_COLOR = False

    scanner = SessionScanner()

    if not scanner.base_dir.exists():
        print(red(f"Error: {scanner.base_dir} not found. Is Claude Code installed?"), file=sys.stderr)
        sys.exit(1)

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

        if args.html:
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
<title>Claude Code Token Tracker</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; --blue: #58a6ff; --purple: #bc8cff; --cyan: #39d2c0;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 24px; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--dim); font-size: 14px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .card-label { color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .card-value { font-size: 28px; font-weight: 700; }
  .card-detail { color: var(--dim); font-size: 13px; margin-top: 4px; }
  .cost { color: var(--green); }
  .section { margin-bottom: 32px; }
  .section-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; background: var(--bg2); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
  th { background: var(--bg3); color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 16px; text-align: left; font-weight: 600; }
  th.right, td.right { text-align: right; }
  td { padding: 12px 16px; border-top: 1px solid var(--border); font-size: 14px; }
  tr:hover td { background: rgba(56, 139, 253, 0.04); }
  .bar-container { display: flex; align-items: center; gap: 8px; }
  .bar { height: 8px; border-radius: 4px; background: var(--blue); min-width: 2px; }
  .bar-label { font-size: 12px; color: var(--dim); white-space: nowrap; }
  .tool-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
  .tool-chip { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
  .tool-chip .count { font-weight: 700; color: var(--blue); }
  .tool-chip .errors { color: var(--red); font-size: 11px; margin-left: 4px; }
  .session-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; cursor: pointer; transition: border-color 0.2s; }
  .session-card:hover { border-color: var(--blue); }
  .session-card.active { border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue); }
  .session-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
  .session-id { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 13px; color: var(--blue); }
  .session-project { font-size: 13px; color: var(--dim); margin-top: 2px; }
  .session-cost { font-size: 22px; font-weight: 700; color: var(--green); }
  .session-meta { display: flex; gap: 16px; font-size: 13px; color: var(--dim); flex-wrap: wrap; }
  .session-meta span { display: flex; align-items: center; gap: 4px; }
  .detail-panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 32px; display: none; }
  .detail-panel.visible { display: block; }
  .token-breakdown { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .token-item { background: var(--bg3); border-radius: 6px; padding: 12px; }
  .token-item .label { font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .token-item .value { font-size: 18px; font-weight: 600; margin-top: 4px; }
  .rec-item { padding: 12px 16px; border-left: 3px solid; margin-bottom: 8px; background: var(--bg3); border-radius: 0 6px 6px 0; }
  .rec-item.high { border-color: var(--red); }
  .rec-item.medium { border-color: var(--yellow); }
  .rec-item.low { border-color: var(--dim); }
  .rec-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
  .rec-detail { font-size: 13px; color: var(--dim); }
  .timestamp { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 12px; }
  @media (max-width: 768px) { body { padding: 12px; } .grid { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<h1>Claude Code Token Tracker</h1>
<p class="subtitle">Generated <span id="gen-time"></span></p>
<div class="grid" id="summary-cards"></div>
<div class="section">
  <div class="section-title">Sessions by Cost</div>
  <div id="sessions-list"></div>
</div>
<div class="detail-panel" id="detail-panel">
  <div class="section-title" id="detail-title">Session Detail</div>
  <div class="token-breakdown" id="detail-tokens"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:20px">
    <div>
      <h3 style="font-size:14px;margin-bottom:12px;color:var(--dim)">Tool Usage</h3>
      <div class="tool-grid" id="detail-tools"></div>
    </div>
    <div>
      <h3 style="font-size:14px;margin-bottom:12px;color:var(--dim)">Recommendations</h3>
      <div id="detail-recs"></div>
    </div>
  </div>
  <div id="detail-issues"></div>
</div>
<div class="section">
  <div class="section-title">All Sessions</div>
  <table>
    <thead>
      <tr>
        <th>Session</th><th>Project</th><th class="right">Cost</th>
        <th class="right">API Calls</th><th class="right">Tool Calls</th>
        <th class="right">Errors</th><th class="right">Output Tokens</th><th>Time</th>
      </tr>
    </thead>
    <tbody id="sessions-table"></tbody>
  </table>
</div>
<script>
const DATA = %%DATA%%;
function fc(c){return c<0.01?'$'+c.toFixed(4):'$'+c.toFixed(2)}
function ft(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)}
function ftime(iso){if(!iso)return'\u2014';const d=new Date(iso);return d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})}
function projectLabel(name){const parts=name.replace(/---/g,'/').replace(/--/g,'/').split('/').filter(p=>p.replace(/-/g,''));return parts.length>2?parts.slice(-2).join('/'):name}
DATA.sort((a,b)=>b.total_cost-a.total_cost);
const totalCost=DATA.reduce((s,d)=>s+d.total_cost,0);
const totalAPI=DATA.reduce((s,d)=>s+d.num_api_calls,0);
const totalTools=DATA.reduce((s,d)=>s+d.num_tool_calls,0);
const totalOut=DATA.reduce((s,d)=>s+d.total_output_tokens,0);
const totalCacheRead=DATA.reduce((s,d)=>s+d.total_cache_read_tokens,0);
const totalErrors=DATA.reduce((s,d)=>s+d.tool_errors.length,0);
document.getElementById('gen-time').textContent=new Date().toLocaleString();
document.getElementById('summary-cards').innerHTML=[
  {label:'Total Cost',value:fc(totalCost),cls:'cost',detail:DATA.length+' sessions'},
  {label:'API Calls',value:totalAPI.toLocaleString(),detail:(totalAPI/DATA.length).toFixed(0)+' avg/session'},
  {label:'Tool Calls',value:totalTools.toLocaleString(),detail:totalErrors+' errors ('+(totalTools?(100*totalErrors/totalTools).toFixed(0):0)+'%)'},
  {label:'Output Tokens',value:ft(totalOut),detail:fc(totalOut/1e6*25)+' at Opus rates'},
  {label:'Cache Read',value:ft(totalCacheRead),detail:'Saved ~'+fc(totalCacheRead/1e6*5*0.9)+' vs uncached'},
].map(c=>`<div class="card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls||''}">${c.value}</div><div class="card-detail">${c.detail}</div></div>`).join('');
const maxCost=DATA[0]?.total_cost||1;
document.getElementById('sessions-list').innerHTML=DATA.map((s,i)=>{
  const errs=s.tool_errors.length;const pct=((s.total_cost/maxCost)*100).toFixed(0);
  return`<div class="session-card" onclick="showDetail(${i})" id="scard-${i}">
    <div class="session-header"><div><div class="session-id">${s.session_id?.slice(0,8)||'?'}...</div><div class="session-project">${projectLabel(s.project||'?')}</div></div><div class="session-cost">${fc(s.total_cost)}</div></div>
    <div style="margin-bottom:8px"><div class="bar-container"><div class="bar" style="width:${pct}%;background:${s.total_cost>5?'var(--red)':s.total_cost>1?'var(--yellow)':'var(--green)'}"></div><span class="bar-label">${ft(s.total_output_tokens)} output</span></div></div>
    <div class="session-meta"><span>${s.num_api_calls} API calls</span><span>${s.num_tool_calls} tools</span>${errs?'<span style="color:var(--red)">'+errs+' errors</span>':''}<span class="timestamp">${ftime(s.start_time)}</span></div>
  </div>`}).join('');
document.getElementById('sessions-table').innerHTML=DATA.map(s=>{
  const errs=s.tool_errors.length;
  return`<tr><td><span class="session-id">${s.session_id?.slice(0,12)||'?'}...</span></td><td style="color:var(--dim);font-size:13px">${projectLabel(s.project||'?')}</td><td class="right cost">${fc(s.total_cost)}</td><td class="right">${s.num_api_calls}</td><td class="right">${s.num_tool_calls}</td><td class="right" style="color:${errs?'var(--red)':'var(--dim)'}">${errs}</td><td class="right">${ft(s.total_output_tokens)}</td><td class="timestamp">${ftime(s.start_time)}</td></tr>`}).join('');
function showDetail(idx){
  document.querySelectorAll('.session-card').forEach(c=>c.classList.remove('active'));
  document.getElementById('scard-'+idx).classList.add('active');
  const s=DATA[idx];const panel=document.getElementById('detail-panel');
  panel.classList.add('visible');panel.scrollIntoView({behavior:'smooth',block:'nearest'});
  document.getElementById('detail-title').innerHTML='Session: <span class="session-id">'+(s.session_id||'?')+'</span> <span style="color:var(--dim);font-size:14px;font-weight:400"> \u2014 '+projectLabel(s.project||'?')+'</span>';
  document.getElementById('detail-tokens').innerHTML=[
    {label:'Input (non-cached)',value:ft(s.total_input_tokens)},{label:'Cache Write',value:ft(s.total_cache_write_tokens)},
    {label:'Cache Read',value:ft(s.total_cache_read_tokens)},{label:'Output',value:ft(s.total_output_tokens)},
    {label:'Total Cost',value:fc(s.total_cost),cls:'cost'},
  ].map(t=>`<div class="token-item"><div class="label">${t.label}</div><div class="value ${t.cls||''}">${t.value}</div></div>`).join('');
  const counts=s.tool_call_counts||{};const errNames={};
  (s.tool_errors||[]).forEach(e=>{errNames[e.name]=(errNames[e.name]||0)+1});
  const sorted=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  document.getElementById('detail-tools').innerHTML=sorted.map(([name,cnt])=>{
    const e=errNames[name]||0;
    return`<div class="tool-chip"><span>${name}</span><span><span class="count">${cnt}</span>${e?'<span class="errors"> '+e+' err</span>':''}</span></div>`}).join('')||'<span style="color:var(--dim)">No tool calls</span>';
  const recs=generateRecs(s);
  document.getElementById('detail-recs').innerHTML=recs.length?recs.map(r=>`<div class="rec-item ${r.severity}"><div class="rec-title">${r.title}</div><div class="rec-detail">${r.detail}</div></div>`).join(''):'<span style="color:var(--green)">No issues found</span>';
  let issues='';
  if(s.duplicate_tools?.length){issues+='<h3 style="font-size:14px;margin:12px 0 8px;color:var(--yellow)">Duplicate Tool Calls</h3>';issues+=s.duplicate_tools.map(d=>`<div style="font-size:13px;color:var(--dim);margin-bottom:4px">${d.tool} x${d.count}</div>`).join('')}
  if(s.large_results?.length){issues+='<h3 style="font-size:14px;margin:12px 0 8px;color:var(--yellow)">Large Results (>10K chars)</h3>';issues+=s.large_results.sort((a,b)=>b.result_size-a.result_size).slice(0,5).map(r=>`<div style="font-size:13px;color:var(--dim);margin-bottom:4px">${r.name}: ${r.result_size.toLocaleString()} chars${r.is_error?' <span style="color:var(--red)">(error)</span>':''}</div>`).join('')}
  document.getElementById('detail-issues').innerHTML=issues}
function generateRecs(s){
  const recs=[];
  (s.duplicate_tools||[]).forEach(d=>{recs.push({severity:'high',title:`Duplicate ${d.tool} calls (${d.count}x)`,detail:`Same input sent ${d.count} times \u2014 wastes tokens and round-trips.`})});
  const nErr=(s.tool_errors||[]).length,nTool=s.num_tool_calls;
  if(nTool>2&&nErr/nTool>0.10){recs.push({severity:'high',title:`High error rate (${nErr}/${nTool} = ${(100*nErr/nTool).toFixed(0)}%)`,detail:'Each error forces a retry, doubling token spend.'})}
  (s.large_results||[]).sort((a,b)=>b.result_size-a.result_size).slice(0,3).forEach(r=>{recs.push({severity:'medium',title:`Large ${r.name} result (${r.result_size.toLocaleString()} chars)`,detail:'Inflates input tokens on subsequent turns.'})});
  if(s.num_api_calls>25){recs.push({severity:'low',title:`Many API round-trips (${s.num_api_calls})`,detail:'Consider batching instructions to reduce calls.'})}
  return recs}
const currentIdx=DATA.findIndex(s=>s.project?.includes('Token-Tracker'));
if(currentIdx>=0)showDetail(currentIdx);
</script>
</body>
</html>'''


def _build_html(analyses, watch_mode=False):
    """Build the HTML string with data injected."""
    serialized = _serialize_analyses(analyses)
    data_json = json.dumps(serialized, default=str)
    html = _HTML_TEMPLATE.replace("%%DATA%%", data_json)
    if watch_mode:
        # Inject live-refresh script: fetches /data every 30s, re-renders without full reload
        refresh_js = """
<script>
(function(){
  const INTERVAL = 30000;
  let _selectedIdx = -1;
  const origShow = window.showDetail;
  window.showDetail = function(idx){ _selectedIdx = idx; origShow(idx); };

  async function refresh(){
    try {
      const resp = await fetch('/data');
      if (!resp.ok) return;
      const fresh = await resp.json();
      fresh.sort((a,b) => b.total_cost - a.total_cost);

      // Patch global DATA in-place
      DATA.length = 0;
      fresh.forEach(d => DATA.push(d));

      // Re-render summary cards
      const totalCost2=DATA.reduce((s,d)=>s+d.total_cost,0);
      const totalAPI2=DATA.reduce((s,d)=>s+d.num_api_calls,0);
      const totalTools2=DATA.reduce((s,d)=>s+d.num_tool_calls,0);
      const totalOut2=DATA.reduce((s,d)=>s+d.total_output_tokens,0);
      const totalCacheRead2=DATA.reduce((s,d)=>s+d.total_cache_read_tokens,0);
      const totalErrors2=DATA.reduce((s,d)=>s+d.tool_errors.length,0);
      document.getElementById('gen-time').textContent=new Date().toLocaleString() + '  (live)';
      document.getElementById('summary-cards').innerHTML=[
        {label:'Total Cost',value:fc(totalCost2),cls:'cost',detail:DATA.length+' sessions'},
        {label:'API Calls',value:totalAPI2.toLocaleString(),detail:(totalAPI2/DATA.length).toFixed(0)+' avg/session'},
        {label:'Tool Calls',value:totalTools2.toLocaleString(),detail:totalErrors2+' errors ('+(totalTools2?(100*totalErrors2/totalTools2).toFixed(0):0)+'%)'},
        {label:'Output Tokens',value:ft(totalOut2),detail:fc(totalOut2/1e6*25)+' at Opus rates'},
        {label:'Cache Read',value:ft(totalCacheRead2),detail:'Saved ~'+fc(totalCacheRead2/1e6*5*0.9)+' vs uncached'},
      ].map(c=>`<div class="card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls||''}">${c.value}</div><div class="card-detail">${c.detail}</div></div>`).join('');

      // Re-render session list
      const maxCost2=DATA[0]?.total_cost||1;
      document.getElementById('sessions-list').innerHTML=DATA.map((s,i)=>{
        const errs=s.tool_errors.length;const pct=((s.total_cost/maxCost2)*100).toFixed(0);
        return`<div class="session-card${i===_selectedIdx?' active':''}" onclick="showDetail(${i})" id="scard-${i}">
          <div class="session-header"><div><div class="session-id">${s.session_id?.slice(0,8)||'?'}...</div><div class="session-project">${projectLabel(s.project||'?')}</div></div><div class="session-cost">${fc(s.total_cost)}</div></div>
          <div style="margin-bottom:8px"><div class="bar-container"><div class="bar" style="width:${pct}%;background:${s.total_cost>5?'var(--red)':s.total_cost>1?'var(--yellow)':'var(--green)'}"></div><span class="bar-label">${ft(s.total_output_tokens)} output</span></div></div>
          <div class="session-meta"><span>${s.num_api_calls} API calls</span><span>${s.num_tool_calls} tools</span>${errs?'<span style="color:var(--red)">'+errs+' errors</span>':''}<span class="timestamp">${ftime(s.start_time)}</span></div>
        </div>`}).join('');

      // Re-render table
      document.getElementById('sessions-table').innerHTML=DATA.map(s=>{
        const errs=s.tool_errors.length;
        return`<tr><td><span class="session-id">${s.session_id?.slice(0,12)||'?'}...</span></td><td style="color:var(--dim);font-size:13px">${projectLabel(s.project||'?')}</td><td class="right cost">${fc(s.total_cost)}</td><td class="right">${s.num_api_calls}</td><td class="right">${s.num_tool_calls}</td><td class="right" style="color:${errs?'var(--red)':'var(--dim)'}">${errs}</td><td class="right">${ft(s.total_output_tokens)}</td><td class="timestamp">${ftime(s.start_time)}</td></tr>`}).join('');

      // Re-render detail if one is selected
      if (_selectedIdx >= 0 && _selectedIdx < DATA.length) showDetail(_selectedIdx);
    } catch(e) { console.warn('refresh failed', e); }
  }
  setInterval(refresh, INTERVAL);
})();
</script>"""
        html = html.replace("</body>", refresh_js + "\n</body>")
    return html


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
        out.append(item)
    return out


def _print_json(analyses):
    """Serialize analyses list to JSON on stdout."""
    out = _serialize_analyses(analyses)
    print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))


if __name__ == "__main__":
    main()
