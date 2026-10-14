"""Probe GET /health for defense-day demo. Exit 0 only when MS-STA B4-V6 defense is ready."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Locked B4-V6 seed42 defense checkpoint (bit-exact with active V6 registry).
_DEFENSE_CKPT_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("defense", "b0"),
        default="defense",
        help="defense (default) requires MS-STA B4-V6; b0 allows historical baseline smoke",
    )
    ap.add_argument("--url", default="http://127.0.0.1:8000/health")
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(args.url, timeout=3) as resp:
            h = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"FAIL connect: {exc}", file=sys.stderr)
        return 1

    ready = h.get("ready") is True
    vsl = h.get("primary_dataset") == "VSL400"
    modes = set(h.get("modes") or [])
    schema = "isolated" in modes and "gloss_llm" in modes and h.get("continuous_news_removed") is True
    stale = (
        "retrieval_index_ready" in h
        or "appendix_continuous_news" in h
        or "ready" not in h
    )
    errors: list[str] = []
    if not ready:
        errors.append("ready!=true")
    if not vsl:
        errors.append("primary_dataset!=VSL400")
    if not schema:
        errors.append("modes/schema invalid")
    if stale:
        errors.append("stale CN-era health fields")

    if args.mode == "defense":
        if h.get("ms_sta_ready") is not True:
            errors.append("ms_sta_ready!=true")
        if h.get("engine") != "ms_sta_v6_b4_v6_ctr":
            errors.append(f"engine!={h.get('engine')!r} expect ms_sta_v6_b4_v6_ctr")
        if h.get("ms_sta_variant") != "b4_v6_ctr":
            errors.append(f"ms_sta_variant={h.get('ms_sta_variant')!r}")
        if h.get("ms_sta_v6") is not True:
            errors.append("ms_sta_v6!=true")
        if h.get("e4_ready") is not False:
            errors.append(f"e4_ready={h.get('e4_ready')!r} expect false")
        if h.get("ms_sta_input_mode") != "cache_backed_offline":
            errors.append("ms_sta_input_mode!=cache_backed_offline")
        if h.get("campaign_status") != "DEFENSE_GO":
            errors.append("campaign_status!=DEFENSE_GO")
        got_sha = str(h.get("ms_sta_checkpoint_sha256") or "")
        if got_sha != _DEFENSE_CKPT_SHA:
            errors.append(
                f"ms_sta_checkpoint_sha256 mismatch got={got_sha[:12] or 'none'} "
                f"expect={_DEFENSE_CKPT_SHA[:12]}"
            )
        v6 = h.get("v6_kpi") or {}
        if v6.get("scope") != "v6_defense_go":
            errors.append(f"v6_kpi.scope={v6.get('scope')!r} expect v6_defense_go")

    summary = {
        k: h.get(k)
        for k in (
            "status",
            "ready",
            "engine",
            "ms_sta_ready",
            "ms_sta_variant",
            "ms_sta_checkpoint_sha256",
            "ms_sta_v6",
            "e4_ready",
            "ms_sta_input_mode",
            "campaign_status",
            "primary_dataset",
            "modes",
            "continuous_news_removed",
            "device",
            "isolated_num_classes",
        )
    }
    ok = not errors
    print(("PASS " if ok else "FAIL ") + json.dumps(summary, ensure_ascii=False))
    if errors:
        print("errors: " + "; ".join(errors), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
