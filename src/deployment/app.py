"""Streamlit demo: VSL400 word-level (isolated + gloss→LLM). Continuous_News removed."""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

# --- Bootstrap BEFORE any `src.*` import (streamlit run does not put repo root on path) ---
_ROOT = Path(__file__).resolve().parents[2]
_ROOT_STR = str(_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)
_os_path = os.environ.get("PYTHONPATH", "")
if _ROOT_STR not in _os_path.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _ROOT_STR if not _os_path else _ROOT_STR + os.pathsep + _os_path
    )

import pandas as pd
import requests
import streamlit as st

from src.data_prep.raw_video_resolve import RawVideoResolver
from src.deployment.input_evidence import (
    load_active_defense_registry,
    rgb_feature_summary,
    sealed_feature_paths,
    v5_feature_paths,
)
from src.deployment.input_evidence_ui import (
    browser_playable_bytes,
    show_front_video,
    skeleton_gif_bytes,
)
from src.deployment.skeleton_visualization import skeleton_summary

API = "http://127.0.0.1:8000"
_DATASETS = ("VSL400", "VSL-Cropped")
_MAX_OPTIONS = 400

_THESIS_VI = (
    "Nghiên cứu cải tiến mô-đun chú ý không gian - thời gian đa luồng "
    "trong nhận diện ngôn ngữ ký hiệu tiếng Việt từ video"
)
_THESIS_EN = (
    "Researching and improving multi-stream spatial-temporal attention module "
    "in Vietnamese sign language recognition from video"
)


def _norm(text: object) -> str:
    return str(text or "").strip().casefold()


def load_clip_table(dataset: str) -> pd.DataFrame:
    """Load front-view (preferred) clips for a dataset from metadata.csv."""
    df = pd.read_csv(_ROOT / "metadata.csv")
    cols = ["video_id", "split", "signer_id", "text_vietnamese", "text_gloss", "dataset", "view"]
    cols = [c for c in cols if c in df.columns]
    out = df[df["dataset"] == dataset][cols].copy()
    if "view" in out.columns:
        front = out[out["view"].astype(str).str.lower().eq("front")]
        if not front.empty:
            out = front
    return out.reset_index(drop=True)


@st.cache_data
def load_clip_options(dataset: str) -> pd.DataFrame:
    return load_clip_table(dataset)


@st.cache_resource(show_spinner="Indexing videos (first run)…")
def get_resolver() -> RawVideoResolver:
    resolver = RawVideoResolver(_ROOT)
    _ = resolver.vsl400_index()
    _ = resolver.vsl_index()
    return resolver


# load_active_defense_registry / sealed_feature_paths / browser helpers:
# imported from src.deployment.input_evidence (shared with defense demo pages).
# v5_feature_paths remains as a compatibility alias for sealed_feature_paths.
load_active_v5_registry = load_active_defense_registry


def clip_label(row: pd.Series) -> str:
    gloss = str(row.get("text_gloss") or "").strip() or "?"
    vi = str(row.get("text_vietnamese") or "").strip() or "?"
    signer = str(row.get("signer_id") or "").replace("signer_", "S")
    return f"{gloss}  ·  {vi}  ·  {row.video_id}  ·  {signer}"


def match_flags(preds: list[dict], gt_gloss: str, gt_vi: str) -> tuple[bool, bool, list[dict]]:
    """Compare Top-k predictions to ground-truth gloss / Vietnamese."""
    g, v = _norm(gt_gloss), _norm(gt_vi)
    rows: list[dict] = []
    top1 = top5 = False
    for i, p in enumerate(preds):
        hit = (_norm(p.get("gloss")) == g) or (_norm(p.get("vietnamese")) == v)
        if hit and i == 0:
            top1 = True
        if hit and i < 5:
            top5 = True
        rows.append(
            {
                "Rank": i + 1,
                "Gloss": p.get("gloss"),
                "Vietnamese": p.get("vietnamese"),
                "Probability": round(float(p.get("probability") or 0), 4),
                "Match GT": "✓" if hit else "",
            }
        )
    return top1, top5, rows


def _fetch_health() -> tuple[requests.Response | None, dict | None, str | None]:
    try:
        resp = requests.get(f"{API}/health", timeout=3)
        return resp, resp.json(), None
    except Exception as exc:
        return None, None, str(exc)


def _mean_std_pct(fractions: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Mean/sample-std in percent from fractions in [0,1]."""
    xs = [100.0 * float(x) for x in fractions]
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, var**0.5


def _engine_label(health: dict, use_ms: bool) -> str:
    if not use_ms:
        return "b0_gate3"
    return str(health.get("engine") or health.get("experiment_variant") or "ms_sta")


def _show_video(video_path: Path) -> None:
    """Render MP4 preview playable in browser (H.264)."""
    show_front_video(video_path, root=_ROOT)


def main() -> None:
    st.set_page_config(
        page_title="VSL400 MS-STA Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Vietnamese Sign Language Recognition — Demo")
    st.info(
        "Đây là hệ thống nhận dạng ký hiệu đơn lẻ, chưa phải hệ thống dịch câu ngôn ngữ ký hiệu liên tục."
    )
    st.caption(
        "Demo đang bảo vệ dùng VSL400, mức từ đơn, mô hình Unified Attention trên RGB và skeleton. "
        "Tập đánh giá tách theo signer-ID. Giao diện đọc feature cache của Unified Attention theo Protocol A. "
        "Không nhận video tải lên tùy ý."
    )

    with st.expander("Đề tài đăng ký / Registered thesis title", expanded=True):
        st.markdown(f"**VI:** {_THESIS_VI}")
        st.markdown(f"**EN:** {_THESIS_EN}")

    st.success(
        "**Hai luận điểm duy nhất**  \n"
        "1) **Unified Attention (mô hình):** hợp nhất RGB–Skeleton có cấu trúc "
        "(Protocol A · Top-1 ≈51,55%).  \n"
        "2) **Optimized inference (hệ thống):** cùng checkpoint, gom ROI và dùng mixed precision. "
        "Số liệu Protocol B lấy từ bằng chứng đã khóa. Giao diện này không đo tốc độ đó."
    )
    st.caption(
        "`signer_id` là mã người ra dấu. "
        "Demo hiện dùng feature cache của Unified Attention. "
        "Optimized inference không chạy trên giao diện này, vì giao diện chỉ minh họa Protocol A."
    )
    try:
        from src.deployment.campaign_evidence import runtime_ui_state

        _st = runtime_ui_state(_ROOT)
        st.warning(
            f"{_st['s5_label_vi']}. "
            "Mô hình khoa học là **Unified Attention**. "
            "Cách suy luận đã chọn trên bằng chứng là **Optimized inference**. "
            "Giao diện đang chạy vẫn là **Unified Attention**, đọc đặc trưng đã tính sẵn (Protocol A). "
            "Mười bảy kiểm tra đầu ra đều đạt. Các kiểm tra này không thay điểm Top-1."
        )
    except Exception as _exc:
        st.caption(f"Chưa đọc được trạng thái chiến dịch: {_exc}")

    with st.expander("Định hướng sau này — Continuous sentence recognition", expanded=False):
        st.markdown(
            "Nhận biết **câu ký hiệu liên tục** / dịch câu là hướng phát triển sau bảo vệ "
            "(CSLR → SLT; embedding Unified Attention làm encoder nền tảng). "
            "Không thuộc số liệu bảo vệ của demo này.\n\n"
            "Chi tiết: `Docs/roadmap/CONTINUOUS_SENTENCE_ROADMAP.md`."
        )

    health_resp, health, health_err = _fetch_health()
    if health_err or health_resp is None or health is None:
        st.error(
            f"API unreachable: `{API}`.\n\n"
            "Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\demo_vsl400.ps1` "
            "(env_demo / `_venvs\\demo`) then reload.\n\n"
            f"`{health_err}`"
        )
        st.stop()

    if health_resp.status_code != 200 or not health.get("ready"):
        st.error(
            f"API not ready (HTTP {health_resp.status_code}, ready={health.get('ready')}). "
            "Need defense ckpt + `metadata.csv`. "
            "See `Docs/phases/04_Deployment.md`."
        )
        st.stop()

    ms_ready = bool(health.get("ms_sta_ready"))
    engine_live = str(health.get("engine") or "—")
    campaign = str(health.get("campaign_status") or "")
    ckpt_sha_live = str(health.get("ms_sta_checkpoint_sha256") or "")
    v3 = health.get("retrain_v3_kpi") or {}
    v6 = health.get("v6_kpi") or {}
    v5 = health.get("v5_kpi") or {}
    b4_t1 = health.get("e2_b4_top1")
    b4_t5 = health.get("e2_b4_top5")
    b0_t1 = health.get("e2_b0_e1_top1")
    delta = health.get("e2_delta_top1_pp")
    gate3_t1 = health.get("b0_gate3_top1")
    e4_ready = bool(health.get("e4_ready"))

    if campaign == "VALIDATION_NO_GO" or health.get("demo_allowed") is False:
        st.error(
            "**Active campaign = VALIDATION_NO_GO.** Demo defense is disabled; "
            "historical checkpoints must not be presented as current evidence."
        )

    status_cols = st.columns([1.1, 1.1, 1.1, 1.1, 1.6])
    status_cols[0].metric("Device", str(health.get("device")))
    if v6.get("scope") == "v6_defense_go":
        confirm = list(v6.get("validation_confirm_top1") or [])
        val_mean, val_std = _mean_std_pct(confirm)
        status_cols[1].metric(
            "Unified Attention · val Top-1",
            f"{val_mean:.2f} ± {val_std:.2f}%",
            help="Mean ± sample std over three confirm seeds from /health.",
        )
        status_cols[2].metric(
            "Unified Attention · test Top-1",
            f"{100.0 * float(v6.get('test_b4_top1') or 0):.2f}%",
            help="Retrospective signer-disjoint confirmation.",
        )
        status_cols[3].metric(
            "Δ vs Simple Combined",
            f"+{float(v6.get('test_delta_top1_pp') or 0):.2f} pp",
            help="Primary paired contrast on test (n from /health).",
        )
        if ms_ready and not e4_ready:
            status_cols[4].success(
                f"Defense GO · Unified Attention · `{ckpt_sha_live[:12]}…`"
            )
        elif ms_ready:
            status_cols[4].error("Optimized-inference flag incompatible with Defense GO UI")
        else:
            status_cols[4].error("Defense registry active, but Unified Attention failed to load")
    elif v5.get("scope") == "v5_defense_go":
        status_cols[1].metric(
            "Unified Attention · val Top-1 (rollback)",
            f"{float(v5.get('validation_b4_top1_pct') or 0):.2f}%",
            f"+{float(v5.get('validation_delta_top1_pp') or 0):.2f} pp vs Simple Combined",
        )
        status_cols[2].metric(
            "Unified Attention · test Top-1 (rollback)",
            f"{100.0 * float(v5.get('test_b4_top1') or 0):.2f}%",
            help="Retrospective signer-disjoint confirmation, n=6165.",
        )
        status_cols[3].metric(
            "Test gain (rollback)",
            f"+{float(v5.get('test_delta_top1_pp') or 0):.2f} pp",
            help="Primary paired contrast: Unified Attention vs Simple Combined.",
        )
        if ms_ready:
            status_cols[4].success(f"V5 rollback · `{engine_live}`")
        else:
            status_cols[4].error("Defense registry active, but Unified Attention v5 failed to load")
    elif v3.get("ready") and (campaign == "VALIDATION_NO_GO" or health.get("demo_allowed") is False):
        status_cols[1].metric(
            "B3 val Top-1",
            f"{float(v3.get('b3_top1_mean_pct') or 0):.2f}%",
            help="Path-B confirmation validation mean (canonical reference).",
        )
        if v3.get("b4r_top1_mean_pct") is not None:
            status_cols[2].metric(
                "B4R val Top-1",
                f"{float(v3.get('b4r_top1_mean_pct') or 0):.2f}%",
                help="Hypothesis; outperform gate FAIL.",
            )
            status_cols[3].metric(
                "B4R−B3",
                f"{float(v3.get('mean_b4r_minus_b3_pp') or 0):+.2f} pp",
            )
        else:
            status_cols[2].metric(
                "B4 val Top-1",
                f"{float(v3.get('b4_top1_mean_pct') or 0):.2f}%",
                help="Failed architecture gate vs B3.",
            )
            status_cols[3].metric(
                "B4−B3",
                f"{float(v3.get('mean_b4_minus_b3_pp') or 0):+.2f} pp",
            )
        status_cols[4].warning("NO-GO · no MS-STA defense ckpt")
    elif b4_t1 is not None:
        status_cols[1].metric("E2 B4 Top-1", f"{float(b4_t1):.2f}%")
        status_cols[2].metric("E2 B4 Top-5", f"{float(b4_t5 or 0):.2f}%")
        status_cols[3].metric("Δ vs B0-E1", f"+{float(delta or 0):.2f} pp")
        if ms_ready:
            status_cols[4].success(
                f"Defense · `{engine_live}` · {health.get('isolated_num_classes') or '—'} classes"
            )
        else:
            status_cols[4].warning(f"B0-only · `{engine_live}` — MS-STA not loaded")
    else:
        status_cols[1].metric("SI Top-1", f"{float(gate3_t1 or 0):.1f}%")
        status_cols[2].metric("SI Top-5", f"{float(health.get('vsl400_top5') or 0):.1f}%")
        status_cols[3].metric("Engine", engine_live)
        status_cols[4].warning(f"B0-only · `{engine_live}` — MS-STA not loaded")

    with st.expander("KPI scope and claim limits", expanded=False):
        if v6.get("scope") == "v6_defense_go":
            confirm = list(v6.get("validation_confirm_top1") or [])
            val_mean, val_std = _mean_std_pct(confirm)
            st.markdown(
                f"- **Tập kiểm định (3 seed):** Top-1 của Unified Attention "
                f"**{val_mean:.2f} ± {val_std:.2f}%**.\n"
                f"- **Tập kiểm tra (seed 42):** xác nhận tách người, đã dùng khi khóa mô hình. "
                f"Top-1 **{100.0 * float(v6.get('test_b4_top1') or 0):.2f}%**, "
                f"n={v6.get('test_n')}; "
                f"cao hơn Simple Combined **+{float(v6.get('test_delta_top1_pp') or 0):.2f} điểm phần trăm**.\n"
                f"- **Mã băm checkpoint:** `{ckpt_sha_live[:16]}…`.\n"
                "- Demo dùng mẫu đã khóa và đặc trưng đã tính sẵn. Demo không phải bằng chứng KPI và không nhận video tùy ý.\n"
                "- Mô hình công bố là **Unified Attention**. Cách suy luận nhanh hơn là **Optimized inference**, số đọc từ tệp bằng chứng.\n"
                "- Giao diện đang chạy **Unified Attention** theo Protocol A. "
                "Optimized inference không gom ROI trên giao diện này."
            )
        elif v5.get("scope") == "v5_defense_go":
            st.markdown(
                f"- **Rollback V5:** Unified Attention **{float(v5.get('validation_b4_top1_pct') or 0):.2f}%** "
                f"vs Simple Combined **{float(v5.get('validation_mlp_top1_pct') or 0):.2f}%**.\n"
                f"- **Test (seed 42):** retrospective signer-disjoint confirmation, "
                f"Unified Attention **{100.0 * float(v5.get('test_b4_top1') or 0):.2f}%**, n={v5.get('test_n')}.\n"
                "- Active SoT is V6 when `v6_kpi` is present; this branch is rollback only."
            )
        elif v3.get("ready"):
            camp = v3.get("campaign") or "retrain_v4"
            if v3.get("b4r_top1_mean_pct") is not None:
                delta_line = (
                    f"B4R−B3 **{float(v3.get('mean_b4r_minus_b3_pp') or 0):+.2f} pp** "
                    f"(outperform gate FAIL)."
                )
            else:
                delta_line = (
                    f"B4−B3 **{float(v3.get('mean_b4_minus_b3_pp') or 0):+.2f} pp**."
                )
            st.markdown(
                f"- **Primary ({camp}):** campaign `{v3.get('campaign_status')}`; "
                f"canonical **B3** Top-1 **{float(v3.get('b3_top1_mean_pct') or 0):.2f}%** "
                f"(sample std {float(v3.get('b3_top1_std_pct') or 0):.2f}); "
                f"{delta_line}\n"
                "- Historical v3 / E1-v2/E2/D4 are **not primary**.\n"
                "- Demo MS-STA defense checkpoint: **none approved**.\n"
                "- Forbidden: Defense GO, B4R outperform, 93.75% as experimental success."
            )
        else:
            st.markdown(
                f"- **E2 B4 / B0-E1** (historical only when explicitly allowed): "
                f"B4 Top-1 **{float(b4_t1 or 0):.2f}%** / Top-5 **{float(b4_t5 or 0):.2f}%** "
                f"vs B0-E1 **{float(b0_t1 or 0):.2f}%** (n={health.get('e2_n') or 6165}).\n"
                f"- **Gate3 B0 historical** (demo fallback baseline): Top-1 **{float(gate3_t1 or 0):.2f}%**.\n"
                "- Always require human confirmation."
            )

    accuracy_study = health.get("accuracy_study") or {}
    if accuracy_study.get("ready"):
        st.markdown("### Post-defense validation optimization")
        st.caption(
            "The learning rate was selected in a bounded seed-42 pilot, frozen, then rerun "
            "on seeds 42, 123 and 456. Architecture, data and initialization are unchanged."
        )
        acc_after, acc_delta, acc_scope = st.columns(3)

        acc_after.metric(
            "Current B4 · validation",
            f"{float(accuracy_study['optimized_top1_mean_pct']):.2f}% Top-1",
            help=(
                f"Three-seed sample std: {float(accuracy_study['optimized_top1_std_pct']):.2f} pp."
            ),
        )
        acc_delta.metric(
            "Paired gain",
            f"+{float(accuracy_study['paired_delta_top1_mean_pp']):.2f} pp",
            help="All three seeds improved on the frozen validation split.",
        )
        acc_scope.metric(
            "Evidence scope",
            "Validation only",
            help="No new E2 or D4 has been run for the current optimized checkpoints.",
        )
        st.info(
            "Validation-only evidence: E2 was not reopened and the live demo still uses the "
            "previously locked B4 checkpoint. This result is not a new test-set claim."
        )

    efficiency = health.get("efficiency_ablation") or {}
    if efficiency.get("ready"):
        st.markdown("### Controlled efficiency ablation (validation only)")
        st.caption(
            "Same B4 architecture and three validation seeds; only the Skeleton temporal "
            "sequence changes from stride 1 to parameter-free average-pool stride 4."
        )
        before_col, after_col, measured_col = st.columns(3)
        before_col.metric(
            "Before · B4 stride 1",
            f"{float(efficiency['base_top1_mean_pct']):.2f}% Top-1",
            help="Mean over seeds 42, 123 and 456 on the frozen validation split.",
        )
        after_col.metric(
            "After · Skeleton stride 4",
            f"{float(efficiency['stride4_top1_mean_pct']):.2f}% Top-1",
            f"{float(efficiency['paired_delta_top1_mean_pp']):+.2f} pp",
            help="Auxiliary validation result; it was not evaluated on E2 test.",
        )
        measured_col.metric(
            "Model-forward median",
            f"{float(efficiency['latency_after_ms']):.2f} ms",
            f"-{float(efficiency['latency_reduction_pct']):.2f}%",
            help=(
                f"Before {float(efficiency['latency_before_ms']):.2f} ms; "
                f"{efficiency['benchmark_device']}; excludes feature extraction, disk I/O, API and UI."
            ),
        )
        st.markdown(
            f"Skeleton temporal score elements per head/sample: "
            f"**{int(efficiency['score_elements_before']):,} -> "
            f"{int(efficiency['score_elements_after']):,} "
            f"(-{float(efficiency['score_reduction_pct']):.2f}%)**. "
            "Peak allocated VRAM did not decrease in the locked benchmark."
        )
        st.info(
            "The live defense engine below remains the locked B4 stride-1 checkpoint. "
            "Stride 4 is validation-only auxiliary evidence, not a replacement E2/demo model."
        )

    with st.expander("Scope & claims (read before presenting)", expanded=False):
        st.markdown(
            "- **Scientific model:** **Unified Attention**.\n"
            "- **Selected online runtime:** **Optimized inference** (same checkpoint + AMP + ROI batching) — evidence-only here.\n"
            "- **Current demo engine:** **Unified Attention cache-backed** / Protocol A.\n"
            "- Đây là hệ thống nhận dạng ký hiệu đơn lẻ 400 lớp; chưa dịch câu liên tục.\n"
            "- Nghiên cứu nén mạng = phụ lục NO-GO; tối ưu hệ thống = Optimized inference trên cùng checkpoint.\n"
            "- Use **split = test** and the locked sealed samples below."
        )

    st.sidebar.header("1. Select clip")
    mode = "isolated"
    st.sidebar.caption("Mode: **word-level / isolated** (defense surface).")

    # Curated defense scenarios; demonstration only, never KPI evidence.
    # Prefer registry demo_set_ms_sta.json (roles match live Unified Attention on these clips).
    demo_pack = _ROOT / "data" / "processed" / "demo_set_ms_sta.json"
    if not demo_pack.is_file():
        demo_pack = _ROOT / "data" / "processed" / "demo_set_ms_sta_v5.json"
    demo_scenarios: dict = {}
    if demo_pack.is_file():
        try:
            demo_scenarios = (json.loads(demo_pack.read_text(encoding="utf-8-sig")).get("scenarios") or {})
        except Exception:
            demo_scenarios = {}
    scenario_order = [
        k
        for k in ("top1_correct", "top1_improvement", "top5_assist")
        if k in demo_scenarios
    ]
    scenario_labels = {
        "top1_correct": "① Unified Attention Top-1 đúng (023292 · CÁI_ĐÈN)",
        "top1_improvement": "① Unified Attention Top-1 đúng (018613 · CHỊ)",
        "top5_assist": (
            "② Top-5 cần xác nhận (018613 · CHỊ)"
            if str(demo_scenarios.get("top5_assist")) == "018613"
            else "② Top-5 cần xác nhận (023292 · CÁI_ĐÈN)"
        ),
    }
    if scenario_order:
        pick = st.sidebar.radio(
            "Defense scenarios (curated)",
            options=["(manual)"] + scenario_order,
            format_func=lambda k: scenario_labels.get(k, k),
            index=1 if scenario_order else 0,
        )
    else:
        pick = "(manual)"

    dataset_choice = st.sidebar.selectbox("Dataset", list(_DATASETS), index=0)
    split_choice = st.sidebar.selectbox(
        "Split",
        ["test", "train"],
        index=0,
        help="test = SI (signers unseen at train time). Prefer test for committee demo.",
    )

    df = load_clip_options(dataset_choice)
    if df.empty:
        st.error(f"No clips for dataset={dataset_choice} in metadata.csv")
        st.stop()

    subset = df[df["split"] == split_choice].reset_index(drop=True)
    if subset.empty:
        st.sidebar.warning(f"Split `{split_choice}` empty — using all rows.")
        subset = df.reset_index(drop=True)

    signers = ["(all)"] + sorted(subset["signer_id"].astype(str).unique().tolist())
    signer_pick = st.sidebar.selectbox("Signer", signers, index=0)
    if signer_pick != "(all)":
        subset = subset[subset["signer_id"].astype(str) == signer_pick].reset_index(drop=True)

    query = st.sidebar.text_input(
        "Search gloss / Vietnamese / video_id",
        placeholder="e.g. ANH, Anh, 019000",
    ).strip()
    if query:
        q = query.casefold()
        mask = (
            subset["video_id"].astype(str).str.casefold().str.contains(q, na=False)
            | subset["text_gloss"].astype(str).str.casefold().str.contains(q, na=False)
            | subset["text_vietnamese"].astype(str).str.casefold().str.contains(q, na=False)
        )
        subset = subset[mask].reset_index(drop=True)

    if subset.empty:
        st.warning("No clips after filter. Broaden search / signer.")
        st.stop()

    display = subset.head(_MAX_OPTIONS)
    if len(subset) > _MAX_OPTIONS:
        st.sidebar.caption(
            f"Showing {_MAX_OPTIONS}/{len(subset)} clips — narrow the search."
        )

    id_list = display["video_id"].astype(str).tolist()
    label_by_id = {
        str(display.iloc[i]["video_id"]): clip_label(display.iloc[i])
        for i in range(len(display))
    }
    default_idx = 0
    if pick != "(manual)" and demo_scenarios.get(pick):
        want = str(demo_scenarios[pick])
        if want in id_list:
            default_idx = id_list.index(want)
        elif want not in id_list:
            # Scenario may be filtered out — force-include by searching full df
            hit = df[df["video_id"].astype(str) == want]
            if not hit.empty:
                subset = hit.reset_index(drop=True)
                display = subset
                id_list = display["video_id"].astype(str).tolist()
                label_by_id = {
                    str(display.iloc[i]["video_id"]): clip_label(display.iloc[i])
                    for i in range(len(display))
                }
                default_idx = 0
                st.sidebar.caption(f"Locked scenario `{want}` forced into selection.")
    selected_id = st.sidebar.selectbox(
        "Clip",
        options=id_list,
        index=default_idx if id_list else 0,
        format_func=lambda vid: label_by_id.get(vid, vid),
        help="Label = gloss · Vietnamese · video_id · signer.",
    )
    row = display[display["video_id"].astype(str) == str(selected_id)].iloc[0]
    clip_key = f"{row.dataset}:{row.video_id}:{row.get('view', '')}"

    if st.session_state.get("_clip_key") != clip_key:
        st.session_state._clip_key = clip_key
        st.session_state.result_payload = None
        st.session_state.pop("confirm_result", None)
        st.session_state.pop("edit_result", None)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**video_id:** `{row.video_id}`")
    st.sidebar.caption(f"{len(subset)} clips · split={split_choice}")

    resolver = get_resolver()
    video_path = resolver.resolve(row)

    st.subheader("2. Two-stream input")
    st.caption(
        "Left: Front-view RGB clip for inspection. Right: the aligned Skeleton v3 sequence "
        "consumed by Unified Attention. Inference reads sealed RGB-token/Skeleton caches, not live extraction."
    )
    col_video, col_skel = st.columns(2, gap="large")
    try:
        rgb_cache_path, skel_cache_path = sealed_feature_paths(str(row.video_id))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        rgb_cache_path = skel_cache_path = Path("__missing__")
        st.error(f"Sealed input-feature contract unavailable: {exc}")

    with col_video:
        st.markdown("**RGB · Front camera**")
        if video_path is not None and video_path.is_file():
            _show_video(video_path)
        else:
            st.error(f"Video file not found for `{row.video_id}`.")
        if rgb_cache_path.is_file():
            rgb_info = rgb_feature_summary(rgb_cache_path)
            st.caption(
                f"RGB token v3 · {rgb_info['windows']} temporal windows · "
                f"regions: {', '.join(rgb_info['regions'])}"
            )
        else:
            st.error("RGB token v3 cache missing — Unified Attention inference will fail closed.")

    with col_skel:
        st.markdown("**Skeleton · 75-joint motion**")
        if skel_cache_path.is_file():
            try:
                skeleton_view = st.radio(
                    "Skeleton visualization",
                    ("Stabilized display (VIZ-ONLY)", "Raw sealed motion"),
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"skeleton_view_{row.video_id}",
                )
                stabilize_display = skeleton_view.startswith("Stabilized")
                gif = skeleton_gif_bytes(
                    str(skel_cache_path.resolve()),
                    skel_cache_path.stat().st_mtime_ns,
                    stabilize_display,
                )
                st.image(gif, width="stretch")
                skel_info = skeleton_summary(skel_cache_path)
                st.caption(
                    f"Skeleton v3 · {skel_info['frames']} frames · {skel_info['joints']} joints · "
                    f"valid {100.0 * float(skel_info['valid_joint_fraction']):.1f}% · "
                    f"shoulder-center error <= {float(skel_info['shoulder_center_max_abs']):.1e}"
                )
                display_note = (
                    "Torso/face smoothing is VIZ-ONLY; hands remain raw."
                    if stabilize_display
                    else "No display smoothing is applied."
                )
                st.caption(
                    "Pose · Face · Left hand · Right hand. Canvas is fixed at the two-shoulder "
                    f"midpoint. {display_note} Unified Attention always consumes the unchanged sealed cache "
                    "(VIZ-ONLY never mutates tensors)."
                )
            except (OSError, ValueError) as exc:
                st.error(f"Skeleton visualization failed closed: {exc}")
        else:
            st.error("Skeleton v3 cache missing — Unified Attention inference will fail closed.")

    st.subheader("3. Ground truth (GT)")
    gt_cols = st.columns(4)
    gt_cols[0].metric("Gloss", str(row.get("text_gloss", "")))
    gt_cols[1].metric("Vietnamese", str(row.get("text_vietnamese", "")))
    gt_cols[2].metric("Signer", str(row.get("signer_id", "")))
    gt_cols[3].metric("Split / view", f"{row.get('split', '')} / {row.get('view', 'front')}")
    st.caption("Ground truth is shown for committee inspection; the API does not use it for inference.")

    ms_ready = bool(health.get("ms_sta_ready"))
    ms_candidate = bool(health.get("ms_sta_candidate_ready"))
    ms_usable = ms_ready or ms_candidate
    if ms_ready:
        st.caption(
            f"MS-STA **defense-ready** · live engine `{health.get('engine')}` · "
            f"ckpt `{str(health.get('ms_sta_checkpoint_sha256') or '')[:12]}…`"
        )
    elif ms_candidate:
        st.caption(
            "MS-STA **D4 candidate** loaded (`ms_sta_candidate_ready=true`, not public GO). "
            "Operator-only."
        )
    else:
        st.caption(
            "MS-STA unavailable. Demo would fall back to **B0 Gate3 only** — "
            "not acceptable for defense presentation. Fix registry/ckpt load."
        )
    use_ms = st.checkbox(
        "Use MS-STA Unified Attention",
        value=bool(ms_ready),
        disabled=not ms_usable,
        help=(
            "Default ON when defense-ready. Uncheck only to contrast B0 Gate3. "
            "Live inference remains Unified Attention cache-backed (Protocol A); "
            "Optimized inference is evidence-only on this UI "
            "(Optimized-inference live switch not authorized)."
        ),
    )
    ret_attn = st.checkbox(
        "Show Unified Attention diagnostics",
        value=bool(ms_ready and use_ms),
        disabled=not ms_usable,
    )
    run = st.button("Run recognition", type="primary")

    if run:
        if not health.get("isolated_classifier_ready"):
            st.error("Classifier not ready.")
            st.stop()
        if use_ms and not ms_usable:
            st.error("MS-STA requested but not ready — uncheck or enable candidate/defense load.")
            st.stop()
        with st.spinner("Running inference…"):
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{API}/predict_clip",
                    json={
                        "video_id": str(row.video_id),
                        "mode": mode,
                        "top_k": 5,
                        "use_llm": True,
                        "use_ms_sta": bool(use_ms and mode == "isolated" and ms_usable),
                        "return_attention": bool(ret_attn and mode == "isolated" and ms_usable),
                    },
                    timeout=180,
                )
                elapsed = time.time() - t0
                if resp.status_code == 409:
                    st.error(f"Hệ thống dừng vì bằng chứng không khớp. Không trả nhãn. {resp.text}")
                    st.stop()
                if resp.status_code != 200:
                    st.error(resp.text)
                    st.stop()
                result = resp.json()
                result["_elapsed"] = elapsed
                result["_ref_vi"] = str(row.get("text_vietnamese") or "")
                result["_ref_gloss"] = str(row.get("text_gloss") or "")
                result["_clip_key"] = clip_key
                st.session_state.result_payload = result
                st.session_state.pop("confirm_result", None)
            except Exception as exc:
                st.error(f"API error: {exc}")

    payload = st.session_state.get("result_payload")
    if payload is not None and payload.get("_clip_key") == clip_key:
        st.markdown("---")
        st.subheader("4. Prediction vs GT")

        gt_gloss = payload.get("_ref_gloss") or ""
        gt_vi = payload.get("_ref_vi") or ""
        elapsed = float(payload.get("_elapsed") or 0)

        preds = payload.get("predictions") or []
        engine_used = str(payload.get("engine") or _engine_label(health, use_ms))
        st.info(f"**Inference engine:** `{engine_used}`")
        top1_hit, top5_hit, table = match_flags(preds, gt_gloss, gt_vi)
        m1, m2, m3 = st.columns(3)
        m1.metric("Top-1 hit", "Yes" if top1_hit else "No")
        m2.metric("Top-5 hit", "Yes" if top5_hit else "No")
        m3.metric("Latency", f"{elapsed:.2f}s")
        if top1_hit:
            st.success(f"Top-1 matches GT (`{gt_gloss}` / {gt_vi}).")
        elif top5_hit:
            st.warning(f"GT in Top-5 but not Top-1 (`{gt_gloss}` / {gt_vi}) — shortlist for human confirm.")
        else:
            st.error(f"GT not in Top-5 (`{gt_gloss}` / {gt_vi}) — do not confirm; show limitation.")
        if table:
            st.dataframe(table, width="stretch", hide_index=True)
        evidence = payload.get("input_evidence") or {}
        if evidence:
            with st.expander("Inputs consumed by Unified Attention", expanded=True):
                e1, e2, e3 = st.columns(3)
                e1.metric("RGB windows", evidence.get("rgb_windows"))
                e2.metric("Skeleton frames", evidence.get("skeleton_frames"))
                e3.metric("Valid joints", f"{100.0 * float(evidence.get('valid_joint_fraction') or 0):.1f}%")
                st.caption(
                    f"mode={evidence.get('mode')} · RGB={evidence.get('rgb_cache')} · "
                    f"Skeleton={evidence.get('skeleton_cache')}"
                )
        if payload.get("attention"):
            with st.expander("Unified Attention model diagnostics", expanded=False):
                st.caption(
                    "These are stream-gate/residual diagnostics, not spatial or temporal attention maps. "
                    f"engine={payload.get('engine')}"
                )
                diagnostics = payload["attention"]
                gate = diagnostics.get("stream_gate_rgb_skeleton")
                if isinstance(gate, list) and len(gate) == 2:
                    st.write({"RGB gate": round(float(gate[0]), 4), "Skeleton/cross gate": round(float(gate[1]), 4)})
                if diagnostics.get("residual_alpha") is not None:
                    st.metric("Residual alpha", f"{float(diagnostics['residual_alpha']):.4f}")
        suggestion = payload.get("suggestion_vietnamese_text") or (
            preds[0].get("vietnamese") if preds else ""
        )

        st.subheader("5. Human confirmation (required)")
        st.caption(payload.get("user_message") or "")
        if suggestion:
            edited = st.text_area(
                "Edit before confirm", value=str(suggestion), key="edit_result"
            )
            confirm = st.checkbox(
                "Xác nhận kết quả đã có người kiểm tra. Không tự gửi cho người dùng.",
                value=False,
                key="confirm_result",
            )
            if confirm:
                st.success(f"Confirmed: **{edited}**  ·  GT: {gt_vi}")
            else:
                st.caption("Not confirmed → not treated as assisted output.")
    elif payload is not None:
        st.info("Clip changed — click **Run recognition** again.")


# Streamlit runs this entry point as __main__; imports expose helpers only.
if __name__ == "__main__":
    main()
