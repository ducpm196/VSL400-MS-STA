"""Compact visual primitives for the B4/E4 defense pages."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Sequence


_SERIES_COLORS = {
    "B4-V6": "#22c55e",
    "B3 CTR concat": "#38bdf8",
    "B3+MLP CTR": "#f59e0b",
    "RGB": "#94a3b8",
    "E4 w192": "#a78bfa",
    "E4-R2 D": "#f87171",
    "E4-R3 A": "#38bdf8",
    "E4-R3 B": "#f59e0b",
    "E4-R3 C": "#f87171",
}


def _streamlit() -> Any:
    """Keep chart-data helpers importable in the hermetic CI environment."""
    import streamlit

    return streamlit


def grouped_accuracy_records(
    *, seeds: Sequence[int], series: dict[str, Sequence[float]]
) -> list[dict[str, float | int | str]]:
    """Build chart rows and reject partially aligned scientific series."""
    if not seeds:
        raise ValueError("seeds_must_not_be_empty")
    rows: list[dict[str, float | int | str]] = []
    for name, values in series.items():
        if len(values) != len(seeds):
            raise ValueError(f"series_length_mismatch:{name}")
        for seed, value in zip(seeds, values, strict=True):
            rows.append({"Seed": int(seed), "Model": str(name), "Top-1 (%)": float(value)})
    return rows


def relative_latency_change_pct(*, baseline_ms: float, candidate_ms: float) -> float:
    """Return candidate-vs-baseline latency change; positive means slower."""
    if baseline_ms <= 0.0 or candidate_ms <= 0.0:
        raise ValueError("latency_must_be_positive")
    return 100.0 * (candidate_ms / baseline_ms - 1.0)


def render_grouped_accuracy_chart(
    rows: list[dict[str, float | int | str]],
    *,
    title: str,
    y_domain: tuple[float, float],
) -> None:
    st = _streamlit()
    models = list(dict.fromkeys(str(row["Model"]) for row in rows))
    colors = [_SERIES_COLORS.get(model, "#cbd5e1") for model in models]
    st.vega_lite_chart(
        rows,
        {
            "title": {"text": title, "anchor": "start", "fontSize": 16},
            "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
            "encoding": {
                "x": {"field": "Seed", "type": "ordinal", "title": "Seed"},
                "xOffset": {"field": "Model"},
                "y": {
                    "field": "Top-1 (%)",
                    "type": "quantitative",
                    "scale": {"domain": list(y_domain), "zero": False},
                    "title": "Top-1 (%)",
                },
                "color": {
                    "field": "Model",
                    "type": "nominal",
                    "scale": {"domain": models, "range": colors},
                    "legend": {"orient": "bottom", "title": None},
                },
                "tooltip": [
                    {"field": "Model", "type": "nominal"},
                    {"field": "Seed", "type": "ordinal"},
                    {"field": "Top-1 (%)", "type": "quantitative", "format": ".2f"},
                ],
            },
            "height": 300,
        },
        width="stretch",
    )


def render_test_comparison_chart(
    rows: Iterable[dict[str, float | str]],
    *,
    x_domain: tuple[float, float] = (0.0, 60.0),
    x_title: str = "Test Top-1 (%)",
) -> None:
    st = _streamlit()
    data = list(rows)
    models = [str(row["Model"]) for row in data]
    colors = [_SERIES_COLORS.get(model, "#cbd5e1") for model in models]
    tooltip: list[dict[str, str]] = [
        {"field": "Model", "type": "nominal"},
        {"field": "Top-1 (%)", "type": "quantitative", "format": ".2f"},
    ]
    if data and all("Top-5 (%)" in row for row in data):
        tooltip.append({"field": "Top-5 (%)", "type": "quantitative", "format": ".2f"})
    st.vega_lite_chart(
        data,
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {
                    "field": "Model",
                    "type": "nominal",
                    "sort": "-x",
                    "title": None,
                },
                "x": {
                    "field": "Top-1 (%)",
                    "type": "quantitative",
                    "scale": {"domain": list(x_domain), "zero": x_domain[0] == 0.0},
                    "title": x_title,
                },
                "color": {
                    "field": "Model",
                    "type": "nominal",
                    "scale": {"domain": models, "range": colors},
                    "legend": None,
                },
                "tooltip": tooltip,
            },
            "height": 250,
        },
        width="stretch",
    )


def render_delta_ci(*, mean: float, low: float, high: float, threshold: float) -> None:
    st = _streamlit()
    rows = [{"label": "E4-R2 - B4", "mean": mean, "low": low, "high": high}]
    st.vega_lite_chart(
        rows,
        {
            "layer": [
                {
                    "mark": {"type": "rule", "strokeWidth": 7, "color": "#f87171"},
                    "encoding": {
                        "y": {"field": "label", "type": "nominal", "title": None},
                        "x": {"field": "low", "type": "quantitative"},
                        "x2": {"field": "high"},
                    },
                },
                {
                    "mark": {"type": "point", "filled": True, "size": 130, "color": "#ffffff"},
                    "encoding": {
                        "y": {"field": "label", "type": "nominal", "title": None},
                        "x": {
                            "field": "mean",
                            "type": "quantitative",
                            "title": "Chênh lệch Top-1 (điểm phần trăm)",
                        },
                        "tooltip": [
                            {"field": "mean", "type": "quantitative", "format": ".2f"},
                            {"field": "low", "type": "quantitative", "format": ".2f"},
                            {"field": "high", "type": "quantitative", "format": ".2f"},
                        ],
                    },
                },
                {
                    "data": {"values": [{"threshold": threshold}]},
                    "mark": {"type": "rule", "strokeDash": [5, 4], "color": "#f59e0b"},
                    "encoding": {"x": {"field": "threshold", "type": "quantitative"}},
                },
            ],
            "height": 105,
        },
        width="stretch",
    )


def render_latency_screening_chart(*, b4_ms: float, e4_ms: float) -> None:
    st = _streamlit()
    rows = [
        {"Model": "B4-V6", "Median latency (ms)": float(b4_ms)},
        {"Model": "E4-R2 D", "Median latency (ms)": float(e4_ms)},
    ]
    st.vega_lite_chart(
        rows,
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": "Model", "type": "nominal", "title": None},
                "x": {
                    "field": "Median latency (ms)",
                    "type": "quantitative",
                    "scale": {"domain": [0, max(b4_ms, e4_ms) * 1.15]},
                    "title": "Median model-only latency (ms) — screening",
                },
                "color": {
                    "field": "Model",
                    "type": "nominal",
                    "scale": {
                        "domain": ["B4-V6", "E4-R2 D"],
                        "range": [_SERIES_COLORS["B4-V6"], _SERIES_COLORS["E4-R2 D"]],
                    },
                    "legend": None,
                },
                "tooltip": [
                    {"field": "Model", "type": "nominal"},
                    {"field": "Median latency (ms)", "type": "quantitative", "format": ".2f"},
                ],
            },
            "height": 145,
        },
        width="stretch",
    )


def render_e4_r3_latency_arms_chart(
    rows: Sequence[dict[str, float | str]],
    *,
    threshold_speedup: float = 0.08,
) -> None:
    """Bar chart of B4/A/B/C medians with dashed 8% faster-than-B4 threshold."""
    st = _streamlit()
    data = list(rows)
    b4 = next(float(r["Median latency (ms)"]) for r in data if r["Model"] == "B4-V6")
    threshold_ms = b4 / (1.0 + float(threshold_speedup))
    models = [str(r["Model"]) for r in data]
    colors = [_SERIES_COLORS.get(m, "#cbd5e1") for m in models]
    ymax = max(float(r["Median latency (ms)"]) for r in data) * 1.18
    st.vega_lite_chart(
        {
            "layer": [
                {
                    "data": {"values": data},
                    "mark": {"type": "bar", "cornerRadiusEnd": 3},
                    "encoding": {
                        "y": {
                            "field": "Model",
                            "type": "nominal",
                            "sort": models,
                            "title": None,
                        },
                        "x": {
                            "field": "Median latency (ms)",
                            "type": "quantitative",
                            "scale": {"domain": [0, ymax]},
                            "title": "Median model-forward latency (ms)",
                        },
                        "color": {
                            "field": "Model",
                            "type": "nominal",
                            "scale": {"domain": models, "range": colors},
                            "legend": None,
                        },
                        "tooltip": [
                            {"field": "Model", "type": "nominal"},
                            {
                                "field": "Median latency (ms)",
                                "type": "quantitative",
                                "format": ".2f",
                            },
                            {"field": "vs B4", "type": "nominal"},
                        ],
                    },
                },
                {
                    "data": {"values": [{"threshold": threshold_ms}]},
                    "mark": {
                        "type": "rule",
                        "strokeDash": [6, 4],
                        "color": "#f59e0b",
                        "strokeWidth": 2,
                    },
                    "encoding": {"x": {"field": "threshold", "type": "quantitative"}},
                },
            ],
            "height": 220,
        },
        width="stretch",
    )


def render_e4_r3_param_chart(rows: Sequence[dict[str, float | str | int]]) -> None:
    st = _streamlit()
    data = list(rows)
    models = [str(r["Model"]) for r in data]
    colors = [_SERIES_COLORS.get(m, "#cbd5e1") for m in models]
    st.vega_lite_chart(
        data,
        {
            "title": {
                "text": "Parameter count (trainable)",
                "anchor": "start",
                "fontSize": 15,
            },
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {
                    "field": "Model",
                    "type": "nominal",
                    "sort": models,
                    "title": None,
                },
                "x": {
                    "field": "Parameters (M)",
                    "type": "quantitative",
                    "title": "Parameters (millions)",
                },
                "color": {
                    "field": "Model",
                    "type": "nominal",
                    "scale": {"domain": models, "range": colors},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "Model", "type": "nominal"},
                    {"field": "Parameters (M)", "type": "quantitative", "format": ".2f"},
                    {"field": "Reduction vs B4", "type": "nominal"},
                ],
            },
            "height": 200,
        },
        width="stretch",
    )


def render_flow(nodes: Sequence[tuple[str, str]], *, accent: str = "#22c55e") -> None:
    st = _streamlit()
    blocks: list[str] = []
    for index, (title, detail) in enumerate(nodes):
        blocks.append(
            '<div class="mssta-flow-node">'
            f'<div class="mssta-flow-title">{escape(title)}</div>'
            f'<div class="mssta-flow-detail">{escape(detail)}</div>'
            "</div>"
        )
        if index < len(nodes) - 1:
            blocks.append('<div class="mssta-flow-arrow">&#8594;</div>')
    st.markdown(
        f"""
<style>
.mssta-flow {{ display:flex; align-items:stretch; gap:8px; width:100%; overflow-x:auto; padding:8px 0 14px; }}
.mssta-flow-node {{ min-width:150px; flex:1; border:1px solid #334155; border-top:3px solid {accent}; border-radius:4px; padding:12px; background:#111827; }}
.mssta-flow-title {{ color:#f8fafc; font-weight:700; font-size:0.95rem; }}
.mssta-flow-detail {{ color:#aab4c3; font-size:0.82rem; margin-top:5px; line-height:1.35; }}
.mssta-flow-arrow {{ display:flex; align-items:center; color:#94a3b8; font-size:1.4rem; }}
@media (max-width: 760px) {{ .mssta-flow {{ flex-direction:column; }} .mssta-flow-arrow {{ transform:rotate(90deg); justify-content:center; height:18px; }} }}
</style>
<div class="mssta-flow">{''.join(blocks)}</div>
""",
        unsafe_allow_html=True,
    )
