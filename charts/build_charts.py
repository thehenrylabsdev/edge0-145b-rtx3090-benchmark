#!/usr/bin/env python3
"""Turn results/results.csv into ECharts specs (charts/specs/*.chart.json) and results/summary.json.

  python3 charts/build_charts.py            then   (cd charts && npm run render)

B* (the batch size the study is about) is computed here, from the medians, by the rule stated in
docs/methodology.md: the largest N whose median aggregate TPS is within 2 % of the peak median.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "charts", "specs")
WITHIN = 0.02

# Vendor's self-reported RTX 4090 numbers for the same Q3_K pack (Hugging Face model card speed table).
VENDOR_4090 = {"chat_decode": 69.9, "agent_decode": 91.7, "prefill_4k": 2042, "prefill_16k": 3179}

# Light theme, from the validated reference palette (dataviz skill references/palette.md).
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
REPO_URL = "github.com/thehenrylabsdev/edge0-145b-rtx3090-benchmark"


def load_rows() -> list[dict]:
    with open(os.path.join(ROOT, "results", "results.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else None
            except ValueError:
                pass
    return [r for r in rows if r["status"] == "ok"]


def decode(rows: list[dict], label: str) -> dict | None:
    return next((r for r in rows if r["label"] == label and r["mode"] == "decode"), None)


def rep_tps(label: str) -> list[float]:
    """Per-rep aggregate tok/s of a decode config, from its result.json."""
    try:
        res = json.load(open(os.path.join(ROOT, "results", "runs", label, "result.json")))
    except OSError:
        return []
    return [r["agg_tps"] for ph in res.get("phases", []) if ph["phase"]["mode"] == "decode"
            for r in ph["reps"] if r.get("agg_tps")]


def log_accept(label: str) -> dict | None:
    """Drafts accepted / offered from the engine log's per-request lines ("drafts accepted X of Y"), summed."""
    acc = off = 0
    try:
        for ln in open(os.path.join(ROOT, "results", "runs", label, "engine.log"), errors="replace"):
            m = re.search(r"drafts accepted (\d+) of (\d+)", ln)
            if m:
                acc, off = acc + int(m.group(1)), off + int(m.group(2))
    except OSError:
        return None
    return {"accepted": acc, "offered": off, "rate": acc / off if off else None}


def pooled(rows: list[dict], label: str) -> dict | None:
    """The decode row, with tps_median/min/max over its reps plus any confirm-<label> re-run."""
    r = decode(rows, label)
    extra = rep_tps(f"confirm-{label}")
    if not r or not extra:
        return r
    vals = rep_tps(label) + extra
    return {**r, "tps_median": statistics.median(vals), "tps_min": min(vals), "tps_max": max(vals),
            "reps": len(vals), "first_pass_median": r["tps_median"],
            "confirm_median": statistics.median(extra)}


def vendor_tool() -> dict:
    """The vendor's own tools/strata_bench.py run on the shipped config (Epic 2)."""
    path = glob.glob(os.path.join(ROOT, "results", "runs", "base-model", "strata_bench", "*", "results.json"))[0]
    r = json.load(open(path))["results"]
    return {"chat_decode": r["chat"]["decode_tps"]["median"], "agent_decode": r["agent"]["turn_decode_tps"]["median"],
            "prefill_4k": r["prompts"]["4K"]["prompt_tps"]["median"],
            "prefill_16k": r["prompts"]["16K"]["prompt_tps"]["median"],
            "prefill_4k_tokens": r["prompts"]["4K"]["prompt_tokens"]["median"],
            "prefill_16k_tokens": r["prompts"]["16K"]["prompt_tokens"]["median"]}


def b_star(points: dict[int, float]) -> tuple[int, int, float]:
    """(B*, N at peak, peak median) for {N: median aggregate TPS}."""
    peak_n = max(points, key=lambda n: points[n])
    peak = points[peak_n]
    return max(n for n, v in points.items() if v >= (1 - WITHIN) * peak), peak_n, peak


def base_option(title: str, subtitle: str, width: int = 1200, height: int = 675) -> dict:
    return {
        "width": width, "height": height, "backgroundColor": SURFACE,
        "textStyle": {"fontFamily": FONT, "color": INK},
        "title": {"text": title, "subtext": subtitle, "left": 40, "top": 28,
                  "textStyle": {"fontSize": 26, "fontWeight": "bold", "color": INK, "fontFamily": FONT},
                  "subtextStyle": {"fontSize": 15, "color": INK_2, "fontFamily": FONT, "lineHeight": 22}},
        "graphic": [{"type": "text", "right": 40, "bottom": 18,
                     "style": {"text": REPO_URL, "fill": MUTED, "fontSize": 13, "fontFamily": FONT}}],
    }


def value_axis(name: str, **kw) -> dict:
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 52,
            "nameTextStyle": {"color": INK_2, "fontSize": 14}, "axisLabel": {"color": INK_2, "fontSize": 13},
            "splitLine": {"lineStyle": {"color": GRID}}, "axisLine": {"show": False}, **kw}


def category_axis(name: str, data: list, **kw) -> dict:
    return {"type": "category", "name": name, "nameLocation": "middle", "nameGap": 40, "data": data,
            "nameTextStyle": {"color": INK_2, "fontSize": 14}, "axisLabel": {"color": INK, "fontSize": 13},
            "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": MUTED}}, **kw}


def line(name: str, color: str, data: list, labels: bool = True, dashed: bool = False) -> dict:
    return {"name": name, "type": "line", "data": data, "symbol": "circle", "symbolSize": 10, "connectNulls": False,
            "lineStyle": {"width": 2.5, "color": color, "type": "dashed" if dashed else "solid"},
            "itemStyle": {"color": color, "borderColor": SURFACE, "borderWidth": 2},
            "label": {"show": labels, "position": "top", "color": INK, "fontSize": 13, "fontWeight": "bold",
                      "formatter": "{@[1]}"}}


def legend(names: list[str], top: int = 112) -> dict:
    return {"data": names, "left": 40, "top": top, "width": 1100, "itemGap": 28, "icon": "roundRect",
            "itemWidth": 18, "itemHeight": 6, "textStyle": {"color": INK, "fontSize": 14}}


def main() -> None:
    rows = load_rows()
    os.makedirs(SPECS, exist_ok=True)
    one = decode(rows, "batch-1")
    if one is None:
        raise SystemExit("results.csv has no batch-1 decode row (the N = 1 reference)")

    # ---- decode batch families: {N: row}
    plain = {1: one, **{n: pooled(rows, f"batch-{n}") for n in range(2, 9)}}
    plain = {n: r for n, r in plain.items() if r}
    bmtp = {1: one, **{n: decode(rows, f"batchmtp-{n}") for n in range(2, 9)}}
    bmtp = {n: r for n, r in bmtp.items() if r}
    nohead = {n: decode(rows, f"batch-nomtp-{n}") for n in range(2, 9)}
    nohead = {n: r for n, r in nohead.items() if r}
    kv8k = {n: decode(rows, f"kvres8k-{n}") for n in range(2, 9)}
    kv8k = {n: r for n, r in kv8k.items() if r}
    combo = {n: decode(rows, f"nomtp-kvres8k-{n}") for n in range(2, 9)}
    combo = {n: r for n, r in combo.items() if r}
    longp = {n: decode(rows, f"long-{n}") for n in range(2, 9)}
    longp = {n: r for n, r in longp.items() if r}

    bs, peak_n, peak = b_star({n: r["tps_median"] for n, r in plain.items()})
    every = {}  # best variant at each N across all decode families
    for fam, pts in (("--batch N", plain), ("--batch N --batch-mtp", bmtp), ("--batch N, MTP head removed", nohead),
                     ("--batch N --kv-resident 8192", kv8k),
                     ("--batch N, MTP head removed, --kv-resident 8192", combo)):
        for n, r in pts.items():
            if n not in every or r["tps_median"] > every[n][1]["tps_median"]:
                every[n] = (fam, r)
    bs_all, peak_n_all, peak_all = b_star({n: fr[1]["tps_median"] for n, fr in every.items()})

    keep = ("label", "tps_median", "tps_min", "tps_max", "reps", "first_pass_median", "confirm_median",
            "per_stream_tps", "tpot_p50_ms", "ttft_p50_s", "accept_rate", "expert_slots", "slots_reported",
            "engine_spec", "drafting")

    def pt(r: dict) -> dict:
        return {k: r.get(k) for k in keep} | {"log_accept_rate": log_accept(r["label"])}

    summary = {
        "b_star_rule": f"largest N whose median aggregate TPS is within {WITHIN:.0%} of the peak median",
        "b_star_plain": {"b_star": bs, "peak_n": peak_n, "peak_tps": peak},
        "b_star_any_variant": {"b_star": bs_all, "peak_n": peak_n_all, "peak_tps": peak_all,
                               "family_at_peak": every[peak_n_all][0], "label_at_peak": every[peak_n_all][1]["label"]},
        "families": {name: {str(n): pt(r) for n, r in sorted(fam.items())} for name, fam in
                     (("plain", plain), ("batch_mtp", bmtp), ("mtp_head_removed", nohead), ("kv_resident_8192", kv8k),
                      ("mtp_head_removed_kv_resident_8192", combo), ("long_prompt", longp))},
        "spec": {lab: pt(r) for lab in ("spec-2", "spec-3", "spec-4", "spec-6", "spec-nomtp", "base-model",
                                         "base-modelbench") if (r := decode(rows, lab))},
        # keyed "label|mode": one prefill config has two prompt lengths
        "prefill": {f"{r['label']}|{r['mode']}": {"tps_median": r["tps_median"], "tps_min": r["tps_min"],
                                                  "tps_max": r["tps_max"], "prompt_tokens": r["prompt_tokens"]}
                    for r in rows if r["mode"].startswith("prefill")},
        "vendor_tool_3090": vendor_tool(), "vendor_4090_self_reported": VENDOR_4090,
    }
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)

    slots = [str(n) for n in range(1, 9)]
    y_max = 5 * math.ceil((max(peak, peak_all) + 3) / 5)  # headroom so the B* label clears the peak's label

    def series_points(fam: dict) -> list:
        return [round(fam[n]["tps_median"], 1) if n in fam else None for n in range(1, 9)]

    # ---- 1. batch curve
    best_fam = every[peak_n_all][0]
    opt = base_option("Aggregate decode throughput vs. batch slots",
                      "Edge0-145B Q3_K, one RTX 3090 · closed loop, N requests in flight · median of 3 × 45 s · "
                      "gaps = not measured\n"
                      f"B* = {bs_all} slots across all variants ({best_fam}, peak {peak_all:.1f} tok/s); "
                      f"plain --batch N alone: B* = {bs} (its peak {peak:.1f} tok/s at N = {peak_n})")
    fams = [("--batch N (N = 1: MTP on; N ≥ 2: MTP drafting off)", BLUE, plain, False),
            ("--batch N --batch-mtp", ORANGE, bmtp, True),
            ("--batch N, MTP head removed", AQUA, nohead, True),
            ("--batch N --kv-resident 8192", YELLOW, kv8k, True),
            ("MTP head removed + --kv-resident 8192", MAGENTA, combo, True)]
    fams = [f for f in fams if f[2] and (f[2] is plain or any(n > 1 for n in f[2]))]
    family_key = {"--batch N": plain, "--batch N --batch-mtp": bmtp, "--batch N, MTP head removed": nohead,
                  "--batch N --kv-resident 8192": kv8k, "--batch N, MTP head removed, --kv-resident 8192": combo}
    best_pts = family_key[best_fam]
    opt.update({
        "legend": legend([f[0] for f in fams], top=118),
        "grid": {"left": 90, "right": 50, "top": 236, "bottom": 90},
        "xAxis": category_axis("Batch slots (N)", slots, boundaryGap=False),
        "yAxis": value_axis("Aggregate output tok/s", min=30, max=y_max, nameGap=48),
        "series": [line(name, c, series_points(fam), labels=fam is best_pts or (fam is plain and not best_pts),
                        dashed=d) for name, c, fam, d in fams],
    })
    lead = next(i for i, f in enumerate(fams) if f[2] is best_pts) if any(f[2] is best_pts for f in fams) else 0
    opt["series"][lead]["markLine"] = {
        "symbol": "none", "silent": True, "lineStyle": {"color": INK_2, "type": "dotted", "width": 1.5},
        "label": {"formatter": f"B* = {bs_all}", "color": INK, "fontSize": 14, "fontWeight": "bold"},
        "data": [{"xAxis": str(bs_all)}]}
    json.dump(opt, open(os.path.join(SPECS, "batch-curve.chart.json"), "w"), indent=1)

    # ---- 2. per-stream cost of batching
    opt = base_option("Per-request speed falls as slots are added",
                      "Median per-stream decode tok/s (each request's output tokens over its own decode time)")
    opt.update({
        "grid": {"left": 90, "right": 50, "top": 130, "bottom": 90},
        "xAxis": category_axis("Batch slots (N), --batch N", [str(n) for n in sorted(plain)]),
        "yAxis": value_axis("Per-stream tok/s", nameGap=48),
        "series": [{"type": "bar", "barWidth": "45%",
                    "data": [round(plain[n]["per_stream_tps"], 1) for n in sorted(plain)],
                    "itemStyle": {"color": BLUE, "borderRadius": [4, 4, 0, 0]},
                    "label": {"show": True, "position": "top", "color": INK, "fontSize": 13, "fontWeight": "bold"}}],
    })
    json.dump(opt, open(os.path.join(SPECS, "batch-per-stream.chart.json"), "w"), indent=1)

    # ---- 3. MTP verify window
    spec_labels = [("spec-2", "--spec 2"), ("spec-3", "--spec 3"), ("spec-4", "--spec 4\n(shipped)"),
                   ("spec-6", "--spec 6"), ("spec-nomtp", "MTP head\nremoved")]
    spec_rows = [(r, name) for lab, name in spec_labels if (r := decode(rows, lab))]
    opt = base_option("MTP verify window: the shortest window wins on a 3090",
                      "N = 1, median aggregate tok/s, label = draft acceptance · the MTP-head-removed run keeps "
                      "--spec 4, so the engine's n-gram lookup still drafts")
    opt.update({
        "grid": {"left": 90, "right": 50, "top": 130, "bottom": 100},
        "xAxis": category_axis("", [name for _, name in spec_rows], nameGap=0),
        "yAxis": value_axis("Aggregate output tok/s", min=0, nameGap=48),
        "series": [{"type": "bar", "barWidth": "50%",
                    "data": [{"value": round(r["tps_median"], 1),
                              "label": {"formatter": f"{r['tps_median']:.1f}\n{(r['accept_rate'] or 0):.0%} acc."},
                              "itemStyle": {"color": BLUE if i == 0 else "#9ec5f4", "borderRadius": [4, 4, 0, 0]}}
                             for i, (r, _) in enumerate(spec_rows)],
                    "label": {"show": True, "position": "top", "color": INK, "fontSize": 13, "lineHeight": 18}}],
    })
    json.dump(opt, open(os.path.join(SPECS, "spec-window.chart.json"), "w"), indent=1)

    # ---- 4. prefill chunk
    chunks = ["256", "512", "1024", "2048", "4096", "8192", "auto"]

    def pf(c: str, mode: str) -> float | None:
        r = next((r for r in rows if r["label"] == f"prefill-{c}" and r["mode"] == mode), None)
        return round(r["tps_median"], 0) if r else None

    def prompt_name(mode: str, fallback: str) -> str:
        t = next((r["prompt_tokens"] for r in rows if r["label"] == "prefill-auto" and r["mode"] == mode), None)
        return f"~{t / 1000:.1f}K-token prompt" if t else fallback

    n4, n16 = prompt_name("prefill-4096", "4K prompt"), prompt_name("prefill-16384", "16K prompt")
    opt = base_option("Prefill throughput vs. --prefill chunk",
                      "Median prompt tok/s over 3 fresh prompts · the engine reports prefill_cap 8192 for auto")
    opt.update({
        "legend": legend([n4, n16], top=100),
        "grid": {"left": 90, "right": 50, "top": 150, "bottom": 90},
        "xAxis": category_axis("--prefill chunk (tokens)", chunks),
        "yAxis": value_axis("Prompt tok/s", min=0, nameGap=56),
        "series": [{"name": n, "type": "bar", "barGap": "8%", "barWidth": "32%", "data": [pf(c, m) for c in chunks],
                    "itemStyle": {"color": col, "borderRadius": [4, 4, 0, 0]},
                    "label": {"show": True, "position": "top", "color": INK, "fontSize": 11}}
                   for n, m, col in ((n4, "prefill-4096", BLUE), (n16, "prefill-16384", ORANGE))],
    })
    json.dump(opt, open(os.path.join(SPECS, "prefill-chunk.chart.json"), "w"), indent=1)

    # ---- 5. 3090 vs vendor 4090 (same tool, same pack; 4090 numbers self-reported by the vendor)
    vt = summary["vendor_tool_3090"]
    pct = {k: vt[k] / VENDOR_4090[k] for k in VENDOR_4090}
    names = ["Chat decode", "Agent decode", "4K prefill", "16K prefill"]
    keys = ["chat_decode", "agent_decode", "prefill_4k", "prefill_16k"]
    opt = base_option("RTX 3090 vs. the vendor's RTX 4090 numbers",
                      "Same Q3_K pack and the vendor's own tools/strata_bench.py · 3090 measured here, "
                      "4090 self-reported on the model card\n"
                      "% = 3090 as a share of the 4090 figure · prompts of 4,019 and 16,058 tokens")
    opt.update({
        "legend": legend(["RTX 3090 (this run)", "RTX 4090 (vendor, self-reported)"], top=118),
        "grid": [{"left": 90, "right": 640, "top": 170, "bottom": 90}, {"left": 660, "right": 50, "top": 170,
                                                                        "bottom": 90}],
        "xAxis": [category_axis("", names[:2], gridIndex=0), category_axis("", names[2:], gridIndex=1)],
        "yAxis": [value_axis("Decode tok/s", gridIndex=0, min=0, nameGap=44),
                  value_axis("Prefill tok/s", gridIndex=1, min=0, nameGap=56)],
        "series": [],
    })
    for i, ks in enumerate((keys[:2], keys[2:])):
        for name, col, src in (("RTX 3090 (this run)", BLUE, vt), ("RTX 4090 (vendor, self-reported)", "#b7b6b0",
                                                                   VENDOR_4090)):
            opt["series"].append({
                "name": name, "type": "bar", "xAxisIndex": i, "yAxisIndex": i, "barGap": "8%", "barWidth": "30%",
                "data": [{"value": round(src[k], 1),
                          "label": {"formatter": (f"{src[k]:.0f}" if src[k] > 200 else f"{src[k]:.1f}") +
                                    (f"\n{pct[k]:.0%}" if src is vt else "")}} for k in ks],
                "itemStyle": {"color": col, "borderRadius": [4, 4, 0, 0]},
                "label": {"show": True, "position": "top", "color": INK, "fontSize": 13, "lineHeight": 18}})
    json.dump(opt, open(os.path.join(SPECS, "vs-4090.chart.json"), "w"), indent=1)

    # ---- 6. social cards: best configuration against plain --batch N
    gain = peak_all / one["tps_median"] - 1
    for name, w, h in (("social-card", 1200, 675), ("social-card-square", 1080, 1080)):
        square = h == w
        chart_top = 500 if square else 262
        txt = [
            {"type": "text", "left": 48, "top": 36,
             "style": {"text": "Edge0-145B on a single RTX 3090", "fill": INK, "fontSize": 40, "fontWeight": "bold",
                       "fontFamily": FONT}},
            {"type": "text", "left": 48, "top": 92,
             "style": {"text": "145B-parameter latent MoE (~10B active) · Q3_K · 24 GB VRAM · edge0-strata (ggml)",
                       "fill": INK_2, "fontSize": 18, "fontFamily": FONT}},
        ]
        stats = [(f"B* = {bs_all}", "requests batched for the\nhighest tokens/s"),
                 (f"{peak_all:.1f}", "tok/s aggregate at B*\n(MTP head off, 8K KV cells)"),
                 (f"+{gain:.0%}", f"vs one request at a time\n({one['tps_median']:.1f} tok/s)"),
                 (f"{vt['prefill_16k']:.0f}", "tok/s prefill on a\n16K-token prompt")]
        cols = 2 if square else 4
        cw = (w - 96) / cols
        for i, (big, small) in enumerate(stats):
            x = 48 + (i % cols) * cw
            y = 140 + (i // cols) * 160
            txt += [{"type": "text", "left": x, "top": y,
                     "style": {"text": big, "fill": BLUE, "fontSize": 44, "fontWeight": "bold", "fontFamily": FONT}},
                    {"type": "text", "left": x, "top": y + 56,
                     "style": {"text": small, "fill": INK_2, "fontSize": 16, "lineHeight": 21, "fontFamily": FONT}}]
        txt.append({"type": "text", "left": 48, "bottom": 22,
                    "style": {"text": f"Method, scripts and raw data: {REPO_URL}", "fill": INK, "fontSize": 16,
                              "fontFamily": FONT}})
        card_fams = [("--batch N (shipped config, --spec 2)", BLUE, plain, False, False),
                     ("MTP head removed", AQUA, nohead, False, True),
                     ("MTP head removed + --kv-resident 8192", MAGENTA, combo, True, False)]
        opt = {"width": w, "height": h, "backgroundColor": SURFACE, "textStyle": {"fontFamily": FONT, "color": INK},
               "graphic": txt,
               "title": {"text": "Aggregate tok/s vs. requests batched (--batch N)", "left": 48, "top": chart_top,
                         "textStyle": {"fontSize": 18, "color": INK, "fontWeight": "bold", "fontFamily": FONT}},
               "legend": {**legend([f[0] for f in card_fams], top=chart_top + 30), "width": w - 96},
               "grid": {"left": 70, "right": 48, "top": chart_top + 92, "bottom": 92},
               "xAxis": category_axis("Batch slots (N)", slots, boundaryGap=False, nameGap=30),
               "yAxis": value_axis("", min=40, max=y_max),
               "series": [line(n, c, series_points(f), labels=lab, dashed=d) for n, c, f, lab, d in card_fams]}
        opt["series"][2]["markLine"] = {
            "symbol": "none", "silent": True, "lineStyle": {"color": INK_2, "type": "dotted", "width": 1.5},
            "label": {"formatter": f"B* = {bs_all}", "color": INK, "fontSize": 14, "fontWeight": "bold"},
            "data": [{"xAxis": str(bs_all)}]}
        json.dump(opt, open(os.path.join(SPECS, f"{name}.chart.json"), "w"), indent=1)

    print(json.dumps({"b_star_plain": summary["b_star_plain"], "b_star_any_variant": summary["b_star_any_variant"]}))


if __name__ == "__main__":
    main()
