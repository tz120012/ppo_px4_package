#!/usr/bin/env python3
"""
使用命令：
python auto_weight_from_csv.py \
  --csv ./log_step_metrics/.csv \
  --index step \
  --outdir auto_weights \
  --topk 16 \
  --huber-mode iqr
"""


import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================
# 工具函数
# ============================
def robust_stats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan,
                    mean=np.nan, std=np.nan, p95=np.nan, p99=np.nan,
                    max=np.nan, min=np.nan)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    return dict(
        median=float(np.median(x)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr if iqr > 1e-12 else 1e-12),
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        p95=float(np.percentile(x, 95)),
        p99=float(np.percentile(x, 99)),
        max=float(np.max(x)),
        min=float(np.min(x)),
    )

def spike_rate(x: pd.Series, p95: float, factor: float = 1.5) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return 0.0
    thr = factor * p95
    return float((x > thr).mean())

def suggest_huber_delta(stats: Dict[str, float], mode: str = "iqr") -> float:
    if mode == "p95":
        base = stats["p95"]
        delta = 0.25 * base if base > 0 else 1.0
    else:
        base = stats["iqr"]
        delta = 0.5 * base if base > 0 else 1.0
    return float(np.clip(delta, 1e-4, 1e6))

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(0.0, v) for v in scores.values()], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        n = len(vals) if len(vals) > 0 else 1
        return {k: 1.0 / n for k in scores}
    vals = vals / s
    return {k: float(v) for k, v in zip(scores.keys(), vals)}

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std()
    if sd is None or sd < 1e-12:
        sd = 1.0
    return (x - mu) / sd

def corr_importance(df: pd.DataFrame, target: str, cols: List[str]) -> Dict[str, float]:
    imp = {}
    if target not in df.columns:
        return {c: 0.0 for c in cols}
    y = zscore(df[target])
    for c in cols:
        try:
            x = zscore(df[c])
            corr = float(pd.concat([x, y], axis=1).corr().iloc[0, 1])
            if np.isnan(corr):
                corr = 0.0
            imp[c] = abs(corr)
        except Exception:
            imp[c] = 0.0
    return imp

# ============================
# 评分策略
# ============================
def score_columns(df: pd.DataFrame, cols: List[str], reward_col_candidates: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    eps = 1e-9
    details = {}

    for c in cols:
        s = robust_stats(df[c])
        s["spike_rate"] = spike_rate(df[c], s["p95"], factor=1.5)
        s["huber_delta_iqr"] = suggest_huber_delta(s, mode="iqr")
        s["huber_delta_p95"] = suggest_huber_delta(s, mode="p95")
        details[c] = s

    reward_col = None
    for cand in reward_col_candidates:
        if cand in df.columns:
            reward_col = cand
            break
    corr_imp = corr_importance(df, reward_col, cols) if reward_col else {c: 0.0 for c in cols}

    iqr_vals = np.array([details[c]["iqr"] for c in cols], dtype=float)
    p95_vals = np.array([details[c]["p95"] for c in cols], dtype=float)
    iqr_scale = (np.log(iqr_vals + eps) - np.log(iqr_vals + eps).min()) / ((np.log(iqr_vals + eps).ptp() + eps))
    p95_scale = (p95_vals - p95_vals.min()) / (p95_vals.ptp() + eps)

    spikes = np.array([details[c]["spike_rate"] for c in cols], dtype=float)
    if np.isnan(spikes).any():
        spikes = np.nan_to_num(spikes, nan=0.0)
    spike_scale = (spikes - spikes.min()) / (spikes.ptp() + eps) if spikes.ptp() >= eps else np.ones_like(spikes) * 0.5

    corr_vals = np.array([corr_imp[c] for c in cols], dtype=float)
    corr_scale = (corr_vals - corr_vals.min()) / (corr_vals.ptp() + eps) if corr_vals.ptp() >= eps else np.ones_like(corr_vals) * 0.5

    a_scale, b_spike, c_corr = 0.35, 0.25, 0.40
    raw_scores = a_scale * (0.5 * iqr_scale + 0.5 * p95_scale) + b_spike * spike_scale + c_corr * corr_scale

    penalty = 1.0 - 0.3 * (spike_scale * (0.5 * iqr_scale + 0.5 * p95_scale))
    penalty = np.clip(penalty, 0.6, 1.0)

    final_scores = raw_scores * penalty
    scores = {c: float(v) for c, v in zip(cols, final_scores)}
    return scores, details

def suggest_weights(scores: Dict[str, float], prefer: Dict[str, float] = None) -> Dict[str, float]:
    adj = {}
    for k, v in scores.items():
        bias = 1.0
        if prefer:
            low = k.lower()
            for key, fac in prefer.items():
                if key in low:
                    bias *= fac
        adj[k] = max(0.0, v) * bias
    return normalize_scores(adj)

# ============================
# 列选择辅助
# ============================
REG_KEYS = ["mean_control_effort", "rms_du", "rms_domega", "rms_dv"]
EVENT_OVER = "overshoot_hits"
EVENT_SAT = "saturation_hits"

def pick_columns_by_keywords(df: pd.DataFrame, keys: List[str]) -> List[str]:
    cols = []
    lowmap = {c.lower(): c for c in df.columns}
    for k in keys:
        name = lowmap.get(k.lower())
        if name and pd.api.types.is_numeric_dtype(df[name]):
            cols.append(name)
    return cols

def suggest_lambda_reg(df: pd.DataFrame, reg_cols: List[str], reward_col: str | None) -> float:
    if not reg_cols or (reward_col is None) or (reward_col not in df.columns):
        return 0.4
    y = zscore(df[reward_col])
    corrs = []
    for c in reg_cols:
        x = zscore(df[c])
        r = float(pd.concat([x, y], axis=1).corr().iloc[0, 1])
        if not np.isnan(r):
            corrs.append(abs(r))
    if not corrs:
        return 0.4
    m = np.clip(np.mean(corrs), 0.0, 1.0)
    return float(0.3 + 0.3 * m)  # 映射到 [0.3, 0.6]

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# ============================
# 主流程
# ============================
def main():
    parser = argparse.ArgumentParser(description="从训练日志 CSV 自动建议（追踪/正则/事件）权重与 Huber/Robust 参数，并导出 sw 参数字典")
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument("--index", default="step", help="索引列名（仅用于报告，无强依赖）")
    parser.add_argument("--outdir", default="auto_weights", help="输出目录")
    parser.add_argument("--topk", type=int, default=16, help="追踪塔最多保留前 k 个重要误差项")
    parser.add_argument("--huber-mode", choices=["iqr", "p95"], default="iqr", help="Huber delta 策略")
    # 偏好
    parser.add_argument("--prefer-pos", type=float, default=1.15)
    parser.add_argument("--prefer-vel", type=float, default=1.00)
    parser.add_argument("--prefer-att", type=float, default=0.95)
    parser.add_argument("--prefer-omega", type=float, default=1.05)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # ---------- 追踪塔列选择 ----------
    # 包含 rmse_* 与 rms_omega
    track_patterns = ["rmse_", "rms_omega"]
    track_cols_raw = [c for c in df.columns if any(p in c.lower() for p in track_patterns)]
    track_cols_raw = [c for c in track_cols_raw if pd.api.types.is_numeric_dtype(df[c])]
    # 排除正则列（避免误伤）
    track_cols_raw = [c for c in track_cols_raw if c.lower() not in REG_KEYS]
    # 排序限量
    track_cols = sorted(track_cols_raw, key=lambda c: df[c].var(), reverse=True)[: args.topk]
    if not track_cols:
        raise ValueError("未找到追踪塔的误差列（包含 'rmse_' 或 'rms_omega'）")

    # ---------- 正则塔 ----------
    reg_cols = pick_columns_by_keywords(df, REG_KEYS)

    # ---------- 事件塔（仅建议系数，不进权重） ----------
    over_col = EVENT_OVER if EVENT_OVER in df.columns else None
    sat_col = EVENT_SAT if EVENT_SAT in df.columns else None

    # ---------- 追踪塔评分 ----------
    reward_cands = ["total_reward", "episode_reward", "segment_reward"]
    track_scores, track_details = score_columns(df, track_cols, reward_cands)
    prefer_track = {
        "pos": args.prefer_pos,
        "vel": args.prefer_vel,
        "att": args.prefer_att,
        "omega": args.prefer_omega,
    }
    track_weights = suggest_weights(track_scores, prefer=prefer_track)
    track_huber = {c: suggest_huber_delta(track_details[c], mode=args.huber_mode) for c in track_cols}
    track_robust = {c: dict(median=track_details[c]["median"], iqr=track_details[c]["iqr"], eps=1e-6) for c in track_cols}

    # ---------- 正则塔评分 ----------
    if reg_cols:
        reg_scores, reg_details = score_columns(df, reg_cols, reward_cands)
        # 偏好：du > domega > control_effort > dv
        reg_weights = suggest_weights(reg_scores, prefer={"rms_du": 1.2, "domega": 1.1, "control_effort": 1.05})
        s = sum(reg_weights.values()) or 1.0
        reg_weights = {k: v / s for k, v in reg_weights.items()}
        reg_huber = {c: suggest_huber_delta(reg_details[c], mode=args.huber_mode) for c in reg_cols}
        reg_robust = {c: dict(median=reg_details[c]["median"], iqr=reg_details[c]["iqr"], eps=1e-6) for c in reg_cols}
    else:
        reg_scores, reg_details, reg_weights, reg_huber, reg_robust = {}, {}, {}, {}, {}

    # ---------- lambda_reg ----------
    reward_col = next((c for c in reward_cands if c in df.columns), None)
    lambda_reg = suggest_lambda_reg(df, reg_cols, reward_col) if reg_cols else 0.0

    # ---------- 事件塔建议 ----------
    lambda_over, lambda_sat = 0.0, 0.0
    if over_col:
        over = robust_stats(df[over_col])
        over_level = clamp01(over["p95"] / 100.0)  # 0-1
        lambda_over = 0.3 + 0.7 * over_level      # ∈[0.3,1.0]
    if sat_col:
        sat = robust_stats(df[sat_col])
        sat_level = clamp01(sat["p95"] / 100.0)
        lambda_sat = 0.2 + 0.6 * sat_level        # ∈[0.2,0.8]

    # ============================
    # 生成与公式兼容的 sw 字典
    # segment_reward = -(
    #   sw["W_pos_xy"] * rmse_pos_xy + ... + sw["W_over"] * overshoot_hits_norm
    # )
    # 约定：追踪塔内部和=1；正则塔内部和=1；总正则强度由 lambda_reg 控制。
    # 因此 sw 中：
    # - 对追踪项：直接使用 track_weights 中匹配的列映射
    # - 对正则项：使用 lambda_reg * reg_weights
    # - 事件项：直接使用 lambda_over / lambda_sat（输入已做 0-1 归一化）
    # ============================

    # 映射表：列名包含关键字 → sw 键
    def has(col: str, key: str) -> bool:
        return key in col.lower()

    # 初始化为 0，避免 KeyError
    sw = {
        "W_pos_xy": 0.0,
        "W_pos_z": 0.0,
        "W_vel_xy": 0.0,
        "W_vel_z": 0.0,
        "W_att_roll": 0.0,
        "W_att_pitch": 0.0,
        "W_omega": 0.0,
        "W_u": 0.0,
        "W_dv": 0.0,
        "W_domega": 0.0,
        "W_du": 0.0,
        "W_sat": 0.0,
        "W_over": 0.0,
    }

    # 追踪塔 → sw
    for c, w in track_weights.items():
        lc = c.lower()
        if has(lc, "rmse_pos_xy") or (has(lc, "pos_xy") and has(lc, "rmse")):
            sw["W_pos_xy"] += w
        elif has(lc, "rmse_pos_z") or (has(lc, "pos_z") and has(lc, "rmse")):
            sw["W_pos_z"] += w
        elif has(lc, "rmse_vel_xy") or (has(lc, "vel_xy") and has(lc, "rmse")):
            sw["W_vel_xy"] += w
        elif has(lc, "rmse_vel_z") or (has(lc, "vel_z") and has(lc, "rmse")):
            sw["W_vel_z"] += w
        elif has(lc, "rmse_att_roll") or (has(lc, "att_roll") and has(lc, "rmse")):
            sw["W_att_roll"] += w
        elif has(lc, "rmse_att_pitch") or (has(lc, "att_pitch") and has(lc, "rmse")):
            sw["W_att_pitch"] += w
        elif has(lc, "rms_omega"):
            sw["W_omega"] += w
        # 其他 rmse_* 会被自动忽略，不影响你的公式

    # 正则塔 → sw（乘以 lambda_reg）
    for c, w in reg_weights.items():
        lc = c.lower()
        if lc == "mean_control_effort":
            sw["W_u"] += lambda_reg * w
        elif lc == "rms_dv":
            sw["W_dv"] += lambda_reg * w
        elif lc == "rms_domega":
            sw["W_domega"] += lambda_reg * w
        elif lc == "rms_du":
            sw["W_du"] += lambda_reg * w

    # 事件系数
    sw["W_over"] = float(lambda_over)
    sw["W_sat"] = float(lambda_sat)

    # ---------- 汇总 JSON ----------
    result = dict(
        index=args.index,
        track=dict(
            used_columns=track_cols,
            weights=track_weights,           # 和=1
            huber_delta=track_huber,
            robust_scale=track_robust,
        ),
        regularization=dict(
            used_columns=reg_cols,
            weights=reg_weights,             # 和=1
            lambda_reg=float(lambda_reg),
            huber_delta=reg_huber,
            robust_scale=reg_robust,
        ),
        events=dict(
            lambda_over=float(lambda_over),
            lambda_sat=float(lambda_sat),
            norm_note="Please normalize hits to [0,1] before applying: x_norm = clip(x/100,0,1)",
        ),
        segment_weight_sw=sw,                # 可直接用于你的奖励公式
        notes=dict(
            reward_reference=reward_col,
            prefer_track=prefer_track,
        ),
    )

    out_json = os.path.join(args.outdir, "weights_huber_sw.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- Markdown 报告 ----------
    md = []
    md.append("# Auto Weight/Huber Suggestions with sw for Segment Reward\n")
    md.append(f"- CSV: `{os.path.abspath(args.csv)}`")
    md.append(f"- Index: `{args.index}`")
    md.append("")
    md.append("## Track Tower (sum=1.0)")
    for c in track_cols:
        md.append(f"- {c}: weight={track_weights.get(c,0):.4f}, delta={track_huber.get(c,0):.6g}, median={track_robust[c]['median']:.6g}, IQR={track_robust[c]['iqr']:.6g}")
    md.append("")
    md.append("## Regularization Tower (sum=1.0)")
    if reg_cols:
        md.append(f"- lambda_reg: {lambda_reg:.3f}")
        for c in reg_cols:
            md.append(f"- {c}: weight={reg_weights.get(c,0):.4f}, delta={reg_huber.get(c,0):.6g}, median={reg_robust[c]['median']:.6g}, IQR={reg_robust[c]['iqr']:.6g}")
    else:
        md.append("- No regularization columns detected.")
    md.append("")
    md.append("## Events Tower (coefficients)")
    md.append(f"- lambda_over: {lambda_over:.3f} (apply to overshoot_hits_norm)")
    md.append(f"- lambda_sat : {lambda_sat:.3f} (apply to saturation_hits_norm)")
    md.append("")
    md.append("## sw Mapping for Your Reward Formula")
    md.append("The following keys match your code exactly. Missing metrics will remain 0.")
    for k, v in sw.items():
        md.append(f"- `{k}` = {v:.6g}")
    md.append("")
    out_md = os.path.join(args.outdir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- Snippet（可直接粘贴到训练） ----------
    snippet = f'''# ==== Auto-generated suggestion (Track/Regularization/Events) ====
SW = {json.dumps(sw, ensure_ascii=False, indent=2)}
TRACK_HUBER_DELTA = {json.dumps(track_huber, ensure_ascii=False, indent=2)}
TRACK_ROBUST = {json.dumps(track_robust, ensure_ascii=False, indent=2)}
REG_HUBER_DELTA = {json.dumps(reg_huber, ensure_ascii=False, indent=2)}
REG_ROBUST = {json.dumps(reg_robust, ensure_ascii=False, indent=2)}

def robust_scale(name, x, cfg):
    # x: tensor; cfg[name] has median/iqr/eps
    import torch
    p = cfg[name]
    return (x - p["median"]) / (p["iqr"] + p.get("eps", 1e-6))

def huber_loss(x, delta):
    import torch
    ax = torch.abs(x)
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))

# Hits normalization reminder:
# overshoot_hits_norm = clip(overshoot_hits / 100, 0, 1)
# saturation_hits_norm = clip(saturation_hits / 100, 0, 1)

# Example of your segment reward:
# segment_reward = -(
#     SW["W_pos_xy"] * rmse_pos_xy +
#     SW["W_pos_z"]  * rmse_pos_z  +
#     SW["W_vel_xy"] * rmse_vel_xy +
#     SW["W_vel_z"]  * rmse_vel_z  +
#     SW["W_att_roll"]  * rmse_att_roll +
#     SW["W_att_pitch"] * rmse_att_pitch +
#     SW["W_omega"]  * rms_omega +
#     SW["W_u"]      * mean_control_effort +
#     SW["W_dv"]     * rms_dv +
#     SW["W_domega"] * rms_domega +
#     SW["W_du"]     * rms_du +
#     SW["W_sat"]    * saturation_hits_norm +
#     SW["W_over"]   * overshoot_hits_norm
# )
'''
    out_snippet = os.path.join(args.outdir, "suggestion_snippet.py")
    with open(out_snippet, "w", encoding="utf-8") as f:
        f.write(snippet)

    print(f"[OK] JSON : {out_json}")
    print(f"[OK] Report: {out_md}")
    print(f"[OK] Snippet: {out_snippet}")
    print("Preview sw:")
    for k in sorted(sw.keys()):
        print(f"  - {k:14s}: {sw[k]:.6f}")

if __name__ == "__main__":
    main()