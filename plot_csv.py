#!/usr/bin/env python3
# python plot_csv.py --csv ./log_step_metrics/episode_20250917-201122/step_metrics.csv  --outdir plots --smooth 0 --skip-constant
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境下绘图
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="读取 CSV 并为每一列绘图（随 step 变化）。"
    )
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument("--outdir", default="plots", help="输出图像目录")
    parser.add_argument("--index", default="step", help="作为横轴的索引列名，默认 step")
    parser.add_argument("--dpi", type=int, default=150, help="图像分辨率 DPI")
    parser.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg", "svg", "pdf"], help="输出图格式")
    parser.add_argument("--sample", type=int, default=0, help="等间距采样间隔（>0 时启用，如 10 表示每 10 个点取一个）")
    parser.add_argument("--smooth", type=int, default=0, help="移动平均窗口大小（>1 时启用平滑）")
    parser.add_argument("--skip-constant", action="store_true", help="跳过常量列（无变化）")
    parser.add_argument("--combine", action="store_true", help="额外生成合并多子图的大图")
    parser.add_argument("--cols", nargs="*", default=None, help="仅绘制指定列（列名列表），默认绘制全部数值列")
    return parser.parse_args()

def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()

def sanitize_filename(name: str) -> str:
    # 将列名转换为可作为文件名的安全字符串
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in name)
    return "_".join(safe.strip().split())

def select_columns(df: pd.DataFrame, cols: List[str] | None) -> List[str]:
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"指定列不存在: {missing}")
        return cols
    # 自动选择数值列（排除 index 列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def should_skip_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=False) <= 1

def downsample(df: pd.DataFrame, step: int) -> pd.DataFrame:
    if step is None or step <= 0:
        return df
    return df.iloc[::step, :]

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_single_series(x, y, title: str, outpath: str, dpi: int):
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()

def arrange_subplot_grid(n: int) -> Tuple[int, int]:
    # 近似方阵布局
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def plot_combined(x, df: pd.DataFrame, columns: List[str], outpath: str, dpi: int):
    rows, cols = arrange_subplot_grid(len(columns))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(columns):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.plot(x, df[col], linewidth=1.2)
        ax.set_title(col)
        ax.grid(True, linestyle="--", alpha=0.3)
    # 关闭多余子图
    total = rows * cols
    for j in range(len(columns), total):
        r = j // cols
        c = j % cols
        fig.delaxes(axes[r, c])
    fig.supxlabel("step")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

def main():
    args = parse_args()

    # 读取 CSV
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"读取 CSV 失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.index not in df.columns:
        print(f"指定索引列 '{args.index}' 不存在。可用列: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # 选择列
    columns = select_columns(df, args.cols)

    # 将 step 作为横轴；确保为数值
    try:
        x = pd.to_numeric(df[args.index], errors="coerce")
    except Exception:
        x = df[args.index]
    # 按需下采样
    mask = slice(None)
    if args.sample and args.sample > 0:
        df = downsample(df, args.sample)
        x = downsample(x.to_frame(), args.sample).iloc[:, 0]

    # 平滑处理（仅对选中列）
    if args.smooth and args.smooth > 1:
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = moving_average(df[col], args.smooth)

    # 输出目录
    ensure_outdir(args.outdir)

    # 过滤常量列
    plotted_cols = []
    for col in columns:
        if col == args.index:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if args.skip_constant and should_skip_constant(df[col]):
            print(f"跳过常量列: {col}")
            continue

        outname = f"{sanitize_filename(col)}.{args.fmt}"
        outpath = os.path.join(args.outdir, outname)
        try:
            plot_single_series(x, df[col], title=col, outpath=outpath, dpi=args.dpi)
            plotted_cols.append(col)
        except Exception as e:
            print(f"绘制列 {col} 失败: {e}", file=sys.stderr)

    # 生成合并大图
    if args.combine and plotted_cols:
        combined_path = os.path.join(args.outdir, f"combined.{args.fmt}")
        try:
            plot_combined(x, df, plotted_cols, combined_path, dpi=args.dpi)
        except Exception as e:
            print(f"生成合并图失败: {e}", file=sys.stderr)

    if plotted_cols:
        print(f"完成绘图，共输出 {len(plotted_cols)} 个单独图像到: {os.path.abspath(args.outdir)}")
    else:
        print("没有可绘制的列（可能都被过滤或不是数值列）")

if __name__ == "__main__":
    main()