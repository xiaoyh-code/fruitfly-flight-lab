#!/usr/bin/env python3
"""Plot the recorded Hall results for the local student teaching pack.

No simulation, model training or publication is performed. Counts and plotted
values are read from completed run files; replay clearances are independently
recomputed using the existing, validated export_hall_replay implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

from export_hall_replay import clearance_metrics, PROXY_CENTER, PROXY_RADIUS, VEHICLE_RADIUS


ROOT = Path(__file__).resolve().parents[1]
COLORS = dict(ink="#17324b", muted="#516171", grid="#dde4ea", green="#187f67",
              blue="#2675b8", red="#c45142", pale="#f2b259", purple="#75529f")
FOOTER = "Scope: one approximate cylinder in a static Hall scene; not a full-hall or real-world collision test."
SOURCE_NAMES = ["history.json", "run-config.json", "seed-manifest.json", "hall-training.json",
                "hall_avoidance-validation-source.json", "hall_avoidance-validation-0.json",
                "hall_avoidance-validation-1.json", "hall_avoidance-audit.json",
                "hall_avoidance-stale-vision.json"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(run_id):
    run = ROOT / "outputs/missions" / run_id
    source = {name: json.loads((run / name).read_text(encoding="utf-8")) for name in SOURCE_NAMES}
    phases = ["validation-source", "validation-0", "validation-1", "audit", "stale-vision"]
    reports = {phase: source[f"hall_avoidance-{phase}.json"] for phase in phases}
    for phase, report in reports.items():
        if report["run_id"] != run_id or report["kind"] != phase:
            raise ValueError(f"Run or phase mismatch: {phase}")
        rows = report["episodes"]
        if not rows or len({r["seed"] for r in rows}) != len(rows):
            raise ValueError(f"Missing or duplicated episodes: {phase}")
        for metric in ("success", "collision"):
            if not np.isclose(np.mean([r[metric] for r in rows]), report[f"{metric}_rate"]):
                raise ValueError(f"Inconsistent {metric} count: {phase}")
    seed_lists = {key: [row["seed"] for row in value["episodes"]] for key, value in reports.items()}
    if not seed_lists["validation-source"] == seed_lists["validation-0"] == seed_lists["validation-1"]:
        raise ValueError("Validation candidates did not use the same seeds")
    if seed_lists["audit"] != seed_lists["stale-vision"]:
        raise ValueError("Audit and stale-feature diagnostic seeds differ")
    if set(seed_lists["audit"]) & set(seed_lists["validation-0"]):
        raise ValueError("Audit and validation seeds overlap")
    train_seeds = {seed for row in source["seed-manifest.json"]["training_rollouts"] for seed in row["seeds"]}
    if train_seeds & (set(seed_lists["audit"]) | set(seed_lists["validation-0"])):
        raise ValueError("Training and evaluation seeds overlap")
    audit = sorted(reports["audit"]["episodes"], key=lambda row: row["seed"])
    clearance = []
    for row in audit:
        value = clearance_metrics(row["trajectory_xyz"])["swept_proxy_clearance_m"]
        if not np.isclose(value, row["min_polyline_clearance"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Clearance mismatch for seed {row['seed']}")
        clearance.append(value)
    fits = [row for row in source["history.json"] if row["phase"] == "fit"]
    return run, source, reports, audit, np.asarray(clearance), fits


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": COLORS["ink"],
        "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a9b6c2", "figure.facecolor": "white",
        "axes.facecolor": "white", "grid.color": COLORS["grid"],
        "grid.linewidth": .8, "axes.axisbelow": True, "svg.fonttype": "none",
        "svg.hashsalt": "fruitfly-hall-student-figures",
    })


def heading(fig, title, subtitle):
    fig.text(.075, .945, title, fontsize=21, fontweight="bold", va="top")
    fig.text(.075, .892, subtitle, fontsize=11.5, color=COLORS["muted"], va="top")


def footer(fig, note, run_id):
    fig.text(.075, .081, note, fontsize=10.5, color=COLORS["muted"], va="top")
    fig.text(.075, .044, FOOTER, fontsize=9.2, color=COLORS["muted"], va="top")
    fig.text(.075, .019, f"Recorded run: {run_id}", fontsize=8.5, color=COLORS["muted"], va="top")


def save(fig, out, name):
    dimensions = np.rint(np.asarray(fig.get_size_inches()) * 160).astype(int).tolist()
    fig.savefig(out / f"{name}.png", dpi=160)
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    return dict(png=f"{name}.png", svg=f"{name}.svg", width_px=dimensions[0], height_px=dimensions[1])


def phase_outcomes(out, reports, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.3), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.72, bottom=.23, wspace=.32)
    heading(fig, "Model selection and held-out evaluation", "Different seed sets are shown in separate panels; bars are episode outcomes, not a learning curve.")
    legend = [Patch(color=COLORS["green"], label="Reached goal"),
              Patch(color=COLORS["red"], label="Recorded proxy collision"),
              Patch(color=COLORS["pale"], label="Other failure (timeout)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.068, .844), ncol=3, frameon=False, fontsize=10.5)
    sets = [("A  Validation: same 8 seeds", ["validation-source", "validation-0", "validation-1"],
             ["Source\ncheckpoint", "Round 0\nselected", "Round 1\nrejected"]),
            ("B  Held-out: same 20 seeds", ["audit", "stale-vision"],
             ["Selected model\nlive features", "Selected model\nstale features"])]
    exact = {}
    for ax, (title, phases, labels) in zip(axes, sets):
        xs = np.arange(len(phases))
        counts = []
        for phase in phases:
            rows = reports[phase]["episodes"]
            n = len(rows)
            success = sum(r["success"] for r in rows)
            collision = sum(r["collision"] for r in rows)
            other = n - success - collision
            if any(r["failure_reason"] != "timeout" for r in rows if not r["success"] and not r["collision"]):
                raise ValueError("Unexpected other failure reason: amend legend")
            counts.append((success, collision, other))
            exact[phase] = dict(episodes=n, successes=success, collisions=collision, timeouts=other)
        bottom = np.zeros(len(phases))
        for category, color in enumerate(("green", "red", "pale")):
            proportions = np.asarray([100 * row[category] / sum(row) for row in counts])
            ax.bar(xs, proportions, bottom=bottom, width=.57, color=COLORS[color], edgecolor="white", linewidth=.8)
            for x, height, base, count in zip(xs, proportions, bottom, counts):
                if count[category]:
                    ax.text(x, base + height / 2, str(count[category]), ha="center", va="center",
                            color="white" if color != "pale" else COLORS["ink"], fontweight="bold", fontsize=14)
            bottom += proportions
        for x, row in zip(xs, counts):
            ax.text(x, 104, f"{row[0]}/{sum(row)} reached", ha="center", fontsize=10.5, fontweight="bold")
        ax.set_title(title, loc="left", pad=22)
        ax.set_xticks(xs, labels, fontsize=11)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(0, 111)
        ax.grid(axis="y")
        ax.set_ylabel("Episodes (%)")
    footer(fig, "Round 0 was selected on validation only. Audit results were not used to select the checkpoint.", run_id)
    return save(fig, out, "phase-outcomes"), exact


def audit_clearance(out, audit, clearance, run_id):
    fig, ax = plt.subplots(figsize=(12.8, 7.3))
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.27)
    minimum = int(np.argmin(clearance))
    heading(fig, "How close did each audit flight come?", "Minimum horizontal envelope clearance, calculated along line segments joining recorded poses (about 10 Hz).")
    xs = np.arange(len(audit))
    colors = [COLORS["blue"]] * len(audit)
    colors[minimum] = COLORS["purple"]
    ax.bar(xs, clearance, color=colors, width=.68)
    ax.axhline(0, color=COLORS["red"], linewidth=1.5)
    ax.axhline(clearance[minimum], color=COLORS["purple"], linestyle=(0, (4, 4)), linewidth=1)
    fig.text(.97, .832, f"Smallest margin: {clearance[minimum]:.3f} m  ·  seed {audit[minimum]['seed']}",
             ha="right", va="top", fontweight="bold", color=COLORS["purple"])
    for x, value in zip(xs, clearance):
        ax.text(x, value + .011, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(-.055, .66)
    ax.set_xticks(xs, [str(row["seed"]) for row in audit], rotation=55, ha="right", fontsize=9.5)
    ax.set_xlabel("Audit seed (ascending order; each bar is one held-out episode)", labelpad=10)
    ax.set_ylabel("Minimum polyline clearance (m)")
    ax.set_yticks([0, .1, .2, .3, .4, .5, .6])
    ax.grid(axis="y")
    ax.text(-.45, -.038, "0 m = horizontal vehicle envelope touches the proxy", fontsize=9.5, color=COLORS["red"])
    footer(fig, "Clearance = nearest XY distance to (0, 0.2) minus (0.45 m proxy radius + 0.09 m vehicle envelope).", run_id)
    return save(fig, out, "audit-clearance")


def trajectories(out, audit, clearance, run_id):
    fig, ax = plt.subplots(figsize=(12.8, 8.1))
    fig.subplots_adjust(left=.075, right=.765, top=.81, bottom=.2)
    heading(fig, "All 20 held-out trajectories", "Local MuJoCo coordinates in metres. Z is approximately 1.2 m; this figure projects the recorded paths onto XY.")
    ax.add_patch(Circle(PROXY_CENTER, PROXY_RADIUS + VEHICLE_RADIUS, facecolor="#fdf0ec",
                        edgecolor=COLORS["red"], linewidth=1.8, linestyle=(0, (4, 3)), zorder=3))
    ax.add_patch(Circle(PROXY_CENTER, PROXY_RADIUS, facecolor="#d6dee5", edgecolor=COLORS["muted"], linewidth=1.4, zorder=4))
    minimum = int(np.argmin(clearance))
    for i, row in enumerate(audit):
        points = np.asarray(row["trajectory_xyz"])
        ax.plot(points[:, 0], points[:, 1], color=COLORS["blue"], alpha=.30, linewidth=1.3, zorder=5)
    worst = np.asarray(audit[minimum]["trajectory_xyz"])
    ax.plot(worst[:, 0], worst[:, 1], color=COLORS["purple"], linewidth=2.5, zorder=6)
    starts = np.asarray([row["trajectory_xyz"][0][:2] for row in audit])
    goals = np.asarray([row["goal"] for row in audit])
    ends = np.asarray([row["trajectory_xyz"][-1][:2] for row in audit])
    ax.scatter(starts[:, 0], starts[:, 1], c=COLORS["green"], s=30, marker="o", zorder=8,
               edgecolors="white", linewidths=.4)
    ax.scatter(goals[:, 0], goals[:, 1], c=COLORS["ink"], s=55, marker="*", zorder=8)
    ax.scatter(ends[:, 0], ends[:, 1], facecolors="white", edgecolors=COLORS["blue"], s=25, zorder=7)
    ax.plot([-1.6, -1.6], [-.22, .22], color=COLORS["green"], linewidth=4, alpha=.25)
    ax.plot([1.6, 1.6], [-.22, .22], color=COLORS["ink"], linewidth=4, alpha=.2)
    ax.text(*PROXY_CENTER, "Approximate\ncylinder\nr = 0.45 m", ha="center", va="center", fontsize=11.5, zorder=10)
    ax.annotate("Start samples", (-1.6, .22), xytext=(-1.83, .70),
                arrowprops=dict(arrowstyle="-", color=COLORS["green"]), color=COLORS["green"])
    ax.annotate("Goal samples", (1.6, .22), xytext=(.95, .70),
                arrowprops=dict(arrowstyle="-", color=COLORS["ink"]), color=COLORS["ink"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.08, .92)
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_xticks(np.arange(-1.5, 1.6, .5))
    ax.grid()
    handles = [Line2D([], [], color=COLORS["blue"], label="20 audit paths"),
               Line2D([], [], color=COLORS["purple"], lw=2.5, label=f"Smallest margin\nseed {audit[minimum]['seed']}"),
               Line2D([], [], color=COLORS["red"], linestyle="--", label="Inflated boundary\nr = 0.54 m"),
               Line2D([], [], linestyle="", color=COLORS["green"], marker="o", label="Start"),
               Line2D([], [], linestyle="", color=COLORS["ink"], marker="*", markersize=10, label="Goal"),
               Line2D([], [], linestyle="", markeredgecolor=COLORS["blue"], markerfacecolor="white", marker="o", label="Recorded endpoint")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.77, .75), frameon=False, fontsize=11, labelspacing=1.3)
    fig.text(.785, .305, "Start / goal sampling\nx = −1.6 / +1.6 m\ny ∈ [−0.22, +0.22] m\n\nGoal tolerance\n0.25 m", fontsize=10.5, color=COLORS["muted"], va="top", linespacing=1.5)
    footer(fig, "The dashed circle includes the 0.09 m conservative vehicle envelope; it is not another physical obstacle.", run_id)
    return save(fig, out, "audit-trajectories")


def training_loss(out, fits, source, run_id):
    fig, ax = plt.subplots(figsize=(12.8, 7.3))
    fig.subplots_adjust(left=.10, right=.96, top=.76, bottom=.25)
    heading(fig, "Recorded training loss: two final-epoch values", "The run saved one final-epoch value per fit. Per-epoch loss history is unavailable; no learning curve is inferred.")
    xs = np.arange(len(fits))
    values = [row["loss"] for row in fits]
    ax.bar(xs, values, width=.5, color=[COLORS["green"] if row["iteration"] == source["hall-training.json"]["selected_iteration"] else COLORS["muted"] for row in fits])
    ax.set_xticks(xs, [f"Round {row['iteration']}\n{row['samples']:,} cumulative training samples" for row in fits])
    for x, value in zip(xs, values):
        ax.text(x, value + .001, f"{value:.7f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylabel("Final epoch: mean batch MSE")
    ax.set_xlabel("Different training sample sets; selected checkpoint = Round 0", labelpad=16)
    ax.set_ylim(0, max(values) * 1.32)
    ax.set_xlim(-.75, max(xs) + .75)
    ax.grid(axis="y")
    epochs = source["run-config.json"]["arguments"]["epochs"]
    dropout = source["run-config.json"]["arguments"]["previous_action_dropout"]
    footer(fig, f"Each fit: {epochs} epochs; previous-action dropout = {dropout:.0%}. Loss is not an independent test-set accuracy measure.", run_id)
    return save(fig, out, "training-fit-loss")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="hall-20260922-replay1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("invalid run-id")
    out = args.output or ROOT / "outputs/teaching" / args.run_id / "figures"
    out.mkdir(parents=True, exist_ok=True)
    run, source, reports, audit, clearance, fits = load_inputs(args.run_id)
    style()
    phase_info, counts = phase_outcomes(out, reports, args.run_id)
    figure_info = [phase_info, audit_clearance(out, audit, clearance, args.run_id),
                   trajectories(out, audit, clearance, args.run_id), training_loss(out, fits, source, args.run_id)]
    captions = [
        dict(id="phase-outcomes", title_zh="模型選擇與獨立驗收", title_en="Model selection and held-out evaluation",
             caption_zh="左圖使用同一組 8 個驗證種子：來源模型及第 0 輪均 8/8 到達；第 1 輪只有 4/8 到達、1 次代理碰撞及 3 次逾時，因此保留第 0 輪。右圖是另一組 20 個未用於訓練或選模的種子：即時視覺特徵 20/20 到達；固定首幀視覺特徵則 0/20 到達、全部逾時。兩組樣本分開展示；結果不能解讀成每一輪訓練都改善，也不能證明模型已學會真實深度。",
             caption_en="Validation used the same 8 seeds for all candidates; held-out tests used a separate set of 20 seeds. Round 0 was selected on validation. Stale-feature failures diagnose dependence on temporal visual input, not causal depth estimation."),
        dict(id="audit-clearance", title_zh="20 次驗收的最小水平淨距", title_en="Minimum horizontal clearance in each audit episode",
             caption_zh=f"每條柱代表一次驗收，以約 10 Hz 儲存位置之間的直線段計算與代理圓柱的最近水平距離，再減 0.45 m 障礙半徑及 0.09 m 保守機身包絡半徑。最小值為 {float(clearance.min()):.9f} m（種子 {audit[int(np.argmin(clearance))]['seed']}）。正值只表示這條折線與此水平代理沒有相交，並非全工廠或真實飛行安全保證。",
             caption_en="Each bar is the minimum distance from recorded XY polyline segments to the proxy centre minus the combined 0.54 m radius. This calculation covers the interpolated replay, not every physics tick or every Hall surface."),
        dict(id="audit-trajectories", title_zh="20 次驗收的俯視路徑", title_en="All 20 held-out XY trajectories",
             caption_zh="藍線是局部 MuJoCo 座標中的實際記錄路徑，紫線為最小淨距的一次。實線圓柱半徑 0.45 m；紅虛線 0.54 m 額外包含 0.09 m 的保守機身包絡，並非另一個實體障礙。起點及目標的 X 固定為 −1.6／+1.6 m，Y 在 ±0.22 m 內隨機。終點只需距目標小於 0.25 m，因此不一定與星形目標重合。這是定高局部避障，未驗證整個工廠碰撞。",
             caption_en="Recorded XY paths with start, goal and endpoint markers. The dashed inflated radius includes the conservative vehicle envelope. Goals are reached within a 0.25 m tolerance; altitude is approximately 1.2 m."),
        dict(id="training-fit-loss", title_zh="已保存的訓練誤差", title_en="Recorded final-epoch training loss",
             caption_zh="現有紀錄只保存兩次訓練各自最後一個 epoch 的平均 batch MSE，數值為 0.0090366062 及 0.0318138190；沒有逐 epoch 誤差曲線。兩次訓練分別使用累計 1,600 及 2,400 筆樣本，均跑 60 個 epoch，並以 50% 機率丟棄上一動作輸入。這是不同資料集合上的訓練誤差，不能當作獨立測試準確率；最終採用第 0 輪權重。",
             caption_en="Only final-epoch mean batch MSE was saved for the two fits. They used different cumulative datasets, so these two bars are neither an epoch learning curve nor independent test accuracy."),
    ]
    for row, info in zip(captions, figure_info):
        row.update(info)
    (out / "captions.json").write_text(json.dumps(dict(run_id=args.run_id, figures=captions), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = dict(
        version=1, run_id=args.run_id, created_at=datetime.now(timezone.utc).isoformat(),
        script="scripts/plot_student_results.py", script_sha256=sha256(Path(__file__)),
        dependencies=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
        sources=[dict(path=str((run / name).relative_to(ROOT)), sha256=sha256(run / name)) for name in SOURCE_NAMES],
        method_sources=[dict(path=name, sha256=sha256(ROOT / name)) for name in
                        ["scripts/export_hall_replay.py", "scripts/train_hall.py", "src/fruitfly_sim/splat_env.py"]],
        formulas=dict(clearance="min over XY trajectory segments of distance(segment, [0, 0.2]) - 0.45 - 0.09; segment projection is clamped to [0,1]",
                      phase_outcomes="counts of recorded episode success / collision flags; remaining failed episodes verified to be timeouts",
                      training_loss="history.json fit.loss = arithmetic mean of minibatch MSE values during the final training epoch, including previous-action dropout"),
        verified=dict(disjoint_training_validation_audit_seeds=True, matched_candidate_validation_seeds=True,
                      matched_live_stale_audit_seeds=True, all_audit_polyline_clearances_recomputed=True),
        measurements=dict(phase_outcomes=counts,
                          min_clearance_m=float(clearance.min()), min_clearance_seed=audit[int(np.argmin(clearance))]["seed"],
                          audit_clearance_by_seed=[dict(seed=row["seed"], min_polyline_clearance_m=float(value)) for row, value in zip(audit, clearance)],
                          fits=fits),
        limitations=["One approximate cylinder only; no full-hall collision mesh.",
                     "Planar, fixed-altitude navigation with ideal goal and velocity inputs.",
                     "Flyvis weights were frozen; the separate action head was trained.",
                     "Per-epoch training losses were not saved.",
                     "The positive polyline margin is not a proof about all physics ticks or real-world safety."],
        figures=[{**info, "sha256": {extension: sha256(out / info[extension]) for extension in ("png", "svg")}} for info in figure_info],
    )
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(out), figures=figure_info, min_clearance_m=float(clearance.min())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
