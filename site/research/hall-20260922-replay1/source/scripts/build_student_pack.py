#!/usr/bin/env python3
"""Assemble a local or approved public teaching archive; never upload anything."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = "hall-20260922-replay1"
LOCAL_PACK = ROOT / "outputs/teaching" / RUN
PUBLIC_PACK = ROOT / "outputs/teaching-public" / RUN
WORKBOOK = ROOT / "outputs/01a0c1ce-44a4-79d3-ac49-a2bb3be060bb/training-data.xlsx"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sanitize(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    if isinstance(value, Mapping):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(sanitize(item) for item in value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def inline(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" rel="noopener">\1</a>', text)


def markdown(text):
    """Render the small, checked-in guide's headings, tables, lists and code."""
    lines = text.splitlines()
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index]); index += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
        elif line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            tag = min(n + 1, 6)
            out.append(f"<h{tag}>{inline(line[n:].strip())}</h{tag}>")
        elif line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", cell) for cell in cells):
                    table.append(cells)
                index += 1
            out.append('<div class="table-scroll"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in table[0]) + "</tr></thead><tbody>")
            out.extend("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in table[1:])
            out.append("</tbody></table></div>")
            continue
        elif re.match(r"(?:\d+\. |\- )", line):
            ordered = bool(re.match(r"\d+\. ", line))
            tag = "ol" if ordered else "ul"
            pattern = r"\d+\. " if ordered else r"\- "
            out.append(f"<{tag}>")
            while index < len(lines) and re.match(pattern, lines[index]):
                out.append("<li>" + inline(re.sub(pattern, "", lines[index], count=1)) + "</li>")
                index += 1
            out.append(f"</{tag}>")
            continue
        elif line.startswith("> "):
            out.append("<blockquote><p>" + inline(line[2:]) + "</p></blockquote>")
        else:
            out.append("<p>" + inline(line) + "</p>")
        index += 1
    return "\n".join(out)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_models(pack):
    import torch
    model_dir = pack / "models"
    model_dir.mkdir(exist_ok=True)
    items = []
    sources = {
        "selected-round-0.pt": ROOT / f"checkpoints/missions/{RUN}/hall-visual-head.pt",
        "rejected-round-1.pt": ROOT / f"checkpoints/missions/{RUN}/hall-head-round-1.pt",
        "source-baseline.pt": ROOT / "checkpoints/missions/hall-20260921-fast/hall-visual-head.pt",
    }
    for name, source in sources.items():
        original = torch.load(source, map_location="cpu", weights_only=False)
        cleaned = sanitize(original)
        target = model_dir / name
        torch.save(cleaned, target)
        saved = torch.load(target, map_location="cpu", weights_only=False)
        assert set(original["state_dict"]) == set(saved["state_dict"])
        assert all(torch.equal(v, saved["state_dict"][k]) for k, v in original["state_dict"].items())
        assert b"/Users/" not in target.read_bytes()
        items.append({"file": f"models/{name}", "source": str(source.relative_to(ROOT)),
                      "source_sha256": digest(source), "packaged_sha256": digest(target),
                      "state_dict_bit_exact": True,
                      "metadata": {k: v for k, v in saved.items() if k != "state_dict"}})
    write_json(model_dir / "provenance.json", {
        "operation": "Project-root absolute path in metadata changed to project-relative path; tensors unchanged.",
        "architecture": "Linear(78,128), Tanh, Linear(128,128), Tanh, Linear(128,2), Tanh",
        "scope": "Learned action heads only; frozen pretrained Flyvis network is not bundled.",
        "checkpoints": items,
    })


def copy_source(pack):
    paths = [
        "scripts/train_hall.py", "scripts/train_visual.py", "scripts/train_curriculum.py",
        "scripts/train_missions.py", "scripts/render_hall_observer.py", "scripts/serve_render_bridge.py",
        "scripts/export_hall_replay.py", "scripts/export_student_data.py", "scripts/plot_student_results.py",
        "scripts/build_student_pack.py", "scripts/build_student_workbook.mjs",
        "tests/test_hall_replay.py", "tests/test_student_data.py",
        "dashboard/render-bridge.html", "dashboard/student-review.html",
        "requirements.in", "requirements.lock.txt", "docs/hall-training.md",
    ] + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src/fruitfly_sim").glob("*.py"))]
    for relative in paths:
        source = ROOT / relative
        if source.exists():
            target = pack / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Preserve source code, but make any example working-directory path portable.
            target.write_text(sanitize(source.read_text()))
    (pack / "source/README.md").write_text(
        "# Method snapshot, not a complete simulation installation\n\n"
        "These project scripts and environment classes document the recorded run. "
        "Dependencies are pinned in requirements.lock.txt. Analyse CSV/NPZ without installing the simulator.\n\n"
        "To rerun training use the existing full FruitFly workspace and the teacher guide. "
        "The Hall PLY, frozen Flyvis model/data, Menagerie assets, browser libraries and virtual environment "
        "are intentionally not bundled. Paths in scripts expect that full workspace. "
        "Workbook rebuilding additionally requires the documented Codex artifact-tool runtime.\n"
    )


def assemble(pack, *, allow_missing_workbook=False, published=False):
    summary = json.loads((pack / "data/summary.json").read_text())
    assert summary["counts"]["training_samples"] == 2400
    assert summary["counts"]["evaluation_episodes"] == 64
    assert summary["counts"]["selected_training_samples"] == 1600
    source_run = ROOT / "outputs/missions" / RUN
    guide = (ROOT / "docs/student-report-guide.md").read_text()
    if published:
        guide = guide.replace(
            "本輪資料與回放先供本機審閱，收到使用者「OK」後才公開。",
            "本輪資料與回放已獲使用者批准公開；以下本機指令供已有完整環境者重現。",
        )
    (pack / "student-report-guide.md").write_text(guide)
    for view in ("follow", "overview"):
        for ext in ("jpg", "mp4"):
            copy_file(source_run / f"observer-{view}.{ext}", pack / f"media/observer-{view}.{ext}")
    write_json(pack / "media/observer-preview.json", sanitize(json.loads((source_run / "observer-preview.json").read_text())))
    copy_file(ROOT / "models/bitcraze_crazyflie_2/LICENSE", pack / "licenses/Crazyflie-LICENSE.txt")
    if WORKBOOK.exists():
        copy_file(WORKBOOK, pack / "training-data.xlsx")
    elif not allow_missing_workbook:
        raise FileNotFoundError("Canonical workbook is not ready")
    copy_models(pack)
    copy_source(pack)
    introduction = (
        "# FruitFly student research data — published experiment\n\n"
        f"Run: {RUN}, 2026-09-22. Approved for publication by the user.\n\n"
        "Public site: https://xiaoyh-code.github.io/fruitfly-flight-lab/\n\n"
        "On the public site, the pipeline, future tasks and other scenes are linked separately. "
        "The 3DGS replay requires the online site and the author's remote Hall scene. "
        "The ZIP link is available on the hosted page; its payload is already extracted here.\n\n"
        if published else
        "# FruitFly student research data — local review\n\n"
        f"Run: {RUN}, 2026-09-22. Publication awaits the user's explicit OK.\n\n"
        "The ZIP download link and local 3DGS replay require the running local services.\n\n"
    )
    (pack / "README.md").write_text(
        introduction +
        "Open index.html for the offline data viewer; open training-data.xlsx for typed, filterable tables. "
        "Figures, tables and MuJoCo videos can be inspected offline.\n\n"
        "## Contents\n\n"
        "- data/: original NPZ, exact CSV exports, sanitized saved JSON, seed manifest, dictionaries and hashes.\n"
        "- figures/: four PNG/SVG plots, bilingual captions and their source hashes.\n"
        "- models/: source, selected and rejected action heads with identical tensors and sanitized metadata.\n"
        "- media/: independent MuJoCo third-person replay; the Hall scan and its RGB footage are excluded.\n"
        "- source/: project code snapshot and pinned dependencies for method inspection, not a standalone install.\n"
        "- student-report-guide.md: teaching prompts, qualified results and repeat-experiment command.\n"
        "- manifest.json: size and SHA-256 of every packaged payload (manifest itself excluded).\n\n"
        "## Interpretation\n\n"
        "2,400 collected examples; selected round 0 fitted 1,600. All 64 evaluations are retained. "
        "Repeated validation seeds and paired control seeds mean these are not 64 independent trials. "
        "20/20 audit successes apply only to the one approximate obstacle, static Hall and goal/velocity-assisted "
        "planar task. This does not demonstrate real depth, full-scene safety or real flight. "
        "Only two final-epoch MSE values exist; do not invent a learning curve.\n\n"
        "## Attribution and scene boundary\n\n"
        "Crazyflie model: Google DeepMind MuJoCo Menagerie / whoenig, MIT (license included). "
        "Flyvis: https://github.com/TuragaLab/flyvis ; MuJoCo: https://mujoco.org/ ; "
        "Gymnasium: https://gymnasium.farama.org/ . Hall source: https://huggingface.co/datasets/amacati/splats . "
        "The Hall scene is All Rights Reserved and is not redistributed in this package. "
        "Citing its source does not grant redistribution permission. "
        "Frozen Flyvis pretrained model files are not redistributed.\n"
    )
    with (pack / "data/episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    for row in episodes:
        for name in ("success", "collision"):
            row[name] = row[name] == "true"
        for name in ("seed", "episode_index", "steps"):
            row[name] = int(row[name])
        for name in ("seconds", "min_polyline_clearance_m", "final_distance_m"):
            row[name] = float(row[name])
    captions = json.loads((pack / "figures/captions.json").read_text())["figures"]
    figures = "\n".join(
        f'<figure class="figure"><a href="figures/{html.escape(f["png"])}"><img loading="lazy" src="figures/{html.escape(f["png"])}" alt="{html.escape(f["title_zh"])}" width="{f["width_px"]}" height="{f["height_px"]}" style="height:auto"></a>'
        f'<figcaption><strong>{html.escape(f["title_zh"])}</strong>{html.escape(f["caption_zh"])}<p><a href="figures/{html.escape(f["png"])}" download>PNG ↓</a><a href="figures/{html.escape(f["svg"])}" download>SVG ↓</a></p></figcaption></figure>'
        for f in captions
    )
    file_specs = [
        ("training-data.xlsx", "Excel 工作簿", "可篩選樣本、逐次結果、公式摘要與欄位字典。"),
        ("data/training_samples.csv", "全部訓練樣本 · CSV", "2,400 行 × 83 欄；78 維 observation、2 維教師 action 及樣本標記。"),
        ("data/episodes.csv", "64 次測試結果 · CSV", "成功、碰撞、時間、淨距、末端距離；包含被淘汰的一輪。"),
        ("data/trajectories.csv", "4,983 個軌跡點 · CSV", "保留實際時間、位置、姿態與 phase／seed 對應。"),
        ("data/audit_paired_control.csv", "20 組配對對照 · CSV", "同一 seed：正常視覺與固定首幀特徵。"),
        ("data/training_history.csv", "兩輪訓練結束值 · CSV", "只包含實際保存的 final-epoch MSE，不是完整 loss 曲線。"),
        ("data/hall-demonstrations.npz", "原始 NumPy 訓練資料 · NPZ", "原檔逐位元保留；observations (2400,78)、actions (2400,2)。"),
        ("data/feature_dictionary.json", "數據欄位與單位 · JSON", "解釋特徵、正規化、座標、action 與缺失欄位。"),
        ("data/seed_manifest.json", "種子分組 · JSON", "訓練、驗證與驗收分組；未保存逐筆訓練 seed 對應。"),
        ("models/selected-round-0.pt", "獲選決策模型 · PyTorch", "第 0 輪擬合 1,600 筆；ZIP 亦附來源及被淘汰模型。"),
        ("data/provenance.json", "原始數據來源及校驗 · JSON", "檔案 SHA-256、導出方法與路徑處理紀錄。"),
        ("README.md", "資料包使用說明", "離線開啟方法、用途範圍、程式快照及素材來源。"),
    ]
    file_html = []
    for path, title, description in file_specs:
        target = pack / path
        size = f"{target.stat().st_size / 1024:,.1f} KiB" if target.exists() else "整理中"
        file_html.append(f'<div class="file"><a href="{path}" download>{title} ↓</a><p>{description}</p><small>{size}</small></div>')
    payload_size = sum(p.stat().st_size for p in pack.rglob("*") if p.is_file() and p.name not in {"index.html", "manifest.json"})
    template = (ROOT / "dashboard/student-review.html").read_text()
    data = json.dumps({"summary": summary, "episodes": episodes}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    replacements = {
        "__FIGURES__": figures, "__FILES__": "\n".join(file_html),
        "__GUIDE__": markdown(guide),
        "__DATA__": data, "__PACKAGE_SIZE__": f"約 {payload_size / 1024 ** 2:.1f} MiB 檔案",
        "__STATUS__": "已公開 · 保存實驗" if published else "本機審閱 · 未公開",
        "__SITE_NAV__": (
            '<a href="../../index.html">訓練 Pipeline</a>'
            '<a href="../../roadmap.html">未來任務</a>'
            '<a href="../../scenes.html">其他場景</a>' if published else ""
        ),
        "__REPLAY_DESCRIPTION__": (
            "圓柱是近似碰撞代理。影片不含工廠 3DGS 掃描素材；網上實景回放會讀取作者提供的遠端場景。"
            if published else "圓柱是近似碰撞代理。影片不含工廠 3DGS 掃描素材；完整實景回放仍可在本機開啟。"
        ),
        "__REPLAY_URL__": "../../scene.html?scene=hall&amp;replay=1" if published else "http://127.0.0.1:8766/?scene=hall&amp;replay=1",
        "__REPLAY_LABEL__": "開啟 3DGS 實景回放 ↗" if published else "開啟本機 3DGS 回放 ↗",
        "__REPLAY_LOCAL_ONLY__": "false" if published else "true",
        "__FOOTER_STATUS__": (
            "已獲批准公開 · 2026-09-22 保存實驗。所有結果均來自這一輪；未來任務會另行記錄。"
            if published else "此版本只供本機審閱；收到你明確回覆 OK 後，才處理網上發佈。"
        ),
    }
    for key, value in replacements.items():
        assert key in template
        template = template.replace(key, value)
    if published:
        assert not re.search(r'href=[\"\']https?://(?:127\.0\.0\.1|localhost)', template)
    (pack / "index.html").write_text(template)
    records = []
    for path in sorted(pack.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(pack))
        assert not path.is_symlink()
        assert path.suffix.lower() != ".ply"
        assert path.name not in {"live.jpg", "latest-rgb.png", "hall_avoidance-evaluation.mp4"}
        # Source tests contain a synthetic /Users/another fixture and scanner
        # strings. Neither identifies a person; actual home paths are forbidden.
        payload = path.read_bytes()
        assert str(Path.home()).encode() not in payload, relative
        if not relative.startswith("source/"):
            assert b"/Users/" not in payload, relative
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    status = "approved_public_experiment" if published else "local_review_awaiting_explicit_OK"
    write_json(pack / "manifest.json", {
        "run_id": RUN, "status": status,
        "archive_scope": "Manifest covers every payload except itself; archive is stored outside this directory.",
        "count": len(records), "total_payload_bytes": sum(row["bytes"] for row in records), "files": records,
    })
    archive = pack.parent / f"{RUN}-student-pack.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(pack.rglob("*")):
            if path.is_file():
                bundle.write(path, str(path.relative_to(pack.parent)))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(records) + 1
    destination = PUBLIC_PACK if published else LOCAL_PACK
    result = {"run_id": RUN, "archive": str((destination.parent / archive.name).relative_to(ROOT)), "bytes": archive.stat().st_size,
              "sha256": digest(archive), "payload_files": len(records), "workbook_included": (pack / "training-data.xlsx").exists(),
              "uploaded": False, "status": status}
    return result


def build(allow_missing_workbook=False, published=False):
    if published and allow_missing_workbook:
        raise ValueError("Published package requires the completed canonical workbook")
    if published:
        # Rebuild only from reviewed data and figures. Staging prevents a partial build
        # from altering either the approved local package or an existing public one.
        manifest = json.loads((LOCAL_PACK / "manifest.json").read_text())
        PUBLIC_PACK.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".student-pack-", dir=PUBLIC_PACK.parent) as directory:
            staging = Path(directory) / RUN
            staging.mkdir()
            for item in manifest["files"]:
                relative = Path(item["path"])
                if relative.parts[0] not in {"data", "figures"}:
                    continue
                assert not relative.is_absolute() and ".." not in relative.parts
                source = LOCAL_PACK / relative
                assert source.is_file() and not source.is_symlink()
                assert digest(source) == item["sha256"], relative
                copy_file(source, staging / relative)
            result = assemble(staging, published=True)
            if PUBLIC_PACK.exists():
                shutil.rmtree(PUBLIC_PACK)
            staging.replace(PUBLIC_PACK)
            (Path(directory) / f"{RUN}-student-pack.zip").replace(PUBLIC_PACK.parent / f"{RUN}-student-pack.zip")
        write_json(PUBLIC_PACK.parent / "build-public.json", result)
    else:
        result = assemble(LOCAL_PACK, allow_missing_workbook=allow_missing_workbook)
        write_json(LOCAL_PACK.parent / "build-review.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-workbook", action="store_true", help="Local draft only; final archive requires XLSX")
    parser.add_argument("--published", action="store_true", help="Build a separate approved public package; never upload or modify the local review package")
    args = parser.parse_args()
    build(allow_missing_workbook=args.allow_missing_workbook, published=args.published)
