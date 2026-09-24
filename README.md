# FruitFly Flight Lab / 果蠅視覺無人機研究

A student research environment: **Gymnasium + MuJoCo + frozen Flyvis vision + a learned navigation head**, with a browser-based 3D Gaussian Splatting camera.

中學生研究環境：**Gymnasium + MuJoCo + 凍結的 Flyvis 視覺網絡 + 學習式導航網絡**，配合瀏覽器 3D Gaussian Splatting 相機。

## Read the complete guide / 閱讀完整指南

- **[繁體中文：逐步安裝、Flyvis 呼叫、訓練與故障排除](simulation/docs/README.zh-Hant.md)**
- **[English: installation, Flyvis API, training and troubleshooting](simulation/docs/README.en.md)**
- [Windows audit and verification scope / Windows 檢查及驗證範圍](simulation/docs/windows-audit.md)

The two guides have matching sections, explanations and identical commands/code examples. Start at step 1; run one command at a time. Use **64-bit Python 3.12**, with **3.12.10** as the recorded reference version.

兩版指南逐節對照，說明及指令一一對應。由第一步開始，每次執行一行。使用 **64 位元 Python 3.12**，原始實驗版本為 **3.12.10**。

**Windows CPU verification — 2026-09-24:** [native Windows CI passed](https://github.com/xiaoyh-code/fruitfly-flight-lab/actions/runs/35945279968) on Python 3.12.10: fresh installation, actual Flyvis inference, **53 tests passed / 9 skipped**, and all five waypoint targets reached without collision. Tested source revision: `33e34f2`. Interactive graphics and complete Hall training still require the student's local render/bridge checks.

**Windows CPU 驗證 — 2026-09-24：** [原生 Windows 自動測試通過](https://github.com/xiaoyh-code/fruitfly-flight-lab/actions/runs/35945279968)，使用 Python 3.12.10，完成全新安裝、真正 Flyvis 推論、**53 項測試通過／9 項略過**，以及五個航點全部到達、零碰撞。測試程式版本：`33e34f2`。互動圖像及完整工廠訓練仍需在學生電腦完成渲染／相機服務檢查。

## What to open / 應該開啟哪一部分

| Purpose / 用途 | Entry / 入口 |
|---|---|
| View saved results; no installation / 查看保存結果，毋須安裝 | [Public website / 公開網站](https://xiaoyh-code.github.io/fruitfly-flight-lab/) |
| Run locally and train / 本機模擬及訓練 | In the GitHub clone, enter `simulation/`, then follow the guide / 在 GitHub clone 內進入 `simulation/`，然後依照指南 |
| Analyse CSV, Excel, figures and video / 分析數據、圖及影片 | [Research archive / 研究資料室](https://xiaoyh-code.github.io/fruitfly-flight-lab/research/hall-20260922-replay1/) |
| Other scenes / 其他場景 | [Separate scene gallery / 獨立場景展示](https://xiaoyh-code.github.io/fruitfly-flight-lab/scenes.html) |

`site/` is the saved website; `simulation/` is the runnable source. The old `site/research/.../source/` directory is an immutable method snapshot, not an installation. GitHub Pages does not execute Python or train models. OpenGL/WebGL graphics checks are separate from dependency installation.

`site/` 是保存的網站；`simulation/` 是可執行程式。舊 `site/research/.../source/` 是當時的方法快照，不是安裝目錄。GitHub Pages 不會執行 Python 或訓練模型。安裝依賴成功後，仍需分別測試 OpenGL／WebGL 圖像功能。

## What the result means / 如何理解結果

The recorded `hall-20260922-replay1` run succeeded on 20/20 held-out local trials with zero proxy contacts and minimum saved-trajectory body-envelope clearance of about 0.287 m. It uses one manually calibrated obstacle proxy, fixed altitude, ideal goal/velocity inputs and a frozen visual network. It is not a complete hall collision map, whole fly brain, visual depth estimator, 4DGS system or real-flight safety result. New runs may differ.

已保存的 `hall-20260922-replay1` 在 20/20 次獨立局部測試到達目標，代理接觸為零，保存軌跡的機身包絡最小淨距約 0.287 米。它使用單一人工校準障礙代理、固定高度、理想目標／速度資料及凍結視覺網絡，未建立完整工廠碰撞地圖、完整果蠅大腦、視覺測距、4DGS 或真機安全驗證。重新訓練的結果可以不同。

## Assets and credits / 資產與來源

[Flyvis](https://github.com/TuragaLab/flyvis), [MuJoCo](https://github.com/google-deepmind/mujoco), [Gymnasium](https://gymnasium.farama.org/), [Menagerie Crazyflie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2), [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

Downloaders fetch and verify upstream model assets separately. The Hall scan is author-hosted and marked All Rights Reserved; it is not redistributed in `simulation/`. Do not assume permission to republish it. Bundled browser libraries retain their license files. School scenes remain previews, with their attribution on the public site.

下載程式會另外取得並校驗上游模型資產。工廠掃描由作者提供，標示 All Rights Reserved；`simulation/` 不會重新分發它，不能假設有權再次公開。瀏覽器函式庫保留授權檔；校園仍屬預覽，來源標示保留於公開網站。
