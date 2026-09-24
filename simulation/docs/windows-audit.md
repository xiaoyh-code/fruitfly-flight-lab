# Windows reproducibility audit / Windows 重現檢查

Audit date / 檢查日期: 2026-09-24. Target / 對象: Windows x64, CPython 3.12.10.

| Finding / 問題 | Change / 修正 |
|---|---|
| Public clone contained only a static archive / 公開 clone 只有靜態資料 | Added maintained `simulation/` sources, requirements and helpers / 加入完整訓練路徑所需程式、依賴及工具 |
| Mac-only Bash, `.venv/bin` and `open` / Mac 專用啟動方式 | Python installer selects `Scripts/python.exe`; launchers use `webbrowser` / 安裝器選擇 Windows Python，跨平台開瀏覽器 |
| Fresh training required old status/history JSON / 新訓練依賴舊進度檔 | Empty workspace initializes pending milestones and empty history / 新工作目錄建立待完成狀態及空紀錄 |
| Chinese JSON used the OS default encoding / 中文 JSON 使用系統預設編碼 | Explicit UTF-8 reads/writes; guide uses `-X utf8` / 明確 UTF-8 讀寫；指南啟用 UTF-8 模式 |
| Flyvis code confused with pretrained weights / 套件與權重混淆 | Separate checksum-verified weight download and runnable RGB example / 權重獨立校驗下載，提供 RGB 呼叫例子 |
| Missing Hall checkpoint/scene/browser bridge / 缺少工廠模型、場景或相機分頁 | Verified restoration of public action heads, author-source scene downloader, three-terminal tutorial / 校驗恢復公開決策模型、作者來源場景下載及三個終端機教學 |
| Historical snapshot appeared runnable / 歷史程式快照容易被當成安裝包 | Root README directs students to `simulation/` / 根 README 指向 `simulation/` |

## Verification scope / 驗證範圍

- The exact recorded `requirements.lock.txt` resolves for CPython 3.12.10 on Windows x86-64. All packages except `antlr4-python3-runtime==4.9.3` have compatible wheels; that dependency builds from pure Python. No alternative loose version set was substituted.
- 已記錄的完整鎖定版本可解析至 Windows x86-64／Python 3.12.10。除了以純 Python 建置的 `antlr4-python3-runtime==4.9.3`，其餘套件均有相容 wheel；沒有另外猜測一組版本。
- Local Mac checks cover core physics tests, camera rendering, actual Flyvis inference, fresh-workspace initialization and checkpoint overwrite protection. The GitHub `Windows student simulation` workflow installs a fresh Windows environment, runs actual Flyvis inference and regression tests. Its latest outcome is available in the repository Actions tab.
- 本機 Mac 測試包括物理、相機渲染、真正 Flyvis 推論、新工作目錄初始化及 checkpoint 覆蓋保護。GitHub `Windows student simulation` 工作流程會在全新 Windows 環境安裝、執行真正 Flyvis 推論及回歸測試；最新結果可在 repo 的 Actions 查看。
- CI physics/Flyvis success does **not** validate a student's graphics driver, interactive MuJoCo viewer, WebGL camera, complete Hall training or a new benchmark result. Run `doctor.py --render`, open the bridge, then make a new training run locally. Windows ARM64, 32-bit Python and WSL are outside this tested profile.
- CI 物理／Flyvis 通過，**不代表**已驗證學生的顯示驅動、MuJoCo 互動視窗、WebGL 相機、完整工廠訓練或新的研究結果。須在學生本機執行 `doctor.py --render`、開啟相機分頁，再進行新訓練。Windows ARM64、32 位元 Python 及 WSL 不在此測試設定內。

## Upstream references / 上游依據

[Flyvis installation](https://turagalab.github.io/flyvis/install/) documents Linux testing with Python 3.9–3.12. This project targets Python 3.12 and plain `flyvis==1.2.0`; the `pretrained` extra pins an old NumPy/Numba stack and is not used here. [Flyvis package configuration](https://github.com/TuragaLab/flyvis/blob/main/pyproject.toml).

[Flyvis 安裝文件](https://turagalab.github.io/flyvis/install/) 記錄 Linux／Python 3.9–3.12 測試。本專案使用 Python 3.12 及普通 `flyvis==1.2.0`；`pretrained` 額外套件組鎖定較舊 NumPy／Numba 組合，本專案不使用。[Flyvis 套件設定](https://github.com/TuragaLab/flyvis/blob/main/pyproject.toml)。

[MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html) explains the bundled native library and macOS-specific `mjpython` viewer launcher. [Python Windows documentation](https://docs.python.org/3.12/using/windows.html) describes the launcher and virtual environments.

[MuJoCo Python 文件](https://mujoco.readthedocs.io/en/stable/python.html) 說明套件內含原生程式庫，及 macOS 專用的 `mjpython` 視窗啟動方式。[Python Windows 文件](https://docs.python.org/3.12/using/windows.html) 說明 Python 啟動器及虛擬環境。
