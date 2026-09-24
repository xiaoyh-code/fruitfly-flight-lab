# FruitFly Flight Lab — Windows 學生操作手冊

[English edition](README.en.md) · [專案 README](../README.md) · [已公開結果](https://xiaoyh-code.github.io/fruitfly-flight-lab/)

中英文兩版包含相同編號的章節、指令、例子及限制說明。程式碼、檔名和套件名稱刻意保持完全相同。請按順序操作；每個檢查點都會說明甚麼功能應該成功，確認後才繼續。

## 1. 你 Clone 了甚麼，為何舊版本不能訓練

這個 repository 分成兩個不同部分：

| Repository 根目錄內的資料夾 | 用途 | 能否執行訓練？ |
|---|---|---|
| `site/` | 已公開的 HTML、圖片、影片、保存結果及場景檢視器 | 不能。它用來展示保存的證據。 |
| `simulation/` | Python 模擬、訓練程式、依賴、安裝工具及本手冊 | 可以，但要先安裝依賴及下載所需模型。 |

較早的公開版本包含網站及教學證據包，但沒有完整、可安裝的模擬環境。Clone 那些版本不會自動安裝 Python、MuJoCo、Flyvis、官方模型權重、Crazyflie 資產或受限制的工廠掃描。結果資料包內名為 `source` 的資料夾是記錄來源用的程式快照，不等於安裝包。

請先執行 `git pull` 更新，再跟隨本手冊。如果 repository 根目錄仍沒有 `simulation/`，表示你使用的是舊版本或另一個 checkout。不要把老師的 `.venv`、`runtime/` 或 macOS `.command` 啟動檔複製到 Windows：虛擬環境及二進位程式都依賴原來的作業系統及安裝路徑。

除非某一步明確寫「repository 根目錄」，**以下所有指令都在 `simulation/` 內執行**。Terminal 是輸入指令的 PowerShell 視窗；瀏覽器分頁則用來查看網頁。使用伺服器提供的網頁時，必須保持對應的 terminal 運行。

## 2. 硬件、軟件、空間及驗證狀態

| 項目 | 學生安裝設定 |
|---|---|
| 作業系統 | 目標：Windows 10/11，64-bit x86-64。本手冊的安裝目標不包括 Windows ARM。 |
| Python | Python **3.12.x**，64-bit。建議從 **3.12.10** 開始，這亦是原始實驗使用的版本。Flyvis 1.2 要求 Python 低於 3.13；這個環境不要選 Python 3.13/3.14。 |
| Git | Git for Windows，或從 GitHub 下載 ZIP。使用 Git 較容易更新。 |
| 顯示功能 | MuJoCo 圖像需要可用的硬件 OpenGL 驅動；工廠圖像需要瀏覽器 WebGL。只計算物理的示範不需要圖像渲染器。 |
| 瀏覽器 | 已啟用硬件加速的近期 Edge 或 Chrome。工廠訓練期間要保持相機橋接分頁開啟。 |
| 運算 | 訓練及 Flyvis 推論使用 CPU。這個實作不要求 CUDA 或 NVIDIA 顯示卡。 |
| 記憶體 | 先從細小的工廠場景開始。8 GB RAM 可作實際起點；16 GB 可為 Python 及瀏覽器提供更多空間。這些是估算，並非 Windows 實測保證。 |
| 磁碟 | 現有網站工作目錄約 **413 MB**；模擬程式原始碼新增的體積較少。Python、wheel、套件快取、環境及結果另外佔用空間。安裝時建議預留 **8–10 GB 可用空間**；工作環境至少粗略預留 **5 GB**。這是保守估算，不是 Windows 安裝實測大小或 Git 下載大小。 |
| 網絡 | 首次安裝需要連接 PyPI、GitHub、下載 Flyvis 權重的 Google Drive，以及工廠資產的主機。查看部分公開場景亦需要存取遠端資產。 |

已記錄的實驗在 macOS / Apple Silicon 上運行。為 Windows 準備安裝流程及自動預檢，與完整完成 Windows 顯示和訓練驗證，是兩件不同的事。單純通過 import 檢查，不代表顯示卡驅動、瀏覽器相機橋接及整個工廠訓練已經可用。請在每部學生電腦完成以下檢查點，並記錄結果。

官方安裝程式：[Python 3.12.10](https://www.python.org/downloads/release/python-31210/) · [Git for Windows](https://git-scm.com/download/win)。安裝 Python 時，請安裝 Python launcher（`py`），並選用 64-bit 安裝程式。安裝完成後重新開啟 PowerShell。

## 3. 下載專案及檢查 Python

開啟 **PowerShell**，然後逐行輸入。不要複製 Markdown 外圍的反引號，亦不要把 `PS C:\...>` 之類的提示符號當作指令。

```powershell
cd $HOME
git clone --depth 1 https://github.com/xiaoyh-code/fruitfly-flight-lab.git
cd fruitfly-flight-lab
git status
py -3.12 --version
py -3.12 -c "import platform, struct; print(platform.platform()); print(struct.calcsize('P') * 8)"
cd simulation
Get-Location
Test-Path .\scripts\setup_student.py
```

預期檢查點：Python 顯示 `3.12.x`、位元數顯示 `64`、目前位置以 `fruitfly-flight-lab\simulation` 結尾，以及 `Test-Path` 顯示 `True`。如果你下載 ZIP，請先解壓，在解壓後的根目錄開啟 PowerShell，並由 Python 檢查開始；Git 指令不適用於 ZIP 資料夾。

如果你已經 Clone 這個 repository，請用以下指令代替再次 Clone：

```powershell
cd $HOME\fruitfly-flight-lab
git pull
cd simulation
```

請使用有寫入權限的本機資料夾。如果學校管理的使用者設定、OneDrive 同步或端點保護軟件阻止產生檔案，請由老師或學校 IT 提供獲准的本機工作目錄。不要停用學校的保護軟件。

## 4. 分階段安裝

安裝工具會在本機建立新的 `.venv`。指令直接使用該環境的 Python 執行檔，所以不需要啟動虛擬環境，亦不需要改 PowerShell execution policy。`-X utf8` 讓 Python 在 Windows 上以一致的方式讀寫文字。

| Profile | 安裝及準備內容 | 首個用途 |
|---|---|---|
| `core` | 基本依賴及官方 Crazyflie 模型資產 | 執行物理模擬及輸出飛行影片；不安裝 PyTorch 或 Flyvis。 |
| `vision` | Core 加上視覺／訓練依賴，以及官方 Flyvis 權重 | 呼叫視覺網絡及執行合成場景視覺訓練。 |
| `hall` | Vision 加上工廠場景、兩個已公開導航 checkpoint，以及保存的 3DGS 重播資料 | 微調工廠導航策略。 |

先安裝 core：

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile core
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py
.\.venv\Scripts\python.exe -X utf8 -m pip check
```

預期檢查點：安裝過程沒有 exception、doctor 沒有必需 core 項目失敗，以及 `pip check` 顯示沒有損壞的依賴關係。如果安裝失敗，不要跳過錯誤繼續；請保存第一個錯誤及完整 terminal 輸出。

第 5 節的 core 測試通過後，可在同一個本機環境增加視覺功能：

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile vision
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --flyvis
```

要進行完整工廠練習，請執行：

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile hall
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --render --flyvis
```

Profile 名稱代表依賴組合，不代表實驗已完成。安裝 `hall` 不會自動訓練策略。請閱讀安裝輸出，了解下載項目、檔案位置，以及存取或 checksum 驗證錯誤。

## 5. 先檢查物理，再檢查圖像

執行只計算物理的航點示範：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --headless
Get-Content .\outputs\demo_summary.json
```

預期檢查點：摘要顯示 `route_completed: true`、`waypoints_reached: 5`，以及 `collision: false`。這是傳統控制器跟隨固定航點，不是已訓練的 Flyvis 策略。單獨使用 `--headless` 不會產生影片或開啟視窗。

接着測試真正的圖像渲染，並產生一段短片：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --render
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --headless --video outputs/flight-fpv.mp4 --png outputs/flight-preview.png
Start-Process .\outputs\flight-fpv.mp4
Start-Process .\outputs\flight-preview.png
```

影片左邊是無人機前向相機，右邊是外部視角。即使沒有開啟互動 MuJoCo 視窗，offscreen rendering 仍然需要可用的顯示卡驅動。

可選的 Windows 互動視窗：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --viewer --camera track --hold-after-route
```

關閉 MuJoCo 視窗即可結束示範。macOS 的互動檢視器使用 `mjpython`；這個 macOS 專用要求不會取代上面的 Windows 指令。

## 6. 使用哪些套件及版本？

請以 [requirements-core.txt](../requirements-core.txt) 及 [requirements-vision.txt](../requirements-vision.txt) 作為安裝規格。學生安裝工具使用這兩個入口。Vision requirements 包含已記錄實驗的完整 [requirements.lock.txt](../requirements.lock.txt)。這組依賴亦已針對 Windows x86-64 / CPython 3.12 完成解析，沒有替换成另一組套件版本。依賴解析不等於完整 Windows 顯示及訓練測試。

| 元件 | 已記錄原始實驗的版本 | 作用 |
|---|---|---|
| Python | 3.12.10 | 執行 Python 程式。 |
| Gymnasium | 1.3.0 | 定義環境的 `reset()` 及 `step()` 接口。 |
| MuJoCo | 3.13.0 | 模擬 Crazyflie 的物理狀態及接觸。 |
| Flyvis | 1.2.0 | 載入預訓練果蠅視覺迴路。 |
| PyTorch | 2.14.0 | 執行 Flyvis 及訓練導航決策網絡。 |
| torchvision | 0.29.0 | 與 PyTorch 配對、視覺依賴組合需要的套件。 |
| Stable-Baselines3 | 2.9.0 | 供獨立的 PPO 基線練習使用；工廠導航網絡並非以 PPO 訓練。 |
| NumPy | 1.26.4 | 處理數值陣列；這個相容設定要保持 NumPy 低於 2。 |
| OpenCV | 4.11.0.86 | 轉換圖像；這個相容設定要保持低於 4.12。 |
| pandas | 2.3.3 | 處理資料；為了與 Flyvis 相容，要保持 pandas 低於 3。 |

學生 requirements 已固定版本，請避免自行單獨升級。請把實際安裝的版本與實驗一起記錄，讓另一位學生可以核對你的環境：

```powershell
.\.venv\Scripts\python.exe -X utf8 --version
.\.venv\Scripts\python.exe -X utf8 -m pip freeze > student-requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m pip show flyvis torch torchvision mujoco gymnasium numpy
```

如果要理解或手動修復安裝，下列指令會執行依賴及官方模型安裝步驟。仍然建議優先使用自動 profile：

```powershell
py -3.12 -X utf8 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install --upgrade pip
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe -X utf8 scripts/download_drone_model.py
.\.venv\Scripts\python.exe -X utf8 scripts/download_flyvis_models.py
```

請透過 requirements 安裝普通的 `flyvis==1.2.0`。不要自行改成假設存在的 `flyvis[pretrained]` 指令：本專案另外取得官方預訓練壓縮檔，驗證 SHA256 後才解壓。PyTorch 和 torchvision 必須配對相容；不要因為另一篇教學使用不同版本，就單獨升級其中一個。

## 7. 如何安裝、載入及呼叫 Flyvis

需要安裝的是兩樣不同的東西：**Flyvis Python 套件**及其**預訓練權重**。`vision` profile 會安裝兩者。下載程式把壓縮檔保存在 `data/flyvis/`；本專案使用 `data/flyvis/results/flow/0000/000` 模型。單獨執行 `pip install flyvis` 不會提供這個模型資料夾。

先試可直接執行的例子：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/flyvis_example.py
```

如果要寫自己的程式，請把以下內容保存為 `simulation/` 內的 `student_flyvis.py`。例子使用程式產生的圖像，所以首次呼叫時不需要模擬器或瀏覽器：

```python
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor

extractor = FlyvisFeatureExtractor(cpu_threads=2)
extractor.reset()

for frame_index in range(10):
    rgb = np.full((240, 320, 3), 128, dtype=np.uint8)
    left = 40 + frame_index * 12
    rgb[:, left:left + 20, :] = 255
    features = extractor.transform(rgb)
    assert features.shape == (72,)
    assert np.isfinite(features).all()
    print(frame_index, features.shape, float(np.linalg.norm(features)))
```

```powershell
.\.venv\Scripts\python.exe -X utf8 student_flyvis.py
```

預期檢查點：輸出十行，每行包含 `(72,)` 形狀及有限數值。首次初始化可能因為套件建立快取而較慢。這是檢查推論能否運行，不是量度深度準確度或證明導航能力。

輸入必須是形狀為 `(height, width, 3)` 的 RGB `uint8` 陣列。OpenCV 通常讀取 BGR；傳入 OpenCV 影格前要先由 BGR 轉成 RGB。封裝程式會把圖像轉成灰階、裁剪中央正方形、用官方 `BoxEye` 產生 **721 個視網膜元素**，再把 T4a–d / T5a–d 的反應整合成 **72 個數值**。它們是視覺活動特徵，並非 72 個以米為單位的距離。

在 Windows，包裝器會在建立神經連接快取前，自動套用 `src/fruitfly_sim/flyvis_compat.py`。它修正鎖定版本 Datamate 1.0.0 在 HDF5 檔案仍開啟時嘗試刪除它的問題（WinError 32），模型權重及陣列數值保持不變。這是已知的[上游修正](https://github.com/flyvis/datamate/commit/3b9792c3c90fb29d741f8185c7aca912aa0c0942)。請使用本專案包裝器／例子；直接呼叫 `flyvis.NetworkView` 會略過這個相容處理。

封裝程式載入 `flyvis.NetworkView(model_dir).init_network()`、呼叫 `eval()` 及 `requires_grad_(False)`，並在影格之間保留神經狀態。請在**每個 episode 之間**呼叫 `reset()`，不要在每張圖像之前重設；逐幀重設會破壞預期的時間序列。每次相機更新代表 0.1 秒模擬時間，內部使用五個各 0.02 秒的神經子步。封裝程式使用 CPU 推論。

可選的獨立視覺反應檢查：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_flyvis.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_flyvis.py --video outputs/flight-fpv.mp4 --video-layout fpv-left
```

這些指令把反應摘要及圖表保存在 `outputs/`。第二個指令只讀取第 5 節示範影片左邊的 FPV 部分。

## 8. 訓練基線實際是甚麼

```text
3DGS RGB image (320 × 240)
  → frozen pretrained Flyvis visual circuit
  → 72 visual features + 2 relative-goal values + 2 velocity values + 2 previous actions
  → PyTorch MLP: 78 → 128 → 128 → 2, with Tanh activations
  → 2 normalized horizontal velocity commands
  → fixed flight controller + MuJoCo physics
```

原始 Flyvis 網絡以 Sintel 資料進行光流任務預訓練。它模擬果蠅視覺系統；本專案並非訓練完整果蠅大腦。Flyvis 權重維持凍結，從教師動作標籤中學習的是我們的導航 MLP。目標及速度數值來自模擬器的理想狀態；這不是只有圖像輸入或真實 GPS 的實驗。工廠任務把高度維持在 1.2 m，並固定 yaw。

教師知道障礙幾何，並在收集資料時提供標籤。策略沒有收到障礙座標、射線距離或解碼後的深度估計。評估時，策略在沒有教師介入的情況下行動。工廠程式使用監督式模仿學習及額外 DAgger 資料收集，不是 PPO 強化學習。DAgger 的意思是：學習中的策略到達一些狀態，再由教師標示那些狀態適合的動作。

已公開的模型沿革是：初始 **M4** 視覺導航網絡 → **`hall-20260921-fast`** 工廠適應 → **`hall-20260922-replay1`** 工廠微調。目前實驗的「來源基線」是上一輪工廠 checkpoint。後一輪候選不一定更好：已公開實驗的驗證結果選中了第 0 輪。

## 9. 準備工廠資產及來源 checkpoint

執行第 4 節的 `hall` profile，然後檢查以下必需路徑：

```powershell
Test-Path .\models\bitcraze_crazyflie_2\cf2.xml
Test-Path .\models\gaussian\robot_hall.ply
Test-Path .\data\flyvis\results\flow\0000\000
Test-Path .\checkpoints\missions\hall-20260921-fast\hall-visual-head.pt
```

四項都應顯示 `True`。工廠掃描約 6.69 MB，是另外個別下載的資產；公開教學 ZIP 沒有包含它。

還原工具會讀取上層 repository 隨附的 `site/research/hall-20260922-replay1/` 公開證據。它只還原下表兩個導航 checkpoint 及 `outputs/scene-replay/hall-replay.json`，不會重建 `outputs/missions/` 下的歷史 run 目錄，亦不會訓練。歷史表格、影片及實驗紀錄仍可在 `site/` 查看：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/restore_public_evidence.py
```

| 公開證據檔案 | 還原後的本機用途 |
|---|---|
| `models/source-baseline.pt` | `checkpoints/missions/hall-20260921-fast/hall-visual-head.pt` — 用來初始化新訓練的上一輪工廠模型。 |
| `models/selected-round-0.pt` | `checkpoints/missions/hall-20260922-replay1/hall-visual-head.pt` — 已公開實驗選中的模型。 |

左欄路徑以該公開證據資料夾為起點。需要還原是因為教學資料包使用方便理解的模型檔名，而訓練程式預期的是本機 checkpoint 階層。原始 M4 訓練亦可自行建立 checkpoint；如果從已公開的工廠基線開始，則不需要先重新產生 M4。

## 10. 執行工廠訓練：三個 terminal、兩個網頁

開啟**三個 PowerShell terminal**，每個都進入 `simulation/`。同一時間只執行一個工廠訓練程序。按照 A → B → C 的順序；瀏覽器會參與渲染，不只是顯示畫面。

### 10A. Terminal A：啟動相機橋接

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/serve_render_bridge.py --port 8770
```

在 Edge 或 Chrome 開啟 [http://127.0.0.1:8770/bridge](http://127.0.0.1:8770/bridge)。等待場景完成載入，並確認網頁顯示相機已就緒。保持這個分頁開啟，並避免電腦睡眠。不要關閉 Terminal A。首次訓練請求前的黑色畫布不等於相機已就緒；訓練前要先解決頁面的錯誤。

### 10B. Terminal B：啟動訓練儀表板

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/serve_missions.py --port 8769
```

開啟 [http://127.0.0.1:8769/](http://127.0.0.1:8769/)。網頁會顯示進度、策略相機影格，以及獨立的 MuJoCo 觀察視角。啟動這個伺服器或重新整理頁面不會開始訓練。全新安裝要到 Terminal C 開始訓練後才會有目前 run；之後可能顯示你上一個本機 run。

### 10C. Terminal C：開始新一輪訓練

複製以下**完整一行**。它明確指定來源基線；否則程式預設會尋找 M4 checkpoint。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_hall.py --source-checkpoint checkpoints/missions/hall-20260921-fast/hall-visual-head.pt --port 8770 --frames 1600 --dagger-frames 800 --rounds 1 --epochs 60 --validation-episodes 8 --audit-episodes 20 --seed-base 3000000 --torch-seed 924
```

程式會建立帶時間戳的 run 資料夾，例如 `hall-YYYYMMDD-HHMMSS`。等待完成後才開始另一個 run。這個學生指令刻意使用新的隨機 seed 區間，以及不同於已公開實驗的 PyTorch seed。你的結果可能不同；請如實報告，不要直接複製已公開的 20/20。下一個獨立實驗要再使用未用過的 seed 區間。

| 參數 | 意思 |
|---|---|
| `--frames 1600` | 收集 1,600 個初始教師標籤樣本。 |
| `--dagger-frames 800 --rounds 1` | 額外收集一組 800 個「學習策略到達的狀態／教師標籤」樣本。第 0 輪加上一個額外輪次，共擬合兩個候選模型。 |
| `--epochs 60` | 每個候選模型在累積樣本集上進行 60 次擬合遍歷。 |
| `--validation-episodes 8` | 用八次驗證試驗比較候選模型。 |
| `--audit-episodes 20` | 用另外 20 次正式試驗評估選中的候選。 |
| `--seed-base 3000000` | 推算互相分開的收集、驗證及正式測試 seed 區間。 |
| `--torch-seed 924` | 設定本輪訓練使用的 PyTorch / NumPy 隨機 seed。 |
| `--port 8770` | 連接 Terminal A 的相機服務；兩邊 port 必須相同。 |

預期檢查點：相機影格計數增加、儀表板階段及數量變化，最後新 run 顯示完成或明確錯誤。訓練樣本指圖像步數，不是 episode 數量。「完成」表示有限次數的實驗已經結束；宣稱成功之前，請先閱讀是否通過及逐次試驗結果。

要停止，可使用儀表板控制，或在訓練 terminal 按 `Ctrl+C`。訓練停止之前要保持相機橋接可用。程式可能保存中斷 checkpoint，但它不等於已完成正式驗收的模型。全部完成後，在兩個伺服器 terminal 按 `Ctrl+C` 停止服務。

## 11. 尋找結果、保存影片及在 3DGS 重播

訓練完成後，檢查自動記錄的 run ID：

```powershell
Get-Content .\outputs\missions\latest.json
$runId = (Get-Content .\outputs\missions\latest.json -Raw | ConvertFrom-Json).run_id
Get-Content ".\outputs\missions\$runId\hall-training.json"
```

以下指令需要在同一個 terminal 保留 `$runId`。新開 terminal 會失去這個變數；請在新 terminal 再執行賦值那一行。不要把例子 `hall-YYYYMMDD-HHMMSS` 當成真正名稱輸入。

| `outputs/missions/<run-id>/` 內的檔案 | 意思 |
|---|---|
| `run-config.json` | 參數、來源模型 hash、程式碼 hash 及實驗設定。 |
| `seed-manifest.json` | 訓練、驗證及正式 audit seeds。 |
| `hall-demonstrations.npz` | 收集到的 78 維觀察及兩個教師動作標籤。 |
| `history.json` / `status.json` | 訓練進度，以及目前或完成狀態。 |
| `hall_avoidance-validation-source.json` | 來源基線在驗證 seeds 上的結果。 |
| `hall_avoidance-validation-0.json` / `hall_avoidance-validation-1.json` | 候選模型的驗證結果。 |
| `hall_avoidance-audit.json` | 選中候選的正式評估。 |
| `hall_avoidance-stale-vision.json` | 固定首幀視覺特徵時，同一模型的表現。 |
| `hall-training.json` | 整體結果及選中的候選。 |
| `hall_avoidance-evaluation.mp4` | 第一個正式測試 episode 的相機影片。 |
| `renderer-performance.json` | 只計相機橋接的耗時，不是整個策略的處理速度。 |

模型另外保存在 `checkpoints/missions/<run-id>/`：`hall-head-round-0.pt`、`hall-head-round-1.pt`，以及選中的 `hall-visual-head.pt`。寫報告時亦要保留失敗候選及試驗。

保持 Terminal A 及其已就緒瀏覽器分頁運行，為已完成 run 產生較大的第三身觀察影片：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/render_hall_observer.py --run-id $runId --port 8770
Start-Process ".\outputs\missions\$runId\observer-follow.mp4"
Start-Process ".\outputs\missions\$runId\observer-overview.mp4"
```

這些是獨立的 MuJoCo 顯示相機，不會放大無人機物理機身，亦不會改變策略接收的 320 × 240 RGB 輸入。程式會重新執行保存策略的一次試驗，並寫入軌跡比較紀錄；請查看 `observer-preview.json`，了解與原 audit 軌跡有否差異。

把成功的 run 匯出至互動 3DGS 重播，再啟動檢視器：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/export_hall_replay.py --run-id $runId --require-clear
.\.venv\Scripts\python.exe -X utf8 scripts/serve_scene.py --port 8766
```

開啟 [http://127.0.0.1:8766/?scene=hall&replay=1](http://127.0.0.1:8766/?scene=hall&replay=1)。`--require-clear` 要求至少 20 次成功試驗、沒有已記錄接觸，以及軌跡沿途代理淨距為正。如果新 run 未通過匯出門檻，既有重播不會被取代；新 run 的原始結果仍然保留。檢視器播放的是已保存姿態，不是重新在線執行策略評估。

## 12. 不訓練，只在本機查看公開網站

這個方法需要 Python，但不需要 Flyvis 環境。請在 **repository 根目錄**開啟 terminal，即同時包含 `site/` 及 `simulation/` 的資料夾：

```powershell
py -3.12 -X utf8 -m http.server 8780 --bind 127.0.0.1 --directory site
```

開啟 [http://127.0.0.1:8780/](http://127.0.0.1:8780/)，並保持 terminal 運行。請使用 HTTP，不要雙擊 `index.html` 以 `file://` 開啟；瀏覽器會限制本機檔案的模組載入及資產請求。這只顯示已保存的公開結果，不會開始訓練。學校場景放在獨立場景圖庫，尚未進行訓練或碰撞校準。

## 13. 可選：訓練初始視覺網絡或 PPO 練習

安裝 `vision` profile 並通過渲染／Flyvis 檢查後，可透過合成 M4 練習建立新的視覺導航網絡，不需要工廠瀏覽器橋接：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_visual.py --frames 4000 --dagger-frames 1500 --rounds 2
```

它會寫入 `checkpoints/training/M4-visual-head.pt` 及 `outputs/training/M4-*.json` / `.npz`。它使用 MuJoCo 合成相機場景，不是工廠掃描。只有當程式的驗證條件要求時，才會執行可選的 DAgger 輪次。重新執行這個練習可能取代固定名稱的 M4 檔案；如果需要保留上一位學生的結果，請先保存副本。

獨立 PPO curriculum 使用理想位置及合成測距輸入。它適合作比較練習，但不是 Flyvis 工廠模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_curriculum.py --help
```

執行 curriculum 前，請先閱讀 stage 選項。不要把 PPO 結果、合成場景 landing 結果及工廠視覺結果合併成一個成功率；它們的輸入、環境及評估任務各有不同。

## 14. 疑難排解：由第一個失敗檢查點開始

| 現象 | 可能原因 | 處理方法 |
|---|---|---|
| Clone 後沒有 `simulation/` | 舊的純網站版本，或錯誤 repository | 在 repository 根目錄執行 `git pull`，確認更新後的目錄。 |
| 找不到 `py` 指令 | 沒有 Python launcher，或 terminal 在安裝前已開啟 | 安裝附 launcher 的 Python 3.12 x64，再重新開啟 PowerShell。 |
| `No suitable Python runtime found` | 未安裝 Python 3.12 | 先安裝 3.12；必須先讓 `py -3.12 --version` 正常執行。 |
| `No matching distribution found` | Python 版本或架構錯誤、沒有指定版本 wheel，或套件索引受限制 | 確认 3.12 / 64-bit，使用學生 requirements，保存套件名稱及完整 pip 錯誤。不要盲目升級 Python。 |
| 安裝後仍有 `ModuleNotFoundError` | 執行了系統 Python，而非 `.venv` Python，或 profile 未完整安裝 | 使用完整 `.\.venv\Scripts\python.exe` 路徑，並安裝需要的 profile。 |
| PowerShell 阻止 `Activate.ps1` | 啟動腳本政策 | 不需要 activation；使用本手冊的完整 Python 路徑即可。 |
| 缺少 `cf2.xml`、mesh 或 `robot_hall.ply` | 模型／場景未下載，或學校網絡封鎖來源 | 重跑對應 setup profile，並閱讀下載錯誤。 |
| `Official pretrained Flyvis model missing` | 已安裝套件，但沒有預訓練權重 | 用 `.venv` Python 執行 `scripts/download_flyvis_models.py`，或重跑 `vision` profile。 |
| `Source checkpoint is missing` | 未還原證據，或來源路徑錯誤 | 執行還原工具，並使用第 10C 節明確指定的 source-checkpoint 參數。 |
| DLL 載入錯誤 / WinError 126 | 架構不符、缺少 Microsoft runtime，或二進位套件不相容 | 檢查 x64 Python 及套件版本；有需要時請學校 IT 安裝官方 Microsoft Visual C++ x64 runtime。保留完整 DLL 錯誤。 |
| GLFW / OpenGL 錯誤或 MuJoCo 黑畫面 | 顯示卡驅動／graphics context 問題 | 先測試純物理；更新官方 GPU 驅動，改在正常本機桌面測試，避免受限制的遠端桌面。 |
| 瀏覽器 WebGL 錯誤或 context 中斷 | 瀏覽器硬件加速不可用或中斷 | 啟用硬件加速、重啟瀏覽器，再重新載入相機分頁。閱讀頁面錯誤。 |
| `Connection refused` | 相機伺服器沒有運行，或 port 錯誤 | 啟動 Terminal A；兩個指令都使用同一個 `8770` port。 |
| 伺服器已開啟，但相機請求 timeout | 相機分頁關閉、未就緒、休眠，或資產載入失敗 | 開啟 `/bridge` 並等待就緒，保持分頁可用及電腦不睡眠，閱讀錯誤。工廠訓練不會改用假的替代圖像。 |
| WinError 10048 / address already in use | 另一個伺服器已佔用 port | 使用已啟動的正確伺服器，或在其 terminal 按 `Ctrl+C` 停止；更改相機 port 時要同步更改伺服器及訓練指令。 |
| 初始化 Flyvis 時出現提及 `unique_cell_types.h5` 的 `WinError 32` | 舊專案程式仍使用有問題的 Datamate 1.0.0 快取寫入方式 | 用 `git pull` 更新，再使用本專案 Flyvis 包裝器／例子，它會自動套用 Windows 相容處理；毋須更改模型權重。 |
| `PermissionError` | 目錄權限、同步，或另一程序佔用檔案 | 關閉相關讀寫程式，並使用學校批准、有寫入權限的本機目錄。 |
| `UnicodeDecodeError` 或中文亂碼 | Windows 預設文字編碼不同 | 使用指令中的 `-X utf8`；修改程式及 JSON 時以 UTF-8 保存。 |
| 下載 checksum 不符 | 下載不完整、收到登入／proxy 頁而非資產，或上游檔案改變 | 停下來檢查來源及網絡，不要移除 checksum 驗證。 |
| 儀表板一直不變 | 只啟動顯示伺服器，或顯示上一個已完成 run | 啟動 Terminal C，檢查新的 run ID、terminal 輸出及 `latest.json`。 |
| 新訓練比 20/20 差 | Seed、套件、數值行為不同，或擬合出的候選較弱 | 保留實測結果及設定。上一輪實驗的分數不是保證。 |

向老師或提交 bug report 時，請提供：Windows 版本、CPU/GPU、`py -3.12 --version`、doctor 輸出、`student-requirements.txt`、完整指令、完整錯誤 traceback，以及存在時的 `run-config.json` / `status.json`。不要傳送密碼憑證或私人檔案。[Microsoft runtime 文件](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)。

## 15. 學習或修改專案時應看哪些檔案

| `simulation/` 內的檔案或資料夾 | 可了解的內容 |
|---|---|
| `scripts/setup_student.py` | 環境安裝及分階段下載。 |
| `scripts/doctor.py` | 安裝、渲染及 Flyvis 預檢。 |
| `scripts/flyvis_example.py` | 可直接執行的最小 Flyvis 特徵例子。 |
| `src/fruitfly_sim/env.py` | Crazyflie 物理、穩定控制，以及基本 Gymnasium 觀察和動作。 |
| `src/fruitfly_sim/flyvis_features.py` | RGB 前處理、官方模型載入、時間狀態及 72 維特徵。 |
| `src/fruitfly_sim/visual_env.py` | 78 維視覺策略輸入及教師標籤。 |
| `src/fruitfly_sim/splat_env.py` | 工廠相機請求、座標偏移及近似碰撞代理。 |
| `scripts/train_visual.py` | `VisualHead`、Tanh 層，以及初始合成視覺訓練。 |
| `scripts/train_hall.py` | 工廠資料收集、監督式擬合、驗證選模、audit 及固定視覺特徵對照。 |
| `scripts/serve_render_bridge.py` + `dashboard/render-bridge.html` | Python ↔ 瀏覽器的相機通訊。 |
| `scripts/serve_missions.py` + `dashboard/missions.html` | 即時訓練儀表板。 |
| `scripts/render_hall_observer.py` | 獨立第三身影片渲染。 |
| `scripts/export_hall_replay.py` | 保存軌跡驗證及 3DGS 重播匯出。 |
| `tests/` | 自動檢查；圖像測試額外需要可用的渲染器。 |

每次只改一個因素、保存獨立 run 目錄，並使用預先安排的保留測試集作比較。適合的初步擴展包括更多單障礙位置，然後多障礙，再研究 landing。新的學校 3DGS 掃描需要先完成座標／比例對齊及碰撞幾何，才可以聲稱在該場景完成訓練避障。

## 16. 報告措辭、證據及授權

已公開的 `hall-20260922-replay1` 共收集 2,400 個訓練樣本。驗證選中第 0 輪，實際擬合首 1,600 個樣本；額外 800 個樣本的 DAgger 候選保留作未成功比較。最終選中模型在 **20/20** 次保留測試到達目標，並有 **0 次已記錄代理碰撞**。保存軌跡的最小水平機身包絡淨距約 **0.287 m**。來源基線與選中模型在同一組驗證試驗都取得 8/8，所以不能據此宣稱驗證成功率提升。

這些結果只適用於一個靜態工廠場景、一個手動擬合的碰撞圓柱、狹窄的起終點範圍、理想目標／速度資訊，以及固定高度的平面飛行。結果不能證明整棟建築安全、以米量度的深度感知、戶外返航能力、學校場景表現、真機轉移或 4DGS 運作。固定首幀的對照量度的是策略對持續更新視覺特徵的依賴；單憑此對照不能證明網絡學會深度。

適合的方法描述是：「本研究使用凍結的預訓練 Flyvis 視覺網絡提取與運動相關的特徵，透過模仿學習訓練小型導航決策網絡，再於結合 MuJoCo 物理模擬的 3D Gaussian Splatting 視覺場景中評估。」描述自己的實驗時，請把數字換成自己 run 的實測結果。

| 資產 | 處理方式 |
|---|---|
| Flyvis | 遵守[官方 repository](https://github.com/TuragaLab/flyvis)、論文、套件授權及模型來源紀錄，保留原始來源引用。 |
| Crazyflie 模型 | 從 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 下載；保留模型隨附的授權及來源紀錄。 |
| TUM Flight Hall 掃描 | 來源標示 **All Rights Reserved**。安裝工具為練習個別下載；公開教學資料包不包含掃描。可以存取不等於取得再分發授權；公開場景衍生媒體或分享掃描前，要檢查來源條款。 |
| Dajing 學校場景 | 現有公開場景使用 CC BY 4.0 署名；請保留署名。該場景是獨立預覽，不是已訓練結果。 |
| 學生結果 | 保留來源 run ID、設定、seed、失敗試驗、模型 hash，並區分已觀察結果與未來任務。 |

公開證據包包含已記錄資料、模型、圖片、影片及程式快照；它不包含所有原始 RGB 訓練影格或全部外部模型資產。請參閱[研究資料室](https://xiaoyh-code.github.io/fruitfly-flight-lab/research/hall-20260922-replay1/)及[未來任務](https://xiaoyh-code.github.io/fruitfly-flight-lab/roadmap.html)。公開新結果是另外需要老師審閱的步驟；執行本手冊指令不會自動推送至 GitHub。
