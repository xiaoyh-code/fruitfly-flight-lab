# FruitFly Flight Lab — Windows student guide

[繁體中文版](README.zh-Hant.md) · [Project README](../README.md) · [Published results](https://xiaoyh-code.github.io/fruitfly-flight-lab/)

The Chinese and English editions contain the same numbered sections, commands, examples, and qualifications. Code, filenames, and package names are intentionally identical in both editions. Follow the steps in order; each checkpoint tells you what should work before you continue.

## 1. What you cloned, and why an older clone could not train

This repository has two different parts:

| Directory at the repository root | Purpose | Does it run training? |
|---|---|---|
| `site/` | Published HTML, figures, videos, saved results, and scene viewer | No. It displays saved evidence. |
| `simulation/` | Python simulation, training code, dependencies, setup tools, and these guides | Yes, after installing dependencies and downloading the required models. |

Earlier public versions contained the website and a teaching evidence package, but not a complete installable simulator. Cloning those versions did not install Python, MuJoCo, Flyvis, official model weights, the Crazyflie assets, or the restricted Hall scan. A folder named `source` inside a results package is a provenance snapshot, not an installation.

Update your clone with `git pull` before following this guide. If the repository root still has no `simulation/` directory, you are using the older release or a different checkout. Do not copy the teacher's `.venv`, `runtime/`, or macOS `.command` launchers to Windows: virtual environments and binaries depend on their original operating system and installation path.

Unless a step explicitly says “repository root,” **run every command below inside `simulation/`**. A terminal is the PowerShell window where you enter a command; a browser tab is where you view a page. Keep server terminals running while using their pages.

## 2. Hardware, software, disk space, and validation status

| Item | Student setup |
|---|---|
| Operating system | Target: Windows 10/11, 64-bit x86-64. Windows ARM is not the documented installation target. |
| Python | Python **3.12.x**, 64-bit. **3.12.10** is the recommended starting version and the source experiment's version. Flyvis 1.2 requires Python below 3.13; do not choose Python 3.13/3.14 for this environment. |
| Git | Git for Windows, or a GitHub ZIP download. Git makes updates easier. |
| Graphics | A working hardware OpenGL driver for MuJoCo images and WebGL in the browser for Hall images. The physics-only demo does not need an image renderer. |
| Browser | A current Edge or Chrome with hardware acceleration enabled. Keep the camera bridge tab open during Hall training. |
| Compute | Training and Flyvis inference use the CPU. CUDA and an NVIDIA GPU are not required by this implementation. |
| Memory | Start with the small factory scene. 8 GB RAM is a practical starting point; 16 GB gives more room for Python and the browser. These are estimates, not measured Windows guarantees. |
| Disk | The existing website working tree is about **413 MB**; simulator source adds much less. Python, wheels, package caches, environments, and outputs require additional space. Reserve **8–10 GB free** for setup; allow at least roughly **5 GB** for the working environment. These are conservative estimates, not a measured Windows installation size or Git download size. |
| Network | First installation needs access to PyPI, GitHub, Google Drive for Flyvis weights, and the Hall asset host. Viewing some published scenes also needs access to remote assets. |

The recorded experiment was run on macOS / Apple Silicon. Windows-oriented setup and automated preflight checks are separate from a complete Windows graphics-and-training validation. A passing import check alone does not prove that your GPU driver, browser camera bridge, or full Hall training works. Complete the checkpoints below on each student's computer and record the outcome.

Official installers: [Python 3.12.10](https://www.python.org/downloads/release/python-31210/) · [Git for Windows](https://git-scm.com/download/win). In the Python installer, install the Python launcher (`py`) and choose the 64-bit installer. Reopen PowerShell after installation.

## 3. Download the project and check Python

Open **PowerShell**, then enter one line at a time. Do not copy the surrounding Markdown backticks or a prompt such as `PS C:\...>`.

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

Expected checkpoint: Python reports `3.12.x`, the bit count is `64`, your location ends in `fruitfly-flight-lab\simulation`, and `Test-Path` prints `True`. If you downloaded a ZIP, extract it first, open PowerShell in its extracted root, and start at the Python checks; Git commands do not apply to a ZIP folder.

If you already cloned this repository, use these commands instead of cloning it again:

```powershell
cd $HOME\fruitfly-flight-lab
git pull
cd simulation
```

Use a writable local folder. If a school's managed profile, OneDrive synchronization, or endpoint protection blocks generated files, ask the teacher or school IT for an approved local working directory. Do not disable school protection software.

## 4. Install one level at a time

The setup tool creates a fresh local `.venv`. Commands use its Python executable directly, so you do not need to activate the environment or change PowerShell's execution policy. `-X utf8` makes Python read and write text consistently on Windows.

| Profile | Installs and prepares | First purpose |
|---|---|---|
| `core` | Core dependencies and official Crazyflie model assets | Run physics and render a flight video; no PyTorch or Flyvis installation. |
| `vision` | Core plus vision/training dependencies and official Flyvis weights | Call the visual network and run synthetic visual training. |
| `hall` | Vision plus the Hall scene, two published navigation checkpoints, and saved 3DGS replay data | Fine-tune the factory navigation policy. |

Start with core:

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile core
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py
.\.venv\Scripts\python.exe -X utf8 -m pip check
```

Expected checkpoint: setup finishes without an exception, the doctor reports no failed required core checks, and `pip check` reports no broken requirements. Do not continue through a failed installation; save the first error and the complete terminal output.

Upgrade the same local environment when the core tests in Section 5 pass:

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile vision
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --flyvis
```

For the complete factory exercise, run:

```powershell
py -3.12 -X utf8 scripts/setup_student.py --profile hall
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --render --flyvis
```

The profile names describe dependencies, not completed experiments. Installing `hall` does not automatically train a policy. Read the setup output for downloads, file locations, and any access or checksum failure.

## 5. Check physics first, then pictures

Run the physics-only waypoint demonstration:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --headless
Get-Content .\outputs\demo_summary.json
```

Expected checkpoint: the summary reports `route_completed: true`, `waypoints_reached: 5`, and `collision: false`. This is a traditional controller following fixed waypoints, not a trained Flyvis policy. `--headless` alone does not produce a video or open a window.

Now test actual image rendering and make a short video:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --render
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --headless --video outputs/flight-fpv.mp4 --png outputs/flight-preview.png
Start-Process .\outputs\flight-fpv.mp4
Start-Process .\outputs\flight-preview.png
```

The video has the drone's forward camera on the left and an external view on the right. Offscreen rendering still needs a working graphics driver even though no interactive MuJoCo window is open.

Optional Windows interactive window:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_demo.py --viewer --camera track --hold-after-route
```

Close the MuJoCo window to end this demonstration. On macOS, the interactive viewer uses `mjpython`; that macOS-specific requirement does not replace the Windows command above.

## 6. Which packages and versions are used?

Use [requirements-core.txt](../requirements-core.txt) and [requirements-vision.txt](../requirements-vision.txt) as the installation specifications. They are the entry points used by student setup. The vision requirements include the exact [requirements.lock.txt](../requirements.lock.txt) from the recorded experiment. That dependency set has also been resolved for Windows x86-64 / CPython 3.12; no alternative package versions were substituted. Dependency resolution is not a complete Windows graphics-and-training test.

| Component | Version in the recorded source experiment | Role |
|---|---|---|
| Python | 3.12.10 | Runs the Python programs. |
| Gymnasium | 1.3.0 | Defines the environment's `reset()` and `step()` interface. |
| MuJoCo | 3.13.0 | Simulates the Crazyflie's physical state and contacts. |
| Flyvis | 1.2.0 | Loads the pretrained fly visual circuit. |
| PyTorch | 2.14.0 | Runs Flyvis and trains the navigation head. |
| torchvision | 0.29.0 | Matching PyTorch vision package required by the vision dependency set. |
| Stable-Baselines3 | 2.9.0 | Used by separate PPO baseline exercises; the Hall head is not trained with PPO. |
| NumPy | 1.26.4 | Numerical arrays; keep the compatibility requirement below NumPy 2. |
| OpenCV | 4.11.0.86 | Image conversion; keep below 4.12 for this compatibility profile. |
| pandas | 2.3.3 | Data processing; keep below pandas 3 for Flyvis compatibility. |

The student requirements pin the versions; avoid upgrading them independently. Record your actual installed versions with your experiment so that another student can check your environment:

```powershell
.\.venv\Scripts\python.exe -X utf8 --version
.\.venv\Scripts\python.exe -X utf8 -m pip freeze > student-requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m pip show flyvis torch torchvision mujoco gymnasium numpy
```

To understand or repair installation manually, these commands perform the dependency and official model steps. The automatic profiles remain the recommended route:

```powershell
py -3.12 -X utf8 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install --upgrade pip
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe -X utf8 scripts/download_drone_model.py
.\.venv\Scripts\python.exe -X utf8 scripts/download_flyvis_models.py
```

Install plain `flyvis==1.2.0` through the requirements file. Do not replace it with an assumed `flyvis[pretrained]` command: this project obtains the official pretrained archive separately and verifies its SHA256 before extracting it. PyTorch and torchvision must remain a compatible pair; do not independently upgrade one because another tutorial uses a different version.

## 7. How to install, load, and call Flyvis

There are two separate things to install: **the Flyvis Python package** and **its pretrained weights**. The `vision` profile installs both. The downloader saves the archive under `data/flyvis/`; this project uses model `data/flyvis/results/flow/0000/000`. Installing only `pip install flyvis` does not supply that model directory.

Try the runnable example first:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/flyvis_example.py
```

For your own program, save the following as `student_flyvis.py` inside `simulation/`. It uses generated images so that no simulator or browser is needed for this first call:

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

Expected checkpoint: ten lines print a `(72,)` shape and finite numbers. The first initialization may take longer while libraries build caches. This checks that inference works; it does not measure depth accuracy or prove navigation ability.

The input must be an RGB `uint8` array with shape `(height, width, 3)`. OpenCV usually reads BGR; convert BGR to RGB before passing an OpenCV frame. The wrapper converts the image to grayscale, crops the center square, uses the official `BoxEye` to create **721 retinal elements**, and pools the T4a–d / T5a–d responses into **72 numbers**. They are visual activity features, not 72 distances in meters.

The wrapper loads `flyvis.NetworkView(model_dir).init_network()`, calls `eval()` and `requires_grad_(False)`, and retains neural state between frames. Call `reset()` **between episodes**, not before every image; resetting every frame destroys the intended temporal sequence. Each camera update represents 0.1 simulated seconds, using five neural substeps of 0.02 seconds. The wrapper uses CPU inference.

Optional independent visual-response checks:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_flyvis.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_flyvis.py --video outputs/flight-fpv.mp4 --video-layout fpv-left
```

These save response summaries and plots under `outputs/`. The second command takes only the left-hand FPV portion of the demonstration video created in Section 5.

## 8. What the trained baseline actually is

```text
3DGS RGB image (320 × 240)
  → frozen pretrained Flyvis visual circuit
  → 72 visual features + 2 relative-goal values + 2 velocity values + 2 previous actions
  → PyTorch MLP: 78 → 128 → 128 → 2, with Tanh activations
  → 2 normalized horizontal velocity commands
  → fixed flight controller + MuJoCo physics
```

The original Flyvis network was pretrained on an optical-flow task using Sintel data. It models a fly visual system; this project does not train a whole fly brain. Flyvis weights stay frozen. Our navigation MLP is the component that learns from teacher action labels. Goal and velocity values come from the ideal simulator state; this is not an image-only or real-GPS experiment. Height is held at 1.2 m and yaw is fixed for the Hall task.

The teacher knows obstacle geometry and supplies labels during collection. The policy does not receive obstacle coordinates, ray distances, or a decoded depth estimate. At evaluation time, the policy acts without the teacher. The Hall script uses supervised imitation learning and additional DAgger collections, not PPO reinforcement learning. In DAgger, the learner visits states and the teacher labels appropriate actions at those states.

The published lineage is: initial **M4** visual navigation head → **`hall-20260921-fast`** factory adaptation → **`hall-20260922-replay1`** factory fine-tuning. “Source baseline” in the current experiment means the previous factory checkpoint. A later candidate is not automatically better: validation selected round 0 in the published run.

## 9. Prepare the factory assets and source checkpoint

Run the `hall` profile from Section 4. Then check these required paths:

```powershell
Test-Path .\models\bitcraze_crazyflie_2\cf2.xml
Test-Path .\models\gaussian\robot_hall.ply
Test-Path .\data\flyvis\results\flow\0000\000
Test-Path .\checkpoints\missions\hall-20260921-fast\hall-visual-head.pt
```

All four checks should print `True`. The Hall scan is about 6.69 MB. It is a separate individually downloaded asset; the public teaching ZIP does not contain it.

The restoration helper reads the published evidence shipped under `site/research/hall-20260922-replay1/` in the parent repository. It restores only the two navigation checkpoints listed below and `outputs/scene-replay/hall-replay.json`; it does not recreate historical run directories under `outputs/missions/` or train anything. Historical tables, videos, and experiment records remain available in `site/`:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/restore_public_evidence.py
```

| Published evidence file | Local use after restoration |
|---|---|
| `models/source-baseline.pt` | `checkpoints/missions/hall-20260921-fast/hall-visual-head.pt` — previous factory model used to initialize a new run. |
| `models/selected-round-0.pt` | `checkpoints/missions/hall-20260922-replay1/hall-visual-head.pt` — selected model from the published run. |

The paths in the left column are relative to that published evidence directory. This restoration is needed because the teaching package uses descriptive model filenames, while training scripts expect the local checkpoint hierarchy. Original M4 training can also create its own checkpoint; it is not necessary to regenerate M4 before starting from the published factory baseline.

## 10. Run factory training: three terminals and two browser pages

Open **three PowerShell terminals**, each inside `simulation/`. Use only one Hall training process at a time. Follow A → B → C; the browser participates in rendering and is not merely a display.

### 10A. Terminal A: start the camera bridge

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/serve_render_bridge.py --port 8770
```

Open [http://127.0.0.1:8770/bridge](http://127.0.0.1:8770/bridge) in Edge or Chrome. Wait until the scene finishes loading and the page reports that the camera is ready. Keep this tab open and the computer awake. Do not close Terminal A. A black canvas before the first training request is not a substitute for a camera-ready status; resolve any page error before training.

### 10B. Terminal B: start the training dashboard

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/serve_missions.py --port 8769
```

Open [http://127.0.0.1:8769/](http://127.0.0.1:8769/). This page shows progress, policy camera frames, and separate MuJoCo observer views. Starting this server or refreshing its page does not start training. A fresh installation has no current run until Terminal C starts one; later it may show your last local run.

### 10C. Terminal C: start a new training run

Copy this **single line**. It uses the source baseline explicitly; otherwise the script defaults to looking for an M4 checkpoint.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_hall.py --source-checkpoint checkpoints/missions/hall-20260921-fast/hall-visual-head.pt --port 8770 --frames 1600 --dagger-frames 800 --rounds 1 --epochs 60 --validation-episodes 8 --audit-episodes 20 --seed-base 3000000 --torch-seed 924
```

The script creates a timestamped run directory, such as `hall-YYYYMMDD-HHMMSS`. Wait for it to finish before starting another run. This student command deliberately uses a new random seed range and a different PyTorch seed from the published run. Your outcome may differ; report it honestly instead of copying the published 20/20 result. Use another unused seed block for a later independent experiment.

| Argument | Meaning |
|---|---|
| `--frames 1600` | Collect 1,600 initial teacher-labeled samples. |
| `--dagger-frames 800 --rounds 1` | Collect one additional set of 800 learner-state / teacher-label samples. Round 0 plus one additional round makes two fitted candidates. |
| `--epochs 60` | Make 60 fitting passes per candidate over the accumulated sample set. |
| `--validation-episodes 8` | Compare candidates on eight validation trials. |
| `--audit-episodes 20` | Evaluate the selected candidate on 20 separate final trials. |
| `--seed-base 3000000` | Derive separate collection, validation, and final-test random seed blocks. |
| `--torch-seed 924` | Set the PyTorch / NumPy random seed used for this training run. |
| `--port 8770` | Connect to Terminal A's camera service; both port numbers must match. |

Expected checkpoint: the camera's frame counter advances, the dashboard's phases and counts change, and the new run eventually reports completion or an explicit error. Training samples are image steps, not episodes. “Completed” means the bounded experiment finished; read its pass/fail and per-trial results before claiming success.

To stop, use the dashboard control or press `Ctrl+C` in the training terminal. Keep the camera bridge available until training stops. The script may save an interrupted checkpoint, which is not the same as a completed audited model. Stop the two servers with `Ctrl+C` when you are finished.

## 11. Find results, save a video, and replay in 3DGS

After training, inspect the automatically recorded run ID:

```powershell
Get-Content .\outputs\missions\latest.json
$runId = (Get-Content .\outputs\missions\latest.json -Raw | ConvertFrom-Json).run_id
Get-Content ".\outputs\missions\$runId\hall-training.json"
```

Keep `$runId` in the same terminal for the commands below. Opening a new terminal loses that variable; rerun its assignment there. Do not type the example `hall-YYYYMMDD-HHMMSS` literally.

| File under `outputs/missions/<run-id>/` | What it means |
|---|---|
| `run-config.json` | Arguments, source model hash, source code hashes, and experimental settings. |
| `seed-manifest.json` | Training, validation, and final audit seeds. |
| `hall-demonstrations.npz` | Collected 78-value observations and two teacher action labels. |
| `history.json` / `status.json` | Training progress and current or completed state. |
| `hall_avoidance-validation-source.json` | Source baseline evaluated on the validation seeds. |
| `hall_avoidance-validation-0.json` / `hall_avoidance-validation-1.json` | Candidate validation results. |
| `hall_avoidance-audit.json` | Final evaluation of the selected candidate. |
| `hall_avoidance-stale-vision.json` | Same model with first-frame visual features held fixed. |
| `hall-training.json` | Overall result and selected candidate. |
| `hall_avoidance-evaluation.mp4` | First final-test episode's camera video. |
| `renderer-performance.json` | Camera-bridge timing only; not complete policy throughput. |

Models are saved separately under `checkpoints/missions/<run-id>/`: `hall-head-round-0.pt`, `hall-head-round-1.pt`, and selected `hall-visual-head.pt`. Keep failed candidates and trials when writing your report.

With Terminal A and its ready browser tab still running, generate the larger third-person observer videos for your completed run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/render_hall_observer.py --run-id $runId --port 8770
Start-Process ".\outputs\missions\$runId\observer-follow.mp4"
Start-Process ".\outputs\missions\$runId\observer-overview.mp4"
```

These are separate MuJoCo display cameras. They do not enlarge the drone's physical body or change the policy's 320 × 240 RGB input. The script reruns a saved policy trial and writes a trajectory-comparison record; inspect `observer-preview.json` for any difference from the saved audit.

Export a successful run to the interactive 3DGS replay, then start its viewer:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/export_hall_replay.py --run-id $runId --require-clear
.\.venv\Scripts\python.exe -X utf8 scripts/serve_scene.py --port 8766
```

Open [http://127.0.0.1:8766/?scene=hall&replay=1](http://127.0.0.1:8766/?scene=hall&replay=1). `--require-clear` requires at least 20 successful trials, no recorded contacts, and positive swept proxy clearance. If the new run fails that export gate, the existing replay is not replaced; the new run's raw results remain available. The viewer plays saved poses, not a fresh online policy evaluation.

## 12. View the published website locally without training

This route needs Python but no Flyvis environment. Open a terminal at the **repository root**, the folder containing both `site/` and `simulation/`:

```powershell
py -3.12 -X utf8 -m http.server 8780 --bind 127.0.0.1 --directory site
```

Open [http://127.0.0.1:8780/](http://127.0.0.1:8780/). Keep that terminal running. Use HTTP rather than double-clicking `index.html` as a `file://` URL; browsers restrict module loading and asset requests from local files. This shows saved public results and does not start training. School scenes are in the separate scene gallery and have not been trained or collision-calibrated.

## 13. Optional: train the initial visual head or PPO exercises

After the `vision` profile and the render/Flyvis checks pass, the synthetic M4 exercise can create a new visual head without using the factory browser bridge:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_visual.py --frames 4000 --dagger-frames 1500 --rounds 2
```

It writes `checkpoints/training/M4-visual-head.pt` and `outputs/training/M4-*.json` / `.npz`. It uses MuJoCo's synthetic camera scene, not the Hall scan. Its optional DAgger rounds run only when the script's validation condition calls for them. Rerunning this exercise can replace its fixed-name M4 files; preserve a copy if you need an earlier student's result.

The separate PPO curriculum uses ideal position and synthetic range inputs. It is useful as a comparison exercise, but it is not the Flyvis Hall model:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_curriculum.py --help
```

Read the stage options before running a curriculum. Do not combine a PPO result, a synthetic landing result, and a Hall visual result into one success-rate claim; they use different inputs, environments, and evaluation tasks.

## 14. Troubleshooting: start with the first failed checkpoint

| Symptom | Likely cause | What to do |
|---|---|---|
| No `simulation/` folder after cloning | Older website-only revision or wrong repository | At the repository root, run `git pull`; confirm the updated file tree. |
| `py` is not recognized | Python launcher missing or terminal opened before installation | Install Python 3.12 x64 with the launcher, then reopen PowerShell. |
| `No suitable Python runtime found` | Python 3.12 is not installed | Install 3.12; `py -3.12 --version` must work first. |
| `No matching distribution found` | Wrong Python version, unsupported architecture, unavailable pinned wheel, or restricted package index | Confirm 3.12 / 64-bit, use the student requirements, and save the package name and full pip error. Do not blindly upgrade Python. |
| `ModuleNotFoundError` after installation | Running system Python instead of `.venv` Python, or an incomplete profile | Use the exact `.\.venv\Scripts\python.exe` path and install the required profile. |
| PowerShell blocks `Activate.ps1` | Activation policy | No activation is needed; use the explicit Python path in this guide. |
| Missing `cf2.xml`, meshes, or `robot_hall.ply` | Model/scene assets were not downloaded, or school network blocked the host | Rerun the corresponding setup profile and inspect the download error. |
| `Official pretrained Flyvis model missing` | Package installed but pretrained weights missing | Run `scripts/download_flyvis_models.py` using `.venv` Python, or rerun the `vision` profile. |
| `Source checkpoint is missing` | Evidence not restored or wrong source path | Run the restoration helper and use the explicit source-checkpoint argument in Section 10C. |
| DLL load error / WinError 126 | Architecture mismatch, missing Microsoft runtime, or incompatible binary package | Check x64 Python and package versions; have school IT install the official Microsoft Visual C++ x64 runtime if needed. Preserve the exact DLL error. |
| GLFW / OpenGL error or black MuJoCo image | GPU driver / graphics context problem | Test physics-only first; update the official GPU driver and try a normal local desktop session rather than a restricted remote session. |
| Browser WebGL error or lost context | Browser graphics acceleration unavailable or interrupted | Enable hardware acceleration, restart the browser, and reload the camera tab. Check the page's error text. |
| `Connection refused` | Camera server is not running or wrong port | Start Terminal A; use the same `8770` port in both commands. |
| Camera request timeout while the server is running | Camera tab closed, not ready, suspended, or failed to load its asset | Open `/bridge`, wait for readiness, keep the tab available and computer awake, and read its error. The Hall trainer has no fake-image fallback. |
| WinError 10048 / address already in use | Another server already occupies the port | Reuse the intended server or stop that terminal with `Ctrl+C`; if changing a camera port, change both the server and trainer. |
| `PermissionError` | Folder permissions, synchronization, or another process holding the file | Close the relevant reader/writer and use a school-approved writable local folder. |
| `UnicodeDecodeError` or garbled Chinese | Different default Windows text encoding | Use `-X utf8` in the exact commands. Save edited scripts and JSON as UTF-8. |
| Download checksum mismatch | Incomplete download, login/proxy page instead of the asset, or changed upstream file | Stop and check the source/network. Do not remove the checksum check. |
| Dashboard does not change | Only a display server is running, or the last completed run is being shown | Start Terminal C and check its new run ID, terminal output, and `latest.json`. |
| New training result is worse than 20/20 | Different seeds, packages, numerical behavior, or a weaker fitted candidate | Keep the measured result and configuration. A previous experiment's score is not a guarantee. |

For a teacher or bug report, include: Windows version, CPU/GPU, `py -3.12 --version`, doctor output, `student-requirements.txt`, exact command, full error traceback, and the run's `run-config.json` / `status.json` when present. Do not send credentials or private files. [Microsoft runtime documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

## 15. Files to read when learning or changing the project

| File or folder under `simulation/` | Read it to understand |
|---|---|
| `scripts/setup_student.py` | Environment setup and staged downloads. |
| `scripts/doctor.py` | Installation, rendering, and Flyvis preflight checks. |
| `scripts/flyvis_example.py` | A minimal executable Flyvis feature example. |
| `src/fruitfly_sim/env.py` | Crazyflie physics, stabilization, basic Gymnasium observations and actions. |
| `src/fruitfly_sim/flyvis_features.py` | RGB preprocessing, official model loading, temporal state, and 72-value features. |
| `src/fruitfly_sim/visual_env.py` | The 78-value visual policy input and teacher labels. |
| `src/fruitfly_sim/splat_env.py` | Hall camera requests, coordinate offset, and approximate collision proxy. |
| `scripts/train_visual.py` | `VisualHead`, its Tanh layers, and initial synthetic visual training. |
| `scripts/train_hall.py` | Factory collection, supervised fitting, validation selection, audit, and stale-vision control. |
| `scripts/serve_render_bridge.py` + `dashboard/render-bridge.html` | Python ↔ browser camera communication. |
| `scripts/serve_missions.py` + `dashboard/missions.html` | Live training dashboard. |
| `scripts/render_hall_observer.py` | Independent third-person video rendering. |
| `scripts/export_hall_replay.py` | Saved trajectory validation and 3DGS replay export. |
| `tests/` | Automated checks; graphics tests additionally require a working renderer. |

Change one factor at a time, keep separate run directories, and compare on a planned held-out test set. Good first extensions include additional single-obstacle positions, then multiple obstacles, and later landing. A new school's 3DGS scan needs coordinate/scale alignment and collision geometry before you can claim trained obstacle avoidance there.

## 16. Report wording, evidence, and licenses

The published `hall-20260922-replay1` run collected 2,400 training samples. Validation selected round 0, which fitted the first 1,600 samples; the additional 800-sample DAgger candidate was retained as an unsuccessful comparison. The final selected model reached the goal in **20/20** held-out trials with **0 recorded proxy collisions**. Its minimum saved-trajectory horizontal body-envelope clearance was approximately **0.287 m**. The source baseline and selected model both achieved 8/8 on the same validation trials, so this does not establish a validation success-rate improvement.

These findings apply to one static factory scene, one manually fitted collision cylinder, a narrow start/goal range, ideal goal/velocity information, and fixed-height planar flight. They do not establish full-building safety, metric depth perception, outdoor return-to-home ability, school-scene performance, real-drone transfer, or 4DGS operation. The unchanged-first-frame comparison measures dependence on updating visual features; it is not by itself proof that the network learned depth.

A suitable methods sentence is: “We used a frozen pretrained Flyvis visual network to extract motion-related features and trained a small navigation head by imitation learning, then evaluated it in a 3D Gaussian Splatting visual scene coupled to MuJoCo physics.” Replace reported numbers with your own run's measurements when describing your experiment.

| Asset | Handling |
|---|---|
| Flyvis | Follow the [official repository](https://github.com/TuragaLab/flyvis), paper, package license, and model provenance. Keep the original source citation. |
| Crazyflie model | Downloaded from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Preserve the model's included license and provenance. |
| TUM Flight Hall scan | Source is marked **All Rights Reserved**. The setup downloads it individually for the exercise; the scan is not bundled in the public teaching package. Access is not a redistribution license. Check the source terms before publishing scene-derived media or sharing the scan. |
| Dajing school scene | The existing published scene uses CC BY 4.0 attribution. Keep that attribution; the scene is a separate preview, not a trained result. |
| Student results | Keep source run IDs, settings, seeds, failed trials, model hashes, and the distinction between observed results and future tasks. |

The published evidence package contains recorded data, models, figures, videos, and a source snapshot; it does not contain every original RGB training frame or all external model assets. See the [research data room](https://xiaoyh-code.github.io/fruitfly-flight-lab/research/hall-20260922-replay1/) and [future tasks](https://xiaoyh-code.github.io/fruitfly-flight-lab/roadmap.html). Publishing a new result is a separate teacher-reviewed step; running these commands does not push changes to GitHub.
