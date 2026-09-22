# 原有工廠：Flyvis 視覺避障重訓

目前只採用最初的 TUM Flight Hall（`models/gaussian/robot_hall.ply`）。三個新下載的 CC0 掃描保留作備用，沒有加入本輪訓練。

## 2026-09-22 本機重訓與第三身 3DGS 回放

最新 run 是 `hall-20260922-replay1`，由 `hall-20260921-fast/hall-visual-head.pt` 繼續微調，Flyvis 保持凍結。約 191 秒完成 2,400 幀資料收集與評估。驗證選中第 0 輪、使用 1,600 幀的新權重；第 1 輪 DAgger 驗證退步，未選用。沒有根據正式 audit 選模型。

- 20 個新 audit seeds（2006208–2006227）：**20/20 到達、0/20 記錄碰撞**。
- 每段相鄰記錄位置之間的直線亦檢查機身包絡，20 條路徑皆未穿越代理；最小水平淨距 **0.28695449 m**。
- 同一模型固定首幀 Flyvis 特徵：0/20 到達、0/20 碰撞。這不是完整視覺測距或全工廠安全證明。
- 来源 checkpoint 及原 M4 保持不變；新訓練／驗證／audit seeds 與上一輪保留區間分開。來源模型與新模型在相同 8 個 validation seeds 都是 8/8 到達，因此不宣稱成功率提高。

[本機第三身避障回放](http://127.0.0.1:8766/?scene=hall&replay=1) 顯示真正 Crazyflie 網格及所有 20 次 audit，可播放／暫停、拖動時間軸、0.5× 慢播、切換「第三身跟機」與「路線全景」。機身維持原始 1× 尺寸，位置加上 `[15.15,-3.15,0]` 對齊工廠，姿態使用記錄的 MuJoCo 四元數；畫格間線性／SLERP 插值。回放不執行新策略推論或物理。

橙色圓柱是手動碰撞代理（中心 `[15.15,-2.95,0.775]`、半徑 0.45 m、高 1.55 m）；白圈是 0.09 m 水平機身包絡。MuJoCo 接觸檢查每 2 ms 執行，另外的軌跡淨距只保證所顯示的 10 Hz 記錄折線沒有穿越該代理。工廠其餘物件未建立完整碰撞網格。

新資料保留真實時間戳與機身姿態，共 1,059 個 audit poses。匯出器逐段驗證間距、源報告統計、時間、四元數與有限數值；18 個測試包含穿越障礙、部分終止步及失敗時不覆寫舊回放等情況：

```bash
.venv/bin/python scripts/export_hall_replay.py --run-id hall-20260922-replay1 --require-clear
.venv/bin/python scripts/serve_scene.py
# 開 http://127.0.0.1:8766/?scene=hall&replay=1
```

資料在 `outputs/scene-replay/hall-replay.json`；`--require-clear` 要求至少 20 次、全部到達、0 接觸、所有軌跡正淨距才會替換該檔案。新 checkpoint、audit 與 observer 影片在各自 run 目錄。seed 2006208 的獨立 MuJoCo 第三身影片已重跑，和原 audit 軌跡最大位置差為 0 m。

這次按要求只在本機運行與檢視，**尚未公開**。`publish/github-pages` 沒有更新，公開站仍保留之前的結果與手動擺位版本。

要重做這次設定，先啟動 8770 camera bridge 並保持瀏覽器 `/bridge` 開啟，然後用新的 run-id 執行：

```bash
.venv/bin/python scripts/train_hall.py --run-id YOUR_NEW_RUN \
  --source-checkpoint checkpoints/missions/hall-20260921-fast/hall-visual-head.pt \
  --seed-base 3000000 --torch-seed 922 --port 8770 \
  --frames 1600 --dagger-frames 800 --rounds 1 --epochs 60 --audit-episodes 20
```

再次重跑時應使用未用過的 seed 區間；本次實際 base 是 2000000。

## 2026-09-21 已完成結果

| 工廠測試 | 到達目標 | 代理碰撞 | 其他失敗 |
|---|---:|---:|---:|
| 更新 Flyvis 特徵的學習策略 | 20/20 | 0/20 | 0 |
| 固定第一幀特徵的同一策略 | 0/20 | 0/20 | 20 次越界 |

正常策略平均約 5.03 秒到達；最近的取樣機身外緣至代理間距約 0.31 m。這是同一局部場景的 20 個新起點／終點種子，不是全廠或真機驗收。對照顯示固定視覺特徵會破壞這個策略的到達能力，尚不能證明學會真實距離估計。

共收集 3,200 幀訓練標籤，依驗證結果選中的模型是第 0 輪（使用前 1,600 幀）；後兩輪的訓練資料和驗證結果仍保留。原始 M4 checkpoint 雜湊不變。全程實際傳回 7,996 張相機影像，取圖延遲中位數 10.16 ms、p95 13.31 ms；此數字只計相機，不代表整個控制迴路。

## 看進度與結果

- 雙擊專案根目錄的 **Open Factory Training.command**，或開 <http://127.0.0.1:8769/>。
- 左圖是真實工廠 3DGS 相機；右圖是 MuJoCo 碰撞代理示意。画面有拍攝時間，停止後顯示最後保存影像。
- 頁面可暫停、繼續、停止本輪；重開頁面不會重新訓練。
- 目前 run：`outputs/missions/hall-20260921-fast`。頁面由 `outputs/missions/latest.json` 選擇最新 run。
- 互動工廠：<http://127.0.0.1:8766/?scene=hall&view=4>。

## 放大 MuJoCo 觀察畫面

儀表板新增獨立的 960×540 觀察區，可切換 **近距離跟機**（`follow`，相機距離約 0.8 m）與 **路線總覽**（`overview`，按起點、終點及障礙範圍取景），亦可按 **全螢幕** 放大，再按 Esc 返回。

這些相機只供顯示（display only）。沒有放大無人機模型，沒有改動質量、碰撞範圍、物理狀態或模型接收的 320×240 真實 3DGS 相機；MuJoCo 觀察圖也不會送入 Flyvis。其障礙仍是簡化碰撞代理，不是完整工廠重建。

目前顯示已保存的 `hall-20260921-fast` 驗收 seed `1008808` 重播：51 幀、10 fps。重播使用相同權重及種子，與原始軌跡的最大位置差為 **0 m**；原始 audit 與 checkpoint 雜湊不變。這是該次驗收的放大回放，不代表重新訓練或新增一輪驗收。證據見該 run 的 `observer-preview.json`；影片與定格圖為 `observer-follow.mp4/.jpg` 及 `observer-overview.mp4/.jpg`。

如需重新產生這段觀察影片，先保留原有 Flight Hall 相機服務及 `http://127.0.0.1:8770/bridge` 分頁，再執行：

```bash
cd .
.venv/bin/python scripts/render_hall_observer.py --run-id hall-20260921-fast --seed 1008808 --port 8770
```

此指令只重播已完成的策略驗收並更新觀察影片，不會訓練或覆寫原始驗收報告。往後新訓練會另存近期的跟機／總覽定格畫面，頁面會標示其更新時間。

## 本輪學甚麼

實際 WebGL 工廠 RGB → 凍結的 Flyvis T4/T5 視覺迴路 → 78 輸入的決策網絡 → 水平速度指令 → MuJoCo。

網絡另外讀取理想相對目標位置、速度和上一次指令；沒有讀取障礙座標、射線或距離讀出器。只微調決策網絡，保留原始 M4 checkpoint。這是示範學習及 DAgger，不是整個果蠅大腦的自主強化學習。

1. 1,600 幀示範，以原有 M4 權重初始化。
2. 兩輪各 800 幀 DAgger：行動完全來自學習策略，示範器只提供訓練標籤。
3. 8 個驗證 seeds 選擇最佳輪次。
4. 20 個新 seeds 正式驗收；相同 20 個 seeds 另跑「固定第一幀視覺特徵」對照。

訓練／驗證／正式測試 seeds 分開，寫入 `seed-manifest.json`。以 12 秒內到達目標 0.25 m 範圍判定到達；本輪門檻為至少 80% 成功、至多 10% 代理碰撞。錄影只涵蓋正式驗收的第一個 seed，各次軌跡另存 JSON。

## 結果檔

- `hall_avoidance-audit.json`：正式驗收、每次結果及軌跡。
- `hall_avoidance-stale-vision.json`：固定視覺特徵對照。
- `hall_avoidance-evaluation.mp4`：第一個驗收 seed 的雙視窗錄影。
- `hall-training.json`：完成後的總結與最佳 checkpoint。
- `checkpoints/missions/hall-20260921-fast/hall-visual-head.pt`：獨立新權重。
- `run-config.json`：參數、來源雜湊、碰撞代理修正及取圖方式。

首次嘗試 `hall-20260921-first` 因背景分頁的非同步 PNG 匯出變慢，在第一段收集期間停止，沒有完成驗收。修正為同步匯出已完成排序的真實畫面後，以新 run 重新收集。現用相機分頁是 <http://127.0.0.1:8770/bridge>，請保留開啟；舊 8768 結果服務仍保留原始轉移測試。

## 限制與下一步

這一輪只有同一場景、固定高度、單個已核對的手動圓柱碰撞代理。工廠其餘牆面、桌椅未建立完整碰撞網格；成功不能代表可在全廠安全飛行。靜態 3DGS 也不是 4DGS。

新 seeds 主要改變局部起點／終點，不等於新場景泛化。固定視覺對照若同樣成功，代表任務可能靠位置及記住路線完成，不能據此宣稱已學識視覺測距。

後續先增加工廠內經核對的路線／物件碰撞範圍，再做多障礙及 Landing。Landing 必須新增可見平台、向下視覺及真實柔和落地判定，不能把接近地面或懸停算作落地。合成 MissionEnv 的 Landing 50/50 結果不是工廠視覺 Landing。

## 再跑另一輪

先啟動相機服務並在瀏覽器開啟其 `/bridge`；確認相機已準備好後，另開 Terminal：

```bash
cd .
.venv/bin/python scripts/serve_render_bridge.py --port 8770
# 在瀏覽器開 http://127.0.0.1:8770/bridge 後，另一個 Terminal 執行：
.venv/bin/python scripts/train_hall.py --port 8770
```

若 8770 已有本專案服務，直接使用既有服務。預設建立新的時間戳 run，不覆寫已完成結果；不用重新下載模型或場景。
