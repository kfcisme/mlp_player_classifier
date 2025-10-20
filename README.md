# mlp_player_classifier

以 **Multi-Layer Perceptron (MLP)** 為核心的玩家行為分類器。此專案承接你先前的 K-Means 分群與特徵工程成果，將玩家的時窗特徵（每 30–60 分鐘或每小時）輸入監督式分類模型，輸出 **玩家類型標籤**（如：建築、探險、生存、紅石、競技、破壞者、社交、掛機），並提供可重現的訓練/推論流程，方便後續銜接 **伺服器負載預測** 與 **動態節能調節**。

> 建議與 [`2026TISF_Kmeans`](https://github.com/kfcisme/2026TISF_Kmeans) 共同使用：以分群/規則產生的弱標註作為初始標籤，再用 MLP 微調與強化。

---

## 功能亮點

- **端到端監督式訓練流程**：資料載入 → 前處理/特徵縮放 → MLP 訓練 → 評估 → 匯出模型。  
- **可配置化**：以 `config/*.yaml` 管理特徵欄位、標籤對映、模型超參數（層數、寬度、Dropout、L2、學習率、批次大小、Epochs 等）。  
- **多種評估指標**：`accuracy`、`precision/recall/f1 (macro/weighted)`、混淆矩陣、ROC-AUC（多類 One-vs-Rest）。  
- **產物可追溯**：自動保存 `runs/`（權重、最佳檢查點、學習曲線、指標報表、confusion matrix PNG、分類報告）。  
- **易於部署**：提供 `inference.py` 與 `serve.py`（可選）進行離線批次或簡易 API 服務。

---

## 專案結構（建議）

> 下列為建議結構；若你的 repo 已有不同命名，請以實際檔名為準。

```
mlp_player_classifier/
├─ config/
│  ├─ experiment_default.yaml       # 模型與訓練參數、特徵清單、標籤對映
│  └─ cluster_to_label.yaml         # （可選）從 K-Means Cluster 映射到最終玩家類型
├─ data/
│  ├─ train/                        # 訓練資料（CSV/Parquet）
│  ├─ valid/                        # 驗證資料
│  └─ test/                         # 測試資料
├─ src/
│  ├─ dataset.py                    # 資料讀取/切分/標準化
│  ├─ features.py                   # 特徵工程與前處理（Winsor/Scaler/Log 等）
│  ├─ model.py                      # MLP 架構定義（PyTorch 或 scikit-learn MLPClassifier）
│  ├─ trainer.py                    # 訓練/早停/儲存最佳模型
│  ├─ metrics.py                    # 評估與報表輸出
│  ├─ infer.py                      # 單批/單檔推論工具
│  └─ utils.py                      # 公用函式（隨機種子、路徑、記錄器）
├─ notebooks/                       # Demo 與探索式分析
├─ runs/                            # 訓練輸出（自動建立）
├─ requirements.txt                 # 相依套件（可用下方最小清單）
├─ train.py                         # CLI：執行訓練/評估
├─ inference.py                     # CLI：批次推論
├─ serve.py                         # （可選）簡易 API 服務（FastAPI/Flask）
├─ README.md                        # 本說明文件
└─ LICENSE                          # 授權（建議：Apache-2.0）
```

---

## 安裝與環境

- Python 3.10+（建議）
- 最小相依（若使用 PyTorch 版）
  ```bash
  pip install -U pandas numpy scikit-learn torch torchvision torchaudio matplotlib pyyaml joblib
  ```
  或（若使用 scikit-learn `MLPClassifier` 版本）
  ```bash
  pip install -U pandas numpy scikit-learn matplotlib pyyaml joblib
  ```

> 若需匯出/讀取 Excel：`openpyxl`；若啟用 API：`fastapi uvicorn`。

---

## 資料格式

- **輸入資料（CSV/Parquet）**：每列代表「玩家 × 時窗」，包含：
  - `player_id, ts_window_start`（或 `ts`）、多個行為特徵欄位（如：`blocks_placed, blocks_broken, chunk_loads, tnt_exploded, entity_kills, items_picked, items_dropped, container_interactions, chat_count, afk_minutes, active_minutes, ...`）
  - `label`（字串或整數類別編碼）。若無標籤，可先由 `2026TISF_Kmeans` 產生弱標籤。

### 標籤範例（建議）
```yaml
# config/cluster_to_label.yaml
0: "Builder"      # 建築玩家
1: "Explorer"     # 探險玩家
2: "Survivor"     # 生存玩家
3: "Redstone"     # 紅石玩家
4: "PvP"          # 競技型玩家
5: "Griefer"      # 破壞者
6: "Social"       # 社交玩家
7: "AFK"          # 掛機玩家
```

---

## 快速開始

### 1) 準備資料與設定
```bash
# 建立資料夾
mkdir -p data/train data/valid data/test runs config

# 放入你的 CSV/Parquet 到 data/*/
# 建立一份設定檔
cat > config/experiment_default.yaml << 'YAML'
seed: 42
task: "multiclass"
label_column: "label"
id_columns: ["player_id", "ts_window_start"]

features:
  - blocks_placed
  - blocks_broken
  - chunk_loads
  - tnt_exploded
  - entity_kills
  - items_picked
  - items_dropped
  - container_interactions
  - chat_count
  - afk_minutes
  - active_minutes

preprocess:
  winsor_p: 0.99
  log1p:    ["blocks_placed", "blocks_broken", "chunk_loads"]
  scaler:   "standard"   # standard / robust / minmax

model:
  type: "pytorch"        # pytorch / sklearn
  hidden_sizes: [128, 64]
  dropout: 0.2
  l2: 1.0e-4

train:
  batch_size: 512
  lr: 1.0e-3
  epochs: 50
  early_stopping_patience: 8

eval:
  save_confusion_matrix: true
  save_classification_report: true
YAML
```

### 2) 訓練
```bash
python train.py --config config/experiment_default.yaml   --train_dir data/train --valid_dir data/valid --out_dir runs/exp1
```

### 3) 測試
```bash
python train.py --config config/experiment_default.yaml   --test_dir data/test --resume runs/exp1/best.ckpt --only_test
```

### 4) 推論（批次）
```bash
python inference.py   --model runs/exp1/best.ckpt   --input data/new_unlabeled.parquet   --output runs/exp1/predictions.parquet
```

### 5)（可選）啟動簡易 API
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
# POST /predict  with JSON rows -> 返回玩家類型預測
```

---

## 評估與輸出

- 主要輸出皆置於 `runs/exp*/`：
  - `best.ckpt` / `best_model.joblib`（模型權重）
  - `metrics.json`（accuracy、macro/weighted F1 等）
  - `confusion_matrix.png`、`learning_curve.png`
  - `classification_report.txt`
- 建議在 README 或論文中同步回報：
  - 交叉驗證結果（K-fold）與標準差
  - 不同特徵組/前處理對結果的影響
  - 與基準模型（僅用玩家人數、或簡單規則）的比較

---

## 與負載預測/節能調節的整合

- 將 **每時窗的玩家類型占比** 匯出為時間序列特徵，給 MLR/LSTM 等負載模型使用（例如：`CPU/RAM/Power ~ 玩家類型占比 + 交互項`）。  
- 在 Auto-Scaler 中根據「當前與短期預測的玩家組成」決定：
  - 應開啟的伺服器數量與配置（Paper/Folia/Velocity 節點）
  - 玩家分配策略（加權貪心、避免 TPS 下降與單機過載）

---

## 小技巧與實務建議

- 固定 `random_state/seed`，記錄資料版本（CSV 匯出時間、Git commit）。  
- 異常值處理：Winsorization + log1p 對重尾分佈（如 `chunk_loads`）常有幫助。  
- 類別不平衡：可嘗試 `class_weight="balanced"`（SKL）或加權 CrossEntropy（PyTorch）。  
- 觀察錯分：混淆矩陣搭配典型樣本，找出可再切的特徵（如 `active_minutes/afk_minutes` 比值）。  
- 以 `Pipeline` 封裝前處理，避免訓練/推論特徵縮放不一致。

---

## 路線圖（Roadmap）

- [ ] 加入 `Optuna` / `Ray Tune` 進行超參數最佳化。  
- [ ] 加入 `SHAP` / `Permutation Importance` 做可解釋性分析。  
- [ ] 與 `2026TISF_Kmeans` 自動銜接：一鍵從弱標註→監督訓練→匯出占比。  
- [ ] 提供 `Dockerfile` 與 `docker-compose.yml`。  
- [ ] 支援 `ONNX` 匯出與推論。

---
