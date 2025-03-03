# MIOU 分割作業

## 專案概述
本專案利用 PyTorch 實作一個基於 U-Net 的影像分割模型，目標是對 DRIVE 資料集中的影像進行分割，並利用 IoU（交集比並，Intersection over Union）與 F1 分數對模型進行評估。模型訓練共 400 個 epochs，最終會輸出分割結果以及量化的評估指標。

---

## 主要特色
- **U-Net 架構**：採用雙重卷積層（DoubleConv）作為基本模組，構建 U-Net 模型。
- **自訂資料集類別**：實作 `DRIVE_Dataset` 類別，便於讀取 DRIVE 資料集中的影像、真實標記（mask）以及 FOV mask。
- **模型訓練**：使用 BCEWithLogitsLoss 搭配 Adam 優化器進行訓練。
- **評估與視覺化**：計算 IoU 和 F1 分數，並利用 Matplotlib 同時顯示原始影像、分割結果和真實標記。
- **結果存檔**：將所有評估指標存入 CSV 檔案（segmentation_metrics.csv）。

---

## 系統需求
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pandas
- Pillow (PIL)

---

## 資料集
本專案使用的資料集為 DRIVE：
- **訓練資料**：位於 `.../archive/DRIVE/training/` 目錄下，包含：
  - `images`：原始影像。
  - `1st_manual`：手動標記的真實分割 mask。
  - `mask`：FOV mask（用於排除無效區域）。
- **測試資料**：位於 `.../DIRVE_TestingSet` 目錄下（請確認實際路徑是否正確）。

請根據實際情況更新程式中資料集的路徑。

---

## 使用說明

### 1. 資料準備
- 更新程式中訓練與測試資料的路徑，確保能正確讀取影像與標記檔案。

### 2. 模型訓練
- 執行程式開始模型訓練：
  ```bash
  python your_script.py
  ```
- 程式將進行 400 個 epochs 的訓練，並於每個 epoch 結束時輸出平均損失值。

### 3. 模型評估
- 訓練完成後，程式將在測試資料上評估模型，計算每張影像的 IoU 與 F1 分數。
- 程式會利用 Matplotlib 顯示每張影像的三個視圖：
  - 原始影像
  - 預測的分割結果（經過 0.5 閾值處理）
  - 真實標記（Ground Truth）
- 所有評估指標將存入 CSV 檔案 `segmentation_metrics.csv`。

---

## 模型架構
本專案採用 U-Net 模型，其主要結構包括：
- 下採樣路徑：由多層 DoubleConv 模組與 MaxPool2d 組成，用於提取影像特徵並降低空間解析度。
- 瓶頸層：最深層的 DoubleConv 模組，負責整合下採樣後的特徵。
- 上採樣路徑：利用 ConvTranspose2d 進行上採樣，並與對應層的特徵圖通過跳接（skip connection）進行拼接。
- 輸出層：通過 1x1 卷積產生最終分割結果。

---

## 訓練細節
- 損失函數：使用 BCEWithLogitsLoss，並結合 FOV mask 進行針對性訓練。
- 優化器：Adam，學習率設為 1e-4。
- 訓練 Epoch：共 400 個 epochs。

---

## 評估指標
- IoU（交集比並）：計算預測結果與真實標記之間的交集與聯集比值。
- F1 分數：根據精確率與召回率計算出的調和平均值，綜合衡量分割效果。

---

## 視覺化
在評估階段，程式會生成一個包含多個子圖的圖形，分別顯示：
- 原始影像
- 預測的分割結果
- 真實標記 這有助於直觀評估模型分割效果。

---

## 輸出結果
- 訓練過程：每個 epoch 輸出平均損失值。
- 模型評估：列印每張影像的 IoU 與 F1 分數，並計算整體平均 IoU。
- CSV 檔案：所有分割評估指標將存入 `segmentation_metrics.csv`。
