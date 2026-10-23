# ML/DL 模型學習路線與練習資料集

Oct 18, 2026 · @BlackUmbrella

## 使用說明

依表格由上而下學：每一列先讀懂原理，再用推薦資料集實作一次。資料集都挑小而乾淨、筆電或免費 Colab 就跑得動的版本。

- **資料集來源**：`sklearn` 內建（`from sklearn.datasets import ...`）、`torchvision` / `torchtext`、Hugging Face Datasets（`load_dataset("名稱")`）、Kaggle、UCI Machine Learning Repository。
- **每列建議流程**：讀原理 → 用套件跑通 → 調參數觀察變化 → 與上一個模型比較成績。
- **標記「選讀」**：了解概念即可，不必實作。
- 需要大量算力或沒有適合入門資料集的模型放在最後的附錄，先略過。

## 一、經典機器學習（1950s–2000s）

這一段是所有後續模型的基礎，全部可用 `scikit-learn` 實作，資料集多半一行就能載入。

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| 感知器 Perceptron | 1958 | 線性分界、錯誤驅動更新 | 多層感知器 MLP | Iris（只取兩類） | sklearn `load_iris` | ☐ |
| 線性回歸 | — | 最小平方法、梯度下降 | Ridge、Lasso、Elastic Net | California Housing | sklearn `fetch_california_housing` | ☐ |
| 邏輯回歸 | — | Sigmoid、交叉熵 | Softmax 多分類 | Breast Cancer Wisconsin | sklearn `load_breast_cancer` | ☐ |
| KNN | 1967 | 距離最近的 K 個鄰居投票 | KD-Tree、Ball Tree、近似最近鄰（FAISS、HNSW） | Digits（8×8 手寫數字） | sklearn `load_digits` | ☐ |
| 決策樹 | 1986 | 資訊增益／Gini 分裂 | ID3 → C4.5 → CART | Titanic | sklearn `fetch_openml("titanic")` | ☐ |
| 樸素貝氏 | — | 貝氏定理＋特徵獨立假設 | Gaussian、Multinomial、Bernoulli NB | SMS Spam Collection | UCI | ☐ |
| SVM | 1995 | 最大間隔、Kernel Trick | SVR、One-Class SVM | Wine | sklearn `load_wine` | ☐ |
| Random Forest | 2001 | Bagging＋隨機特徵 | Extra Trees | Adult（Census Income） | UCI 或 sklearn `fetch_openml("adult")` | ☐ |
| AdaBoost | 1995 | 加權弱分類器 | — | Breast Cancer Wisconsin | sklearn `load_breast_cancer` | ☐ |
| GBDT | 2001 | 擬合殘差的梯度提升 | XGBoost、LightGBM、CatBoost | House Prices（Ames 房價） | sklearn `fetch_openml("house_prices")` | ☐ |
| K-Means | 1967 | 迭代更新群中心 | K-Means++、Mini-Batch | Wholesale Customers | UCI「Wholesale customers」 | ☐ |
| DBSCAN | 1996 | 密度相連成群 | HDBSCAN | sklearn 合成資料 | sklearn `make_moons` | ☐ |
| GMM | — | EM 演算法、軟分群 | — | Iris | sklearn `load_iris` | ☐ |
| PCA | 1901 | 最大變異方向投影 | Kernel PCA | Digits | sklearn `load_digits` | ☐ |
| t-SNE／UMAP | 2008／2018 | 保留局部鄰近結構的非線性降維 | — | MNIST | `fetch_openml("mnist_784")` | ☐ |
| HMM | 1960s | 隱藏狀態＋轉移機率 | — | 選讀 | — | ☐ |
| CRF | 2001 | 序列標註的條件機率 | BiLSTM-CRF | CoNLL-2003（NER） | Hugging Face `conll2003` | ☐ |

## 二、早期神經網路（1980s–1990s）

從這裡開始改用 PyTorch；建議先用 NumPy 手刻一次 MLP 的反向傳播，之後的模型都會好懂很多。

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP＋反向傳播 | 1986 | 鏈鎖律計算梯度、非線性激活函數 | 各種激活函數（Sigmoid、Tanh、ReLU） | MNIST | `torchvision.datasets.MNIST` | ☐ |
| RNN | 1986 | 隱藏狀態在時間步之間傳遞 | Bi-RNN | 名字分類（判斷姓名語言） | PyTorch 官方教學「NLP From Scratch」附的資料 | ☐ |
| LSTM | 1997 | 輸入、遺忘、輸出閘門解決梯度消失 | Bi-LSTM、Stacked LSTM | IMDB 影評情感分類 | Hugging Face `imdb` | ☐ |
| GRU | 2014 | 精簡為更新、重置兩個閘門 | — | IMDB（與 LSTM 比較） | Hugging Face `imdb` | ☐ |
| LeNet-5 | 1998 | 卷積＋池化＋全連接 | — | MNIST | `torchvision.datasets.MNIST` | ☐ |

## 三、深度學習復興（2006–2016）

CNN 系列建議都用 CIFAR-10 訓練，方便直接比較各架構的準確率；大型架構可改用 `torchvision.models` 的預訓練權重做遷移學習。

### 影像分類 CNN

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| AlexNet | 2012 | ReLU、Dropout、GPU 訓練 | — | CIFAR-10 | `torchvision.datasets.CIFAR10` | ☐ |
| VGG | 2014 | 堆疊 3×3 小卷積核 | VGG16、VGG19 | CIFAR-10 | 同上 | ☐ |
| GoogLeNet／Inception | 2014 | 多尺度並行卷積、1×1 卷積降維 | Inception v2–v4 | CIFAR-10 | 同上 | ☐ |
| ResNet | 2015 | 殘差連接 | ResNeXt、Wide ResNet | CIFAR-10 | 同上 | ☐ |
| DenseNet | 2017 | 每層連接所有前層 | — | CIFAR-10 | 同上 | ☐ |
| MobileNet | 2017 | 深度可分離卷積 | MobileNet v2/v3、ShuffleNet | Oxford Flowers 102（遷移學習） | `torchvision.datasets.Flowers102` | ☐ |
| EfficientNet | 2019 | 寬度、深度、解析度等比縮放 | EfficientNet v2 | Oxford-IIIT Pet（遷移學習） | `torchvision.datasets.OxfordIIITPet` | ☐ |
| ConvNeXt | 2022 | 借鏡 Transformer 設計的 CNN | — | Oxford-IIIT Pet（遷移學習） | 同上 | ☐ |

### 偵測與分割

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN | 2015 | 兩階段：區域提議＋分類 | R-CNN → Fast R-CNN → Faster R-CNN | Penn-Fudan 行人 | PyTorch 官方「TorchVision Object Detection」教學 | ☐ |
| YOLO | 2016 | 單階段，整張圖一次預測框 | 多個後續版本、SSD | COCO128（COCO 小樣本） | Ultralytics 套件內建 | ☐ |
| FCN | 2015 | 全卷積做逐像素分類 | — | 選讀 | — | ☐ |
| U-Net | 2015 | 編碼器—解碼器＋跳接 | U-Net++ | Oxford-IIIT Pet（分割遮罩） | `torchvision.datasets.OxfordIIITPet` | ☐ |
| Mask R-CNN | 2017 | Faster R-CNN＋遮罩分支 | — | Penn-Fudan 行人 | 同 Faster R-CNN 教學 | ☐ |

### 詞向量、序列與生成模型

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Word2Vec | 2013 | 由上下文學詞向量（CBOW、Skip-gram） | GloVe、FastText | text8 | `gensim` 下載器 | ☐ |
| Seq2Seq＋Attention | 2014–15 | 編碼器—解碼器、注意力對齊 | Bahdanau、Luong Attention | 英法翻譯句對 | PyTorch 官方「Seq2Seq Translation」教學 | ☐ |
| Autoencoder | 1980s | 壓縮再重建 | Denoising AE | Fashion-MNIST | `torchvision.datasets.FashionMNIST` | ☐ |
| VAE | 2013 | 潛在空間機率分布＋重參數化 | β-VAE、VQ-VAE | Fashion-MNIST | 同上 | ☐ |
| GAN | 2014 | 生成器與判別器對抗 | DCGAN、WGAN | MNIST | `torchvision.datasets.MNIST` | ☐ |
| DCGAN | 2015 | 用卷積建 GAN | StyleGAN（選讀） | CelebA（縮小版） | `torchvision.datasets.CelebA` | ☐ |
| CycleGAN | 2017 | 無配對影像轉換、循環一致性 | Pix2Pix（配對版） | horse2zebra | TensorFlow Datasets `cycle_gan` | ☐ |

關鍵訓練技術（Adam、Batch Normalization、Dropout、資料增強）不需要獨立資料集，直接在上面的 CIFAR-10 實驗中開關比較即可。

## 四、Transformer 時代（2017 至今）

這一段以「微調小型預訓練模型」為主，不從頭訓練；Hugging Face `transformers` 套件幾乎都能直接載入。

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | 2017 | Self-Attention、多頭注意力、位置編碼 | — | 手刻小型字元級模型：Tiny Shakespeare | Karpathy「nanoGPT」專案內附 | ☐ |
| BERT | 2018 | 雙向 Encoder、遮罩語言模型預訓練 | RoBERTa、ALBERT、DistilBERT、DeBERTa | SST-2 情感分類 | Hugging Face `glue`（`sst2`） | ☐ |
| GPT（GPT-1/2） | 2018–19 | Decoder 自回歸生成 | GPT-3 之後的大模型（附錄） | 微調 GPT-2 small：Tiny Shakespeare | 同上 nanoGPT | ☐ |
| T5 | 2019 | 所有任務轉成文字到文字 | Flan-T5 | CNN/DailyMail 摘要（取子集） | Hugging Face `cnn_dailymail` | ☐ |
| BART | 2019 | 去噪 Encoder-Decoder | — | XSum 摘要（取子集） | Hugging Face `xsum` | ☐ |
| ViT | 2020 | 圖片切 patch 當 token | DeiT、Swin Transformer | CIFAR-10（微調預訓練 ViT） | `torchvision.datasets.CIFAR10` | ☐ |
| DETR | 2020 | 用 Transformer 直接預測物件集合 | Deformable DETR | 選讀 | — | ☐ |
| CLIP | 2021 | 圖文對比學習、零樣本分類 | BLIP、LLaVA | CIFAR-10 零樣本分類（不需訓練） | 同上；模型用 Hugging Face `openai/clip-vit-base-patch32` | ☐ |
| LoRA／QLoRA | 2021／2023 | 只訓練低秩矩陣的高效微調 | PEFT 套件其他方法 | Alpaca 指令資料，微調小型開源 LLM | Hugging Face `tatsu-lab/alpaca` | ☐ |
| RAG | 2020 | 檢索外部文件再生成 | 向量資料庫、重排序 | 自己的 PDF 或筆記 | 自備 | ☐ |

實作 Transformer 時，強烈建議跟著 nanoGPT 從零寫一次，Self-Attention 會一次弄懂。

## 五、擴散模型（2020 至今）

從頭訓練只建議做 DDPM 小實驗；Stable Diffusion 一類改用 Hugging Face `diffusers` 推論與微調。

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| DDPM | 2020 | 逐步加噪再學習去噪 | DDIM（加速取樣） | MNIST 或 Fashion-MNIST | `torchvision.datasets` | ☐ |
| Latent Diffusion／Stable Diffusion | 2022 | 在潛在空間做擴散 | Classifier-Free Guidance | 不需資料，直接推論 | `diffusers` 預訓練模型 | ☐ |
| ControlNet | 2023 | 加入邊緣、姿勢等條件控制 | — | 不需資料，直接推論 | `diffusers` | ☐ |
| DreamBooth／LoRA 微調 | 2022–23 | 用少量圖片教模型新主題或風格 | — | 自己拍的 5–20 張照片 | 自備 | ☐ |
| DiT、Flow Matching | 2023 起 | Transformer 骨幹、直線化生成路徑 | Rectified Flow、Consistency Models | 選讀 | — | ☐ |

## 六、其他分支：強化學習與圖神經網路

強化學習用 `gymnasium` 環境就不需要資料集；GNN 建議用 PyTorch Geometric（PyG），內建經典圖資料。

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集／環境 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Q-Learning | 1989 | 查表更新動作價值 | SARSA | FrozenLake | `gymnasium` | ☐ |
| DQN | 2013／2015 | 神經網路近似 Q 值、經驗回放 | Double DQN、Dueling DQN | CartPole | `gymnasium` | ☐ |
| Policy Gradient | 1992 | 直接優化策略（REINFORCE） | A2C、A3C | CartPole | `gymnasium` | ☐ |
| PPO | 2017 | 限制每次策略更新幅度 | TRPO；後來用於 RLHF | LunarLander | `gymnasium`（套件 `stable-baselines3` 有現成實作） | ☐ |
| AlphaGo／AlphaZero | 2016／2017 | 蒙地卡羅樹搜尋＋神經網路 | MuZero | 選讀（可用井字棋自製簡化版） | — | ☐ |
| GCN | 2017 | 鄰居特徵加權聚合 | — | Cora 論文引用網路 | PyG `Planetoid("Cora")` | ☐ |
| GraphSAGE | 2017 | 取樣鄰居、可處理新節點 | — | Cora 或 PubMed | PyG `Planetoid` | ☐ |
| GAT | 2018 | 用注意力決定鄰居權重 | GATv2 | Cora | PyG `Planetoid` | ☐ |

## 七、時間序列模型

建議統一用 ETTh1 當深度學習段的共用資料集，就能把 LSTM 到 Chronos 的成績放在一起比；每個模型都先和「季節性 naive」基準比較。套件：`statsmodels`、`statsforecast`、`neuralforecast`、`darts`。

### 統計模型

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| 指數平滑 | 1950s | 近期觀測權重較高 | Holt、Holt-Winters、ETS | AirPassengers（航空乘客月資料） | `sktime` 的 `load_airline` 或 Kaggle | ☐ |
| ARIMA | 1970 | 自回歸＋移動平均＋差分 | SARIMA、ARIMAX、Auto-ARIMA | AirPassengers | 同上 | ☐ |
| VAR | 1980 | 多條序列互相回歸 | VECM | 美國總經資料 | `statsmodels.datasets.macrodata` | ☐ |
| ARCH／GARCH | 1982／1986 | 建模波動率隨時間變化 | EGARCH、GJR-GARCH | S&P 500 日報酬率 | `yfinance` 套件下載；模型用 `arch` 套件 | ☐ |
| Kalman Filter | 1960 | 狀態空間的預測＋校正 | Extended／Unscented KF | 自己模擬的帶雜訊軌跡 | NumPy 生成 | ☐ |
| STL 分解 | 1990 | 拆成趨勢、季節、殘差 | MSTL（多季節） | AirPassengers | 同上 | ☐ |
| Prophet | 2017 | 加法分解＋節假日效應 | NeuralProphet | Peyton Manning 維基百科瀏覽量 | Prophet 官方範例資料 | ☐ |

### 機器學習方法

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM／XGBoost＋特徵工程 | 2014 起 | lag、滾動統計、日期特徵轉成表格問題 | `mlforecast` 套件 | Store Sales（厄瓜多零售） | Kaggle「Store Sales - Time Series Forecasting」 | ☐ |

### 深度學習模型

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM／GRU 預測 | 1997／2014 | 滑動視窗做序列到數值 | Seq2Seq 多步預測 | ETTh1（變壓器油溫） | GitHub「ETDataset」 | ☐ |
| DeepAR | 2017 | 輸出機率分布、多序列共同訓練 | — | Electricity | `gluonts` 內建資料集 | ☐ |
| TCN | 2018 | 空洞因果卷積 | WaveNet | ETTh1 | 同上 | ☐ |
| N-BEATS | 2019 | 純 MLP，可拆趨勢與季節 | N-HiTS | M4（取子集） | `neuralforecast` 內建 | ☐ |
| TFT | 2019 | LSTM＋Attention、支援已知未來變數 | — | Stallion 飲料銷售 | `pytorch-forecasting` 官方教學 | ☐ |
| Informer | 2021 | ProbSparse Attention | — | ETTh1 | 同上 | ☐ |
| Autoformer | 2021 | 序列分解＋自相關機制 | FEDformer（頻域） | ETTh1 | 同上 | ☐ |
| DLinear | 2023 | 分解後各接一層線性 | NLinear | ETTh1 | 同上 | ☐ |
| PatchTST | 2023 | 切 patch＋通道獨立 | — | ETTh1 | 同上 | ☐ |
| TimesNet | 2023 | 依週期轉二維再卷積 | — | ETTh1 | 同上 | ☐ |
| iTransformer | 2024 | 每個變數當一個 token | — | ETTh1 | 同上 | ☐ |
| TSMixer | 2023 | 純 MLP 混合時間與變數維度 | — | ETTh1 | 同上 | ☐ |

### 時序基礎模型（零樣本預測，安裝即可用）

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Chronos（Amazon） | 2024 | 數值量化成 token，套用語言模型 | Chronos-Bolt | ETTh1 或 AirPassengers | `chronos-forecasting` 套件 | ☐ |
| TimesFM（Google） | 2024 | 大規模預訓練的 Decoder 模型 | — | ETTh1 | `timesfm` 套件 | ☐ |

### 其他時序任務

| 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 完成 |
| --- | --- | --- | --- | --- | --- | --- |
| Isolation Forest | 2008 | 異常點較容易被隨機切開 | — | NAB（Numenta 異常基準） | GitHub「NAB」 | ☐ |
| LSTM Autoencoder | 2015 起 | 重建誤差大即異常 | Anomaly Transformer | NAB | 同上 | ☐ |
| DTW＋KNN | 1990s | 動態時間校正距離 | — | UCR 時序分類資料庫（如 GunPoint） | `aeon` 或 `sktime` 內建 | ☐ |
| ROCKET | 2020 | 大量隨機卷積核＋線性分類器 | MiniRocket、InceptionTime | UCR（如 GunPoint） | 同上 | ☐ |

## 附錄：暫時略過的模型與原因

以下模型需要大量算力、沒有適合入門的資料集，或已被淘汰，建議前面都完成後再回頭看論文或部落格解說。

| 模型 | 略過原因 | 完成 |
| --- | --- | --- |
| DBN／RBM | 已被現代訓練方法取代，歷史意義為主 | ☐ |
| StyleGAN、Sora 等影片生成 | 訓練需要大量 GPU，入門以閱讀為主 | ☐ |
| GPT-3 以後的大型語言模型、MoE、RLHF、DPO | 無法自行預訓練；可用 LoRA 微調小模型體驗 | ☐ |
| 推理模型（o1、DeepSeek-R1 類） | 以強化學習訓練，算力需求高；可直接使用 API 觀察行為 | ☐ |
| SAM、DINOv2、MAE | 可直接用預訓練模型推論，但原理建議先學完 ViT | ☐ |
| Mamba／SSM | 原理數學較重，實作工具仍在演進 | ☐ |
| NeRF、3D Gaussian Splatting | 需要多視角照片與 3D 背景知識 | ☐ |
| TimeGPT、Moirai、Lag-Llama 等其他時序基礎模型 | 與 Chronos、TimesFM 概念相近，選一兩個體驗即可 | ☐ |
| Swin Transformer、Deformable DETR | ViT、DETR 的延伸，先懂原版再看 | ☐ |

