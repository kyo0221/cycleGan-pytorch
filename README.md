# cycleGAN-pytorch

天候変換（晴れ⇔曇り）を行うCycleGANのPyTorch実装です。教師なし学習により、ペア画像なしで画像から画像への変換を実現します。

## 概要

CycleGANは、対応するペア画像がない状況で、あるドメインの画像を別のドメインの画像に変換する深層学習モデルです。このプロジェクトでは、特に天候変換（晴天↔曇天）に特化した実装を提供しています。

## サンプル結果

### 変換前（晴天）→ 変換後（曇天）

| 変換前（Sunny） | 変換後（Cloudy） |
|:---:|:---:|
| ![Before](samples/before.png) | ![After](samples/after.png) |


## プロジェクト構成

```
cycleGan-pytorch/
├── cycle_gan.py          # 推論スクリプト
├── train.py              # 訓練スクリプト
├── config/               # 設定ファイル
│   ├── params.yaml       # 推論用パラメータ
│   └── train.yaml        # 訓練用パラメータ
├── models/               # モデル定義
│   └── networks.py       # ジェネレータ・識別器
├── util/                 # ユーティリティ
│   ├── loss.py          # 損失関数
│   ├── image_pool.py    # 画像プール
│   └── visualizer.py    # 可視化・ログ
├── checkpoints/         # 学習済みモデル
├── datasets/            # データセット
├── logs/               # 訓練ログ
└── README.md
```

## 必要環境

- Python 3.7+
- PyTorch 1.8+
- torchvision
- PIL (Pillow)
- PyYAML
- tqdm
- pathlib


## 使用方法

### 1. 推論（画像変換）

学習済みモデルを使って画像変換を実行：

```bash
python3 cycle_gan.py <画像ディレクトリ>
```

#### 例
```bash
# test_imagesディレクトリの画像を変換
python3 cycle_gan.py test_images

# 結果はtest_images_translatedディレクトリに保存される
```

#### 入力・出力
- **入力**: PNG画像が含まれるディレクトリ
- **出力**: 元のディレクトリ名に`_translated`を付けたディレクトリに変換結果を保存

### 2. 訓練

新しいデータセットでモデルを訓練：

```bash
python3 train.py <ドメインA画像ディレクトリ> <ドメインB画像ディレクトリ>
```

#### 例
```bash
# sunny画像とcloudy画像でモデルを訓練
python3 train.py sunny_images cloudy_images
```

## 設定

### 推論設定 (config/params.yaml)

```yaml
checkpoint_path: "./checkpoints/sunny2cloudy/latest_net_G_A.pth"  # モデルパス
output_suffix: "_translated"    # 出力ディレクトリ接尾辞
image_size: [200, 88]          # 処理画像サイズ [幅, 高さ]
input_nc: 3                    # 入力チャネル数
output_nc: 3                   # 出力チャネル数
n_blocks: 9                    # ResNetブロック数
use_dropout: false             # ドロップアウト使用
device: "cuda"                 # 使用デバイス ("cuda" or "cpu")
```

### 訓練設定 (config/train.yaml)

```yaml
epochs: 100                    # 訓練エポック数
image_size: [200, 88]         # 画像サイズ
name: "sunny2cloudy"          # モデル名
```

## 学習済みモデル

`checkpoints/sunny2cloudy/`に以下のモデルが含まれています：
- `latest_net_G_A.pth`: A→B（晴→曇）ジェネレータ
- `latest_net_G_B.pth`: B→A（曇→晴）ジェネレータ  
- `latest_net_D_A.pth`: ドメインA識別器
- `latest_net_D_B.pth`: ドメインB識別器