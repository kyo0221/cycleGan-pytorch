# cycleGAN-pytorch

天候変換（晴れ⇔曇り）を行うCycleGANのPyTorch実装です。教師なし学習により、ペア画像なしで画像から画像への変換を実現します。

## 概要

CycleGANは、対応するペア画像がない状況で、あるドメインの画像を別のドメインの画像に変換する深層学習モデルです。このプロジェクトでは、特に天候変換（晴天↔曇天）に特化した実装を提供しています。

## サンプル結果

### 🌞 晴れ → 曇り変換 (Sunny to Cloudy)

| サンプル1 | サンプル2 | サンプル3 |
|:---:|:---:|:---:|
| **変換前** | **変換前** | **変換前** |
| ![Sunny1](samples/sunny2cloudy/sunny_01.png) | ![Sunny2](samples/sunny2cloudy/sunny_02.png) | ![Sunny3](samples/sunny2cloudy/sunny_03.png) |
| **変換後** | **変換後** | **変換後** |
| ![Cloudy1](samples/sunny2cloudy/cloudy_01.png) | ![Cloudy2](samples/sunny2cloudy/cloudy_02.png) | ![Cloudy3](samples/sunny2cloudy/cloudy_03.png) |

### ☁️ 曇り → 晴れ変換 (Cloudy to Sunny)

| サンプル1 | サンプル2 | サンプル3 |
|:---:|:---:|:---:|
| **変換前** | **変換前** | **変換前** |
| ![Cloudy1](samples/cloudy2sunny/cloudy_01.png) | ![Cloudy2](samples/cloudy2sunny/cloudy_02.png) | ![Cloudy3](samples/cloudy2sunny/cloudy_03.png) |
| **変換後** | **変換後** | **変換後** |
| ![Sunny1](samples/cloudy2sunny/sunny_01.png) | ![Sunny2](samples/cloudy2sunny/sunny_02.png) | ![Sunny3](samples/cloudy2sunny/sunny_03.png) |


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
├── samples/              # サンプル画像
│   ├── sunny2cloudy/     # 晴れ→曇り変換サンプル
│   └── cloudy2sunny/     # 曇り→晴れ変換サンプル
├── checkpoints/         # 学習済みモデル
├── datasets/            # データセット
├── logs/               # 訓練ログ
├── LICENSE              # ライセンスファイル
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
注意：画像フォルダ内のファイル名は00001.png, 00002.png, ...と続くことが前提となっています

#### 入力・出力
- **入力**: PNG画像が含まれるディレクトリ
- **出力**: 元のディレクトリ名に`_translated`を付けたディレクトリに変換結果を保存

### 2. 訓練

新しいデータセットでモデルを訓練：

```bash
python3 train.py <ドメインA画像ディレクトリ> <ドメインB画像ディレクトリ>
```

注意：画像フォルダ内のファイル名は00001.png, 00002.png, ...と続くことが前提となっています
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

## ライセンス

このプロジェクトは [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) リポジトリからコードを引用・改変しており、**BSDライセンス**の下で公開されています。

### 著作権情報

以下の著作権表示とライセンス条項が適用されます：

- **この実装**: Copyright (c) 2024, your_name
- **CycleGAN**: Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
- **pix2pix**: Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
- **DCGAN**: Copyright (c) 2015, Facebook, Inc.

詳細なライセンス条項については [LICENSE](LICENSE) ファイルをご確認ください。

### 使用条件

- ソースコードの再配布時：著作権表示とライセンス条項の保持が必要
- バイナリ形式での再配布時：ドキュメントに著作権表示とライセンス条項の記載が必要
- 免責条項：本ソフトウェアは「現状のまま」提供され、一切の保証はありません

### 学術利用

学術研究での使用時は、以下の論文を引用してください：

### 引用文献
```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
