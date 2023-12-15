# 3dp-webcam-detection
将来的に印刷の失敗検知機能として機能する予定のPythonスクリプトです。


現在は3Dプリンターのノズルの位置をウェブカメラで検出できます。

## 概要

このプロジェクトは、3Dプリンターのウェブカメラからの映像を解析し、ノズルの正確な位置をリアルタイムで特定することを目的としています。OpenCVを利用した画像処理技術により、ノズルの動きを追跡し、プリントプロセスをモニタリングします。

## 特徴

- ノズルの位置検出
- OpenCVによる画像認識

## 必要条件

- Python 3.6 以上
- OpenCV ライブラリ
- NumPy ライブラリ

## インストール

リポジトリをクローンします：

git clone https://github.com/Cola0912/3dp-webcam-detection.git


依存関係をインストールします：

pip install -r requirements.txt


## 使い方

スクリプトを実行するには、以下のコマンドを使用します：
python3 nozzle_detection.py


## ライセンス

このプロジェクトは [MITライセンス](LICENSE) のもとで公開されています。

