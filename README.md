# catbot

ローカルLLMの検証用に、軽量なモデルを使用して猫を模倣したチャットボットを構築するプロジェクト  

とりあえずcatbotを触ってみたい場合はこちら
https://huggingface.co/spaces/yokomachi/catbot

このリポジトリには、以下のプログラムが含まれる
・catbot.py
    ローカルでチャットボットを実行するためのスクリプト
・direct_chatbot.py
    ローカルで猫を模倣しないシンプルなチャットボットを実行するためのスクリプト
・finetune_cat_model(_v2).py
    LLMをファインチューニングするためのスクリプト
    無印とv2の違いはトレーニングパラメータとベースモデルのみ
・model_download.py
    Hugging Faceからモデルをダウンロードするスクリプト
    チャットボットやファインチューニングでは直接リポジトリを参照できるのでこのスクリプトは必須ではない
・catbot/
    streamlit製、Hugging Face Spacesで公開するデモアプリ



また、以下のリソースをHugging Faceで公開
・ファインチューニング後のモデル
    https://huggingface.co/yokomachi/finetuned_catbot
・ファインチューニング用のデータセット
    https://huggingface.co/datasets/yokomachi/cat_conversations_jp
・モデルのデモアプリ
    https://huggingface.co/spaces/yokomachi/catbot

ローカルで動作する日本語の猫チャットボットです。rinna/japanese-gpt2-xsmallモデルを使用して、猫のような応答を生成します。

## 特徴

- ローカルで動作（インターネット接続不要）
- 軽量な日本語モデル（rinna/japanese-gpt2-xsmall）を使用
- 量子化モデルにも対応（メモリ使用量を削減）
- LangChainを使用した柔軟な拡張性

## セットアップ

### 1. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. モデルのダウンロード

以下のスクリプトを実行して、モデルをダウンロードします：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名
model_name = "rinna/japanese-gpt2-xsmall"
save_path = "./models/rinna-japanese-gpt2-xsmall"

# モデルとトークナイザーのダウンロード
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 保存
import os
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"モデルを {save_path} に保存しました")
```

### 3. （オプション）モデルの量子化

メモリ使用量を削減したい場合は、以下のスクリプトを実行してモデルを量子化できます：

```bash
python quantize_model.py
```

## 使い方

### 通常版（標準モデル）

```bash
python cat_chatbot.py
```

### 量子化版（メモリ使用量削減）

```bash
python cat_chatbot_quantized.py
```

## 会話例

```
===== 猫チャットボット =====
「終了」と入力すると終了します。

あなた: こんにちは

猫: にゃ〜。（毛づくろいしながらちらっとあなたを見る）

あなた: お腹すいた？

猫: にゃん！（急に目を輝かせる）ご主人様、魚くれるの？待ってたにゃ！（尻尾をピンと立てて期待している）

あなた: 終了

猫: にゃ〜（さようなら）
```

## カスタマイズ

`cat_chatbot.py`または`cat_chatbot_quantized.py`の`CAT_PERSONALITY`変数を編集することで、猫の性格や特徴をカスタマイズできます。

## 注意事項

- モデルのサイズによってはメモリ使用量が多くなる場合があります
- 生成される応答は確率的なものであり、毎回異なる結果になります
- 小さなモデルを使用しているため、応答の品質には限界があります

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

使用しているrinna/japanese-gpt2-xsmallモデルのライセンスについては、[Hugging Faceのモデルカード](https://huggingface.co/rinna/japanese-gpt2-xsmall)を参照してください。 