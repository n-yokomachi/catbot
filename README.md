# catbot

ローカルLLMの検証用に、軽量なモデルを使用して猫を模倣したチャットボットを構築するプロジェクト  

とりあえずcatbotを触ってみたい場合はこちら
https://huggingface.co/spaces/yokomachi/catbot

このリポジトリには、以下のプログラムが含まれる
- catbot.py
    - ローカルでチャットボットを実行するためのスクリプト
- direct_chatbot.py
    - ローカルで猫を模倣しないシンプルなチャットボットを実行するためのスクリプト
- finetune_cat_model(_v2).py
    - LLMをファインチューニングするためのスクリプト
    - 無印とv2の違いはトレーニングパラメータとベースモデルのみ
- model_download.py
    - Hugging Faceからモデルをダウンロードするスクリプト
    - チャットボットやファインチューニングでは直接リポジトリを参照できるのでこのスクリプトは必須ではない
- catbot/
    - streamlit製、Hugging Face Spacesで公開するデモアプリ



また、以下のリソースをHugging Faceで公開  
- ファインチューニング後のモデル
    - https://huggingface.co/yokomachi/finetuned_catbot
- ファインチューニング用のデータセット
    - https://huggingface.co/datasets/yokomachi/cat_conversations_jp
- モデルのデモアプリ
    - https://huggingface.co/spaces/yokomachi/catbot

