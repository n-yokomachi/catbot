#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def clean_response(response):
    """モデルの回答からコンテキストやプロンプトを除去し、解答だけを抽出する関数"""
    # デバッグ用に元の応答を表示（コメントアウト）
    # print(f"生成されたテキスト: {response}")
    
    # プロンプト部分を削除して応答のみを取得
    if "AIアシスタント:" in response:
        response = response.split("AIアシスタント:")[-1].strip()
    elif "aiアシスタント:" in response:
        response = response.split("aiアシスタント:")[-1].strip()
    
    # 次の「ユーザー:」がある場合はそこまでを取得
    if "ユーザー:" in response:
        response = response.split("ユーザー:")[0].strip()
    
    # システムメッセージを除去
    response = re.sub(r'システム:.*?\n', '', response)
    
    # 「あなた:」や「Human:」などのパターンを除去
    response = re.sub(r'(あなた|Human|ヒューマン|人間):.*?[\n|$]', '', response)
    
    # 「以下は～会話です」などの定型文を除去
    response = re.sub(r'以下は.*?会話です。.*?[\n|$]', '', response)
    response = re.sub(r'AIアシスタントは親切で、丁寧で、誠実です。', '', response)
    
    # 余分な空白や改行を整理
    response = re.sub(r'\n+', '\n', response).strip()
    
    # 応答が空の場合のデフォルトメッセージ
    if not response.strip():
        response = "こんにちは！どのようにお手伝いできますか？"
    
    # 応答が長すぎる場合は最初の2文だけを取得
    sentences = re.split(r'(?<=[。！？])', response)
    if len(sentences) > 2:
        response = ''.join(sentences[:2]).strip()
    
    return response

def main():
    print("===== シンプルなチャットボット (rinna) =====")
    print("モデルをロード中: rinna/japanese-gpt2-small")
    
    try:
        # トークナイザーとモデルをロード
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
        tokenizer.do_lower_case = True  # rinnaモデル用の設定
        
        # モデルをロード
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")
        
        # GPUが利用可能な場合はGPUに移動
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"デバイス: {device}")
        
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("チャットボットの準備ができました！")
        print("「終了」と入力すると終了します。")
        
        # チャットループ
        while True:
            user_input = input("\nあなた: ")
            if user_input.lower() in ["終了", "exit", "quit"]:
                print("AIアシスタント: さようなら！またいつでも話しかけてください。")
                break
            
            try:
                # プロンプトを作成
                prompt = f"""以下は、ユーザーとAIアシスタントの会話です。
AIアシスタントは親切で、丁寧で、誠実です。

ユーザー: {user_input}
AIアシスタント:"""
                
                # 入力をトークナイズ
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                
                # 応答を生成
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=50,  # 短めの応答に調整
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3
                    )
                
                # 生成されたテキストをデコード
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 応答を整形して解答だけを抽出
                response = clean_response(generated_text)
                
                # 応答を表示
                print(f"AIアシスタント: {response}")
                
            except Exception as e:
                print(f"応答生成中にエラーが発生しました: {e}")
                print("AIアシスタント: すみません、処理中にエラーが発生しました。別の質問をお試しください。")
    
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main() 