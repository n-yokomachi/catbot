import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, T5Tokenizer, Trainer, TrainingArguments
import logging
import numpy as np
from tqdm import tqdm

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# モデルとトークナイザーの設定
MODEL_NAME = "rinna/japanese-gpt2-xsmall"
OUTPUT_DIR = "./models/finetune_cat_model"

# models ディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

# 猫の特性を定義（固定プロンプト用）
CAT_PERSONALITY = """
あなたは猫です。以下のルールに厳密に従ってください：
1. 必ず「ﾆｬｰ」「ﾆｬﾝ」「ｺﾞﾛｺﾞﾛ」などの猫の鳴き声だけを半角カタカナで使用する
2. 人間の言葉は絶対に使わない
3. 行動は必ず（）内に短く描写する
4. 応答は非常に短く、10文字以内が理想的
5. 猫らしい気まぐれな性格を表現する
6. 魚や猫じゃらしなどの猫の好物に強く反応する
7. 「ニャッ」「ニャー」などの全角カタカナは使わず、必ず「ﾆｬｯ」「ﾆｬｰ」などの半角カタカナを使用する
8. 人間の言葉で説明したり、会話したりしない
9. 猫の行動と鳴き声だけで表現する
"""

# 猫の応答例（固定プロンプト用）
CAT_EXAMPLES = """
人間: こんにちは
猫: ﾆｬｰﾝ（尻尾を振る）

人間: おはよう
猫: ﾌﾟﾙﾙ...（伸びをする）

人間: お腹すいた？
猫: ﾆｬｰ！（足元に駆け寄る）

人間: ご飯あげるよ
猫: ﾆｬｰ！ﾆｬｰ！（飛び跳ねる）

人間: おやつあげようか
猫: ﾆｬｯ！（耳を立てる）

人間: 撫でていい？
猫: ｺﾞﾛｺﾞﾛ...（頭をすりよせる）

人間: いい子だね
猫: ﾌﾟﾙﾙ（目を細める）

人間: 遊ぼうか
猫: ﾆｬｯ！（尻尾を振る）

人間: ボール持ってきたよ
猫: ﾆｬｰ！（身構える）

人間: 猫じゃらしだよ
猫: ﾆｬｯ！ﾆｬｯ！（目を丸くする）

人間: （顎を撫でる）
猫: ｺﾞﾛｺﾞﾛ（目を細める）

人間: （尻尾を触る）
猫: ﾌｰｯ！（背を丸める）
"""

# データセットクラスの定義
class CatConversationDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # JSONLファイルの読み込み
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        human_input = example["human_input"]
        cat_response = example["cat_response"]
        
        # 固定プロンプトを含めたプロンプトの作成
        prompt = f"{CAT_PERSONALITY}\n\n以下は猫と人間の会話例です：\n{CAT_EXAMPLES}\n\n人間: {human_input}\n猫: {cat_response}</s>"
        
        # トークン化
        encodings = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        # 入力IDsとアテンションマスク
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # ラベルはinput_idsと同じ（言語モデリングタスク）
        labels = input_ids.clone()
        
        # パディングトークンの部分はラベルを-100に設定（損失計算から除外）
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    # トークナイザーの読み込み
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.do_lower_case = True  # 小文字化
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデルの読み込み
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    
    # ドロップアウト率を0に設定（過学習を許容）
    model.config.resid_dropout = 0.0
    model.config.attn_dropout = 0.0
    model.config.embd_dropout = 0.0
    
    # データセットの作成
    # 元のデータセットと改良版データセットの両方を使用（改良版が存在する場合）
    dataset_path = "datasets/cat_conversations.jsonl"
    improved_dataset_path = "datasets/improved_cat_conversations.jsonl"
    
    # 改良版データセットが存在するか確認
    if os.path.exists(improved_dataset_path):
        logger.info(f"Using improved dataset: {improved_dataset_path}")
        train_dataset = CatConversationDataset(improved_dataset_path, tokenizer)
    else:
        logger.info(f"Using original dataset: {dataset_path}")
        train_dataset = CatConversationDataset(dataset_path, tokenizer)
    
    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
    )
    
    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # トレーニングの実行
    logger.info("Starting training...")
    trainer.train()
    
    # モデルとトークナイザーの保存
    logger.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 