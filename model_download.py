from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名
model_name = "rinna/japanese-gpt2-xsmall"

# 保存先ディレクトリ
save_directory = "./models/rinna-japanese-gpt2-xsmall"

# モデルとトークナイザーのダウンロードと保存
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 保存
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"モデルとトークナイザーを {save_directory} に保存しました")