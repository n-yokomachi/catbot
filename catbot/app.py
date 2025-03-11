import streamlit as st
import torch
import nest_asyncio
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# .envファイルから環境変数を読み込む
load_dotenv()

# LangSmith関連の環境変数を設定
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# nest_asyncioを適用
nest_asyncio.apply()

# torch.classes.__path__を空のリストに設定
torch.classes.__path__ = []

# ページ設定
st.set_page_config(
    page_title="catbot",
    page_icon="🐈",
    layout="centered"
)

# 猫の特性を定義
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
10. 応答は必ず「鳴き声」か「鳴き声（行動）」の形式にする
"""

# 猫の応答例
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

@st.cache_resource
def load_langchain_model():
    """LangChainモデルをロードする関数（キャッシュ付き）"""
    # Hugging Faceからモデルをロード
    model_path = "yokomachi/rinnya"
    
    # トークナイザーとモデルをロード
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.do_lower_case = True  # rinnaモデル用の設定
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデルをロード
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # GPUが利用可能な場合はGPUに移動
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Hugging Face pipelineの作成
    # Torchのエラーを回避するために設定を修正
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # no_repeat_ngram_sizeパラメータを削除（問題の原因となる可能性があるため）
    )
    
    # LangChain HuggingFacePipelineの作成
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    # プロンプトテンプレートの作成
    template = """
{cat_personality}

以下は猫と人間の会話例です：
{cat_examples}

人間: {user_input}
猫:"""
    
    prompt = PromptTemplate(
        input_variables=["cat_personality", "cat_examples", "user_input"],
        template=template
    )
    
    # 新しいRunnableSequenceの作成
    chain = (
        {
            "cat_personality": lambda x: CAT_PERSONALITY,
            "cat_examples": lambda x: CAT_EXAMPLES,
            "user_input": RunnablePassthrough()
        } 
        | prompt 
        | llm
    )
    
    return chain, device

def extract_cat_response(generated_text):
    """生成されたテキストから猫の応答部分を抽出する関数"""
    # 「猫:」の後の部分を抽出
    if "猫:" in generated_text:
        response = generated_text.split("猫:")[-1].strip()
    else:
        response = generated_text.strip()
    
    return response

def post_process_response(response):
    """応答の後処理を行う関数（最小限の処理のみ）"""
    # 応答の整形（空白の削除のみ）
    response = response.strip()
    
    # 最初の改行または対話の区切りで切る
    if "\n" in response:
        response = response.split("\n")[0].strip()
    
    # 応答が空の場合のみデフォルトの猫の鳴き声を返す
    if not response.strip():
        return "ﾆｬｰ"
    
    return response

def generate_cat_response_with_langchain(chain, user_input):
    """LangChainを使って猫の応答を生成する関数"""
    
    # 応答を生成
    result = chain.invoke(user_input)
    
    # 結果から応答テキストを取得
    generated_text = result
    
    # 応答を抽出
    response = extract_cat_response(generated_text)
    
    # 応答を後処理
    response = post_process_response(response)
    
    return response

# アプリのタイトルと説明
st.title("🐈 catbot")
st.markdown("""
猫とじゃれあうチャットボット
""")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🐈"):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# モデルのロード（初回のみ実行され、その後はキャッシュから取得）
try:
    chain, device = load_langchain_model()
    model_loaded = True
except Exception as e:
    st.error(f"モデルのロード中にエラーが発生しました: {e}")
    model_loaded = False

# ユーザー入力
if prompt := st.chat_input("猫に話しかけてみよう"):
    # ユーザーのメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ユーザーのメッセージを履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if model_loaded:
        # 猫の応答を生成
        with st.chat_message("assistant", avatar="🐈"):
            with st.spinner("猫が考え中..."):
                try:
                    response = generate_cat_response_with_langchain(chain, prompt)
                    st.markdown(response)
                    
                    # 応答を履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = "ﾆｬ？（首を傾げる）"
                    st.markdown(error_message)
                    st.error(f"エラーが発生しました: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        with st.chat_message("assistant", avatar="🐈"):
            st.markdown("ﾆｬｰ...（モデルが読み込めませんでした）")
            st.session_state.messages.append({"role": "assistant", "content": "ﾆｬｰ...（モデルが読み込めませんでした）"})

# 会話をクリアするボタン
if st.button("会話をクリア"):
    st.session_state.messages = []
    st.rerun() 