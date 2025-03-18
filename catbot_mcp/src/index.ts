import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { HuggingFaceInference } from "@langchain/community/llms/hf";
import * as dotenv from "dotenv";
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

// 環境変数の読み込み
dotenv.config({ path: resolve(dirname(fileURLToPath(import.meta.url)), '../.env') });

// 猫の性格設定
const CAT_PERSONALITY = `
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
10. 応答は必ず「鳴き声」か「鳴き声（行動）」の形式にする`;

// 猫の応答例
const CAT_EXAMPLES = `
人間: こんにちは
猫: ﾆｬｰﾝ（尻尾を振る）

人間: おはよう
猫: ﾌﾟﾙﾙ...（伸びをする）

人間: お腹すいた？
猫: ﾆｬｰ！（足元に駆け寄る）`;

/*
 * Create server instance
 */
const server = new McpServer({
  name: "catbot",
  version: "1.0.0",
});

/*
 * HuggingFace モデルの初期化
 */
const model = new HuggingFaceInference({
  model: "yokomachi/rinnya",
  apiKey: process.env.HUGGINGFACE_API_KEY,
  temperature: 0.7,
  maxTokens: 100,
  topP: 0.9,
});

/**
 * Register message tool
 * ここにMCPサーバが提供するツールのロジックを記述
 */
server.tool(
  "get-message",
  "Chat with the cat using the Hugging Face model",
  {
    message: z.string().describe("Message to send to the cat"),
  },
  async ({ message }) => {
    try {
      // プロンプトの作成
      const prompt = `${CAT_PERSONALITY}

        以下は猫と人間の会話例です：
        ${CAT_EXAMPLES}

        人間: ${message}
        猫:`;

      // モデルを使用して応答を生成
      const response = await model.call(prompt);

      // 応答から猫の返事部分を抽出
      let catResponse = extractCatResponse(response);
      
      // 応答の後処理
      catResponse = postProcessResponse(catResponse);

      return {
        content: [
          {
            type: "text",
            text: catResponse,
          },
        ],
      };
    } catch (error) {
      console.error("Error generating response:", error);
      return {
        content: [
          {
            type: "text",
            text: "ﾆｬ？（首を傾げる）",
          },
        ],
      };
    }
  }
);

/**
 * 生成されたテキストから猫の応答部分を抽出する関数
 */
function extractCatResponse(generatedText: string): string {
  // テキストの整形
  const text = generatedText.trim();
  
  // 「猫:」の後の部分を抽出
  if (text.includes("猫:")) {
    const parts = text.split("猫:");
    return parts[parts.length - 1].trim();
  }
  
  return text;
}

/**
 * 応答の後処理を行う関数（最小限の処理のみ）
 */
function postProcessResponse(response: string): string {
  // 応答の整形（空白の削除）
  let processedResponse = response.trim();
  
  // 「人間:」が含まれる場合、それ以降を削除
  if (processedResponse.includes("人間:")) {
    processedResponse = processedResponse.split("人間:")[0].trim();
  }
  
  // 最初の改行または対話の区切りで切る
  if (processedResponse.includes("\n")) {
    processedResponse = processedResponse.split("\n")[0].trim();
  }
  
  // 応答が空の場合は、デフォルトの猫の鳴き声を返す
  if (!processedResponse) {
    return "ﾆｬｰ";
  }
  
  return processedResponse;
}

/**
 * the main function to run the server
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Catbot MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
