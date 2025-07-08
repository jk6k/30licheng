import os
from dotenv import load_dotenv

# 載入 .env 文件中的環境變數
load_dotenv()

# 現在您就可以在程式碼中透過 os.getenv() 來取得這些變數了
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
qwen_model_name = os.getenv("QWEN_MODEL_NAME")

print(f"成功讀取到 DashScope API Key 的開頭: {dashscope_api_key[:6]}...")
print(f"將使用的 Qwen 模型: {qwen_model_name}")

# 接下來就可以初始化 LangChain 的相關組件了
# from langchain_community.chat_models import ChatTongyi
#
# chat = ChatTongyi(
#     model_name=qwen_model_name,
#     dashscope_api_key=dashscope_api_key
# )
# ... 您的 LangChain 應用邏輯 ...