
from src.llm.client import LLMClient
from dotenv import load_dotenv

load_dotenv()
print(LLMClient().chat([{"role":"user","content":"Reply with: pong"}]))