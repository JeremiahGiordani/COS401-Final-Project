import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AI_SANDBOX_API_KEY")

BASE_URL = "https://api-ai-sandbox.princeton.edu/"

API_VERSION = "2024-06-01"