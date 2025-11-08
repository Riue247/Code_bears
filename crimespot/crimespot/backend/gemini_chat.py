import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import google.generativeai as genai


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    config = json.load(_f)

gemini_api_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Missing Gemini API key. Set config['gemini_api_key'] or GEMINI_API_KEY.")

genai.configure(api_key=gemini_api_key)
chat_model_name = config.get("chat_model", config.get("gemini_model", "models/gemini-2.5-flash"))
generation_config = {
    "temperature": config.get("gemini_temperature", 0.2),
    "top_p": config.get("gemini_top_p", 0.9),
    "top_k": config.get("gemini_top_k", 40),
}
chat_model = genai.GenerativeModel(chat_model_name, generation_config=generation_config)


def run_chat(message: str) -> str:
    response = chat_model.generate_content(message, stream=False)
    if getattr(response, "text", None):
        return response.text.strip()
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
            parts.append(getattr(part, "text", "") or "")
    return "".join(parts).strip()


def main():
    parser = argparse.ArgumentParser(description="Lightweight Gemini chat bridge.")
    parser.add_argument("--message", required=True, help="User message to send to Gemini.")
    args = parser.parse_args()

    try:
        text = run_chat(args.message)
        print(json.dumps({"response": text}, ensure_ascii=False))
    except Exception:
        print("Python error:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
