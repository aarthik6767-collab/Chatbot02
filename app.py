import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder="templates")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"
CONFIG_PATH = BASE_DIR / "chatbot_config"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in .env")

client = genai.Client(api_key=API_KEY)
SYSTEM_PROMPT = CONFIG_PATH.read_text(encoding="utf-8")

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.5,
                max_output_tokens=800,
            ),
        )
        return jsonify({"reply": response.text or "I couldn't generate a response."})
    except Exception as exc:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "Unable to process your request right now."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
