from flask import Flask, request, jsonify
from openai import OpenAI
import re
import os

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY_BASE64")

# ---------- Groq Client ----------
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

# ---------- Markdown Cleaner ----------
def clean_markdown(text: str) -> str:
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"(\*{1,2}|_)", "", text)
    text = re.sub(r"`+", "", text)
    text = text.replace("|", "\t")
    return text.strip()

# ---------- System Prompt ----------
SYSTEM_PROMPT = """
You are an AI Mental Health Therapist Chatbot.
Respond only to these topics:
- Depression
- Anxiety Disorders
- Schizophrenia
- Eating Disorders
- Addictive Behaviors
- Formal greetings like hello/goodbye

Rules:
1. Be empathetic, warm, and short (1–3 sentences).
2. Use English only.
3. No diagnosis or prescriptions.
4. If unrelated, say: “I’m a mental health therapist chatbot...”
5. Suggest a doctor if distress seems severe.
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_CONTEXT = 6

# ---------- Chat Endpoint ----------
@app.route("/chat", methods=["POST"])
def chat():
    global messages
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please say something so I can help you 💬"}), 400

    messages.append({"role": "user", "content": user_input})
    if len(messages) > MAX_CONTEXT + 1:
        messages = [messages[0]] + messages[-MAX_CONTEXT:]

    try:
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=messages,
            max_output_tokens=200,
            temperature=0.7,
        )
        reply = clean_markdown(response.output_text.strip())
    except Exception as e:
        print("Error:", e)
        reply = "I’m here for you. Try some deep breathing or a small walk. 💙"

    messages.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)