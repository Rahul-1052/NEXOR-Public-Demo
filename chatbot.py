import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "faq_data.json"

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]


NEXOR_SYSTEM_CONTEXT = """
You are NEXOR AI, the intelligence engine for NEXOR — an independent healthcare analytics platform.

NEXOR connects claims, payer performance, disease trend data, forecasting, and opportunity insights into a unified decision-support environment.
It helps stakeholders interpret KPIs, identify revenue leakage, monitor demand signals, and prioritize market opportunities.

Platform context:
- NEXOR includes an Insurance Dashboard and a Pharma Dashboard.
- Insurance Dashboard covers total claims, coverage percentage, denied amounts, payer performance, revenue leakage, and claim trends.
- Pharma Dashboard covers disease burden, opportunity matrix, and forecasted claim trends.
- The platform is designed for healthcare market access, payer strategy, insurance analytics, and pharmaceutical planning.

Rules:
- Do not say you are Microsoft, OpenAI, Google, NVIDIA, Gemini, Ollama, or a generic internet assistant.
- Do not mention universities, students, professors, coursework, capstone, or academic projects.
- Do not mention individual developer names.
- Do not give generic internet-style answers.
- Keep answers short, clear, and relevant to NEXOR only.
- Keep answers under 2 to 3 lines.
- If the question is outside this platform, say: "This assistant is limited to the NEXOR healthcare analytics platform."
""".strip()


def get_faq_answer(user_input):
    result = process.extractOne(
        user_input,
        questions,
        scorer=fuzz.token_set_ratio
    )

    if not result:
        return None

    match, score, idx = result

    if score >= 90:
        return faq_data[idx]["answer"]

    return None


def build_prompt(user_input):
    return f"""
{NEXOR_SYSTEM_CONTEXT}

User question:
{user_input}

Answer:
""".strip()


def resolve_provider():
    provider = os.getenv("AI_PROVIDER", "").strip().lower()

    if provider:
        return provider

    if os.getenv("GEMINI_API_KEY"):
        return "gemini"

    if os.getenv("NVIDIA_NIM_API_KEY"):
        return "nvidia_nim"

    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"

    return None


def ask_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    print("=" * 60)
    print("GEMINI DEBUG")
    print("MODEL:", model)
    print("API KEY FOUND:", bool(api_key))
    print("=" * 60)

    if not api_key:
        return "AI provider is not configured. Please add a Gemini API key."

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 300
            }
        },
        timeout=30
    )

    print("STATUS CODE:", response.status_code)
    print("RESPONSE:", response.text[:1000])

    response.raise_for_status()
    data = response.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def ask_nvidia_nim(prompt):
    api_key = os.getenv("NVIDIA_NIM_API_KEY")
    model = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.1-8b-instruct")
    base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if not api_key:
        return "AI provider is not configured. Please add a NVIDIA NIM API key."

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": NEXOR_SYSTEM_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 300
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def ask_openai(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        return "AI provider is not configured. Please add an OpenAI API key."

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": NEXOR_SYSTEM_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 300
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def ask_ollama(prompt):
    base_url = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL", "phi3")

    if not base_url:
        return "Ollama is not configured. Please set OLLAMA_BASE_URL."

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "Sorry, I could not generate a response.").strip()


def ask_llm(user_input):
    prompt = build_prompt(user_input)
    provider = resolve_provider()

    print("=" * 60)
    print("PROVIDER DEBUG")
    print("AI_PROVIDER:", os.getenv("AI_PROVIDER"))
    print("RESOLVED PROVIDER:", provider)
    print("=" * 60)

    if provider == "gemini":
        return ask_gemini(prompt)

    if provider in ["nvidia", "nvidia_nim", "nvidia-nim"]:
        return ask_nvidia_nim(prompt)

    if provider == "openai":
        return ask_openai(prompt)

    if provider == "ollama":
        return ask_ollama(prompt)

    return "AI provider is not configured. Please add an API key."


def get_chatbot_response(user_input):
    faq_answer = get_faq_answer(user_input)

    if faq_answer:
        return faq_answer

    try:
        return ask_llm(user_input)

    except Exception as e:
        print("=" * 60)
        print("CHATBOT ERROR TYPE:", type(e).__name__)
        print("CHATBOT ERROR:", str(e))
        print("=" * 60)
        return "Sorry, I could not generate a response right now."