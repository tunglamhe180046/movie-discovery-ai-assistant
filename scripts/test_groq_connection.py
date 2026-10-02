import os
import json
import urllib.request
from pathlib import Path

def load_env(env_path):
    config = {}
    if not env_path.exists():
        return config
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                config[k] = v
    return config

def main():
    root_dir = Path(__file__).resolve().parent.parent
    env = load_env(root_dir / ".env")
    api_key = env.get("GROQ_API_KEY")
    base_url = env.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = env.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    print(f"Testing Groq API connection...")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if api_key else ''}")

    if not api_key:
        print("[FAIL] Missing GROQ_API_KEY in .env")
        return False

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Respond in 5 words: Confirm connection to Groq API."}
        ],
        "temperature": 0.2,
        "max_tokens": 50
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            print(f"[SUCCESS] Groq API response: {content}")
            return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[FAIL] HTTP {e.code}: {err_body}")
        # Try fallback model if model was invalid
        if "model_not_found" in err_body or "decommissioned" in err_body:
            print("Attempting with fallback model 'llama-3.1-8b-instant'...")
            payload["model"] = "llama-3.1-8b-instant"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                print(f"[SUCCESS] Fallback Groq API response: {content}")
                return True
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
