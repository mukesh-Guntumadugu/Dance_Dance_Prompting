"""
check_models.py
===============
Verifies that the Gemini API key in .env is working and that paid-tier models
are accessible. Run this before starting a long batch job.

Usage:
    python3 src/check_models.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not set in .env")
    sys.exit(1)

print(f"🔑 API key found: {api_key[:8]}...{api_key[-4:]}\n")

client = genai.Client(api_key=api_key)

# ── Models to test ────────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    ("gemini-2.0-flash-001", "Free-tier friendly, 1M context"),
    ("gemini-1.5-pro",       "Paid tier, 2M context — used by extract_gemini_onsets.py"),
    ("gemini-1.5-flash",     "Paid tier, faster/cheaper alternative"),
]

TEST_PROMPT = "Reply with exactly: OK"

# ── List all available models first ──────────────────────────────────────────
print("📋 Available models for your API key:")
print("─" * 60)
try:
    available = []
    for m in client.models.list():
        name = m.name  # e.g. "models/gemini-1.5-pro-001"
        short = name.replace("models/", "")
        # Only show generative models relevant to our use case
        if "gemini" in short.lower():
            print(f"   {short}")
            available.append(short)
    if not available:
        print("   (no gemini models returned)")
except Exception as e:
    print(f"   ❌ Could not list models: {e}")
    available = []

print()

# ── Test generate_content on a few known models ───────────────────────────────
print(f"{'Model':<30} {'Description':<45} {'Status'}")
print("─" * 90)

working_models = []
for model_name, description in MODELS_TO_TEST:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=TEST_PROMPT
        )
        reply = (response.text or "").strip()[:30]
        print(f"  ✅ {model_name:<27} {description:<45} → \"{reply}\"")
        working_models.append(model_name)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            status = "⚠️  Rate limited (key works, but quota hit)"
            working_models.append(model_name)  # key is valid, just rate limited
        elif "404" in err or "NOT_FOUND" in err:
            status = "❌ Model not found (invalid name)"
        elif "403" in err or "PERMISSION_DENIED" in err:
            status = "❌ Permission denied (billing not enabled?)"
        elif "401" in err or "API_KEY_INVALID" in err:
            status = "❌ Invalid API key"
        else:
            status = f"❌ {err[:60]}"
        print(f"  {model_name:<30} {description:<45} {status}")

print("─" * 90)

if working_models:
    print(f"\n✅ {len(working_models)}/{len(MODELS_TO_TEST)} models accessible.")
    print(f"   Recommended for extract_gemini_onsets.py: {working_models[-1]}")
else:
    print("\n❌ No models accessible. Check your API key and billing settings.")

# ── Qwen check ────────────────────────────────────────────────────────────────
print("\nChecking Qwen local setup...")
try:
    from src.qwen_interface import DEFAULT_MODEL_ID
    print(f"✅ Qwen interface available. Default model: {DEFAULT_MODEL_ID}")
except ImportError:
    print("❌ qwen_interface could not be imported")
