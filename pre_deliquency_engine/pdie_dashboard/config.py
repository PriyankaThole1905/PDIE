"""
PDIE Configuration Module
Loads API keys from .env file and provides helpers to check if services are configured.
"""

import os
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed yet, will use os.environ

# ─── Twilio Config ───
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

# ─── Groq Config ───
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# ─── Gemini Config (Fallback) ───
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# ─── Test Config ───
TEST_PHONE_NUMBER = os.getenv('TEST_PHONE_NUMBER', '')


def is_configured(service: str) -> bool:
    """Check if a service has valid credentials configured."""
    if service == 'twilio':
        return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)
    elif service == 'groq':
        return bool(GROQ_API_KEY)
    elif service == 'gemini':
        return bool(GEMINI_API_KEY)
    elif service == 'whatsapp':
        return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_NUMBER)
    return False


def get_status_summary() -> dict:
    """Get status of all configured services."""
    return {
        'twilio_sms': is_configured('twilio'),
        'twilio_whatsapp': is_configured('whatsapp'),
        'groq_ai': is_configured('groq'),
        'gemini_ai': is_configured('gemini'),
        'test_phone': bool(TEST_PHONE_NUMBER),
    }
