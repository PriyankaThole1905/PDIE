"""
PDIE Real Calling Module
Places actual outbound calls via Twilio Voice API.
Uses TwiML <Say> to speak the call script to the customer.
Falls back to simulation when Twilio is not configured.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import config
from datetime import datetime
from typing import Dict, Any


def make_call(to: str, script_text: str, voice: str = 'Polly.Aditi') -> Dict[str, Any]:
    """
    Place a real outbound voice call via Twilio.
    
    The call plays a text-to-speech message using TwiML <Say>.
    Uses Amazon Polly's 'Aditi' voice (Indian English) for natural speech.
    
    Args:
        to: Recipient phone (E.164 format, e.g. +917357138972)
        script_text: The text that will be spoken to the customer
        voice: TTS voice to use (default: Polly.Aditi for Indian English)
        
    Returns:
        Dict with call_sid, status, and metadata
    """
    if not config.is_configured('twilio'):
        return _simulated_call(to, script_text, reason='Twilio not configured')
    
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse
        
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        
        # Build TwiML response — speak the script
        twiml = VoiceResponse()
        twiml.pause(length=1)
        
        # Split script into sections for natural pauses
        sections = script_text.split('\n')
        for section in sections:
            section = section.strip()
            if section:
                twiml.say(section, voice=voice, language='en-IN')
                twiml.pause(length=1)
        
        twiml.say(
            'If you need any help, you can call us back at 1800-XXX-XXXX. '
            'Thank you for banking with Barclays. Goodbye.',
            voice=voice, language='en-IN'
        )
        
        # Place the call
        call = client.calls.create(
            twiml=str(twiml),
            from_=config.TWILIO_PHONE_NUMBER,
            to=to,
        )
        
        return {
            'success': True,
            'live': True,
            'call_sid': call.sid,
            'from': config.TWILIO_PHONE_NUMBER,
            'to': to,
            'status': call.status,  # 'queued', 'ringing', 'in-progress', 'completed'
            'initiated_at': datetime.now().isoformat(),
            'twiml_used': str(twiml)[:300] + '...',
            'voice': voice,
            'api_response': f'{call.status} — SID: {call.sid}',
        }
    except Exception as e:
        return {
            'success': False,
            'live': True,
            'to': to,
            'error': str(e),
            'initiated_at': datetime.now().isoformat(),
        }


def build_call_script_text(call_script: dict) -> str:
    """
    Convert a call_script dictionary into a natural speech text.
    
    Args:
        call_script: Dict with 'opening', 'empathy_hook', 'offer', 'close' keys
        
    Returns:
        Full script text suitable for TTS
    """
    parts = []
    
    if call_script.get('opening'):
        parts.append(call_script['opening'])
    
    if call_script.get('empathy_hook'):
        parts.append(call_script['empathy_hook'])
    
    if call_script.get('offer'):
        parts.append(call_script['offer'])
    
    if call_script.get('close'):
        parts.append(call_script['close'])
    
    return '\n'.join(parts)


def _simulated_call(to: str, script_text: str, reason: str = '') -> Dict[str, Any]:
    """Simulate a call response when Twilio is not configured."""
    import numpy as np
    return {
        'success': True,
        'live': False,
        'call_sid': f'SIM-CALL-{np.random.randint(100000, 999999)}',
        'from': '[SIMULATED]',
        'to': to,
        'status': 'SIMULATED',
        'initiated_at': datetime.now().isoformat(),
        'voice': 'Polly.Aditi',
        'api_response': f'[SIMULATED] {reason}',
    }
