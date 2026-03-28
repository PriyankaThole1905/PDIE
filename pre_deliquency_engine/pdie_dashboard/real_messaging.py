"""
PDIE Real Messaging Module
Sends actual SMS and WhatsApp messages via Twilio API.
Falls back to simulation when Twilio is not configured.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import config
from datetime import datetime
from typing import Dict, Any


def send_sms(to: str, message: str) -> Dict[str, Any]:
    """
    Send a real SMS via Twilio.
    
    Args:
        to: Recipient phone number (E.164 format, e.g. +917357138972)
        message: Message body text
        
    Returns:
        Dict with message_sid, status, timestamps, and delivery info
    """
    if not config.is_configured('twilio'):
        return _simulated_response(to, message, 'SMS', reason='Twilio not configured')
    
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        
        twilio_msg = client.messages.create(
            body=message,
            from_=config.TWILIO_PHONE_NUMBER,
            to=to
        )
        
        return {
            'success': True,
            'live': True,
            'message_sid': twilio_msg.sid,
            'channel': 'SMS',
            'from': config.TWILIO_PHONE_NUMBER,
            'to': to,
            'status': twilio_msg.status,  # 'queued', 'sent', 'delivered'
            'sent_at': datetime.now().isoformat(),
            'character_count': len(message),
            'api_response': f'{twilio_msg.status} — SID: {twilio_msg.sid}',
        }
    except Exception as e:
        return {
            'success': False,
            'live': True,
            'channel': 'SMS',
            'to': to,
            'error': str(e),
            'sent_at': datetime.now().isoformat(),
        }


def send_whatsapp(to: str, message: str) -> Dict[str, Any]:
    """
    Send a real WhatsApp message via Twilio Sandbox.
    
    IMPORTANT: Recipient must first join the Twilio WhatsApp Sandbox
    by sending "join <sandbox-keyword>" to +14155238886 on WhatsApp.
    
    Args:
        to: Recipient phone (E.164 format, e.g. +917357138972)
        message: Message body text
        
    Returns:
        Dict with message_sid, status, timestamps
    """
    if not config.is_configured('whatsapp'):
        return _simulated_response(to, message, 'WhatsApp', reason='Twilio WhatsApp not configured')
    
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        
        # WhatsApp numbers need the whatsapp: prefix
        wa_to = f'whatsapp:{to}' if not to.startswith('whatsapp:') else to
        wa_from = config.TWILIO_WHATSAPP_NUMBER
        if not wa_from.startswith('whatsapp:'):
            wa_from = f'whatsapp:{wa_from}'
        
        twilio_msg = client.messages.create(
            body=message,
            from_=wa_from,
            to=wa_to
        )
        
        return {
            'success': True,
            'live': True,
            'message_sid': twilio_msg.sid,
            'channel': 'WhatsApp',
            'from': wa_from,
            'to': wa_to,
            'status': twilio_msg.status,
            'sent_at': datetime.now().isoformat(),
            'character_count': len(message),
            'api_response': f'{twilio_msg.status} — SID: {twilio_msg.sid}',
        }
    except Exception as e:
        return {
            'success': False,
            'live': True,
            'channel': 'WhatsApp',
            'to': to,
            'error': str(e),
            'sent_at': datetime.now().isoformat(),
        }


def send_message(to: str, message: str, channel: str = 'SMS') -> Dict[str, Any]:
    """
    Send a message via the specified channel.
    Dispatcher that routes to SMS or WhatsApp.
    
    Args:
        to: Recipient phone (E.164 format)
        message: Message body
        channel: 'SMS' or 'WhatsApp'
        
    Returns:
        Delivery result dict
    """
    if channel.lower() == 'whatsapp':
        return send_whatsapp(to, message)
    else:
        return send_sms(to, message)


def _simulated_response(to: str, message: str, channel: str, reason: str = '') -> Dict[str, Any]:
    """Generate a simulated response when Twilio is not configured."""
    import numpy as np
    return {
        'success': True,
        'live': False,
        'message_sid': f'SIM-MSG-{np.random.randint(100000, 999999)}',
        'channel': channel,
        'from': '[SIMULATED]',
        'to': to,
        'status': 'SIMULATED',
        'sent_at': datetime.now().isoformat(),
        'character_count': len(message),
        'api_response': f'[SIMULATED] {reason}',
    }
