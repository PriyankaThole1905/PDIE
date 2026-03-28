"""
PDIE Real AI Engine
Uses Groq API (LLaMA 3) for real agentic reasoning and response synthesis.
Falls back to hardcoded responses when API key is not configured.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import config
import json
from datetime import datetime
from typing import Dict, Any, Optional


# System prompt that grounds the AI as a PDIE specialist
SYSTEM_PROMPT = """You are the PDIE (Pre-Delinquency Intervention Engine) AI Specialist at Barclays Bank.
Your role is to analyze customer financial data, identify pre-delinquency risk signals, and recommend 
the best intervention strategies to prevent loan defaults.

You are empathetic, data-driven, and solution-focused. You NEVER use threatening language.
Your goal is to help customers avoid financial distress while protecting the bank's assets.

Key principles:
- Always lead with empathy — customers are people facing financial stress
- Use behavioral economics (loss aversion, social proof, choice architecture)
- Recommend the least invasive intervention first
- Back recommendations with data and risk signal analysis
- Generate concise, actionable outputs

Format your responses using markdown for readability:
- Use **bold** for key metrics and recommendations
- Use bullet points for lists
- Use emoji sparingly for visual emphasis
- Keep responses concise but comprehensive
"""


def _get_groq_client():
    """Initialize and return the Groq client."""
    try:
        import groq
        client = groq.Groq(api_key=config.GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"[PDIE AI] Failed to initialize Groq: {e}")
        return None


def generate_response(prompt: str, customer_context: dict = None) -> Dict[str, Any]:
    """
    Generate a response using a Multi-Model router (Primary: Groq, Fallback: Gemini).
    This matches the 'LLM Layer' architecture with provider fallbacks.
    """
    # ─── PROVIDER 1: GROQ (Primary) ───
    if config.is_configured('groq'):
        client = _get_groq_client()
        if client:
            try:
                full_prompt = prompt
                if customer_context:
                    context_str = json.dumps(customer_context, indent=2, default=str)
                    full_prompt = f"Customer Context:\n```json\n{context_str}\n```\n\nQuery: {prompt}"

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2048
                )
                return {
                    'success': True,
                    'live': True,
                    'provider': 'Groq (Primary)',
                    'response': response.choices[0].message.content,
                    'model': 'llama-3.3-70b-versatile',
                    'generated_at': datetime.now().isoformat(),
                }
            except Exception as e:
                print(f"[PDIE AI] Groq failure, attempting fallback: {e}")

    # ─── PROVIDER 2: GEMINI (Fallback) ───
    if config.is_configured('gemini'):
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            full_prompt = f"{SYSTEM_PROMPT}\n\n"
            if customer_context:
                context_str = json.dumps(customer_context, indent=2, default=str)
                full_prompt += f"Context: {context_str}\n\n"
            full_prompt += f"Query: {prompt}"

            response = model.generate_content(full_prompt)
            return {
                'success': True,
                'live': True,
                'provider': 'Gemini (Fallback 1)',
                'response': response.text,
                'model': 'gemini-1.5-pro',
                'generated_at': datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"[PDIE AI] Gemini failure: {e}")

    # ─── FALLBACK: MOCKED RESPONSE ───
    return {
        'success': False,
        'live': False,
        'response': '[All AI Providers Offline — using local templates]',
        'model': 'none',
        'error': 'No configured AI providers responded',
    }


def run_agentic_query(
    query: str,
    query_type: str,
    customer_data: dict,
    tool_results: dict
) -> Dict[str, Any]:
    """
    Run a full agentic query — send accumulated tool results to Groq
    for synthesis into a coherent, actionable answer.
    """
    if not config.is_configured('groq'):
        return {
            'success': False,
            'live': False,
            'response': None,  # Signal to use hardcoded fallback
        }
    
    # Build a rich prompt with all tool results
    tools_context = ""
    for tool_name, result in tool_results.items():
        result_str = json.dumps(result, indent=2, default=str)
        if len(result_str) > 1500:
            result_str = result_str[:1500] + "\n... (truncated)"
        tools_context += f"\n### Tool: {tool_name}()\n```json\n{result_str}\n```\n"
    
    query_instructions = {
        'summarize': 'Provide a concise customer risk summary with key facts, risk signals, and recommended action.',
        'script': 'Generate a complete, empathetic call script with opening, empathy hook, offer, close, and objection handlers.',
        'compare': 'Compare all recovery pathways with pros/cons and give a clear recommendation with fallback.',
        'explain_risk': 'Explain WHY this customer is flagged as high risk, citing specific behavioral signals and their impact.',
        'predict': 'Predict what happens with vs without intervention, including financial impact and ROI.',
        'full_plan': 'Create a complete autonomous intervention plan with timeline, steps, channel strategy, and expected outcomes.',
        'automate': 'Summarize all automation actions taken: message sent, reminders scheduled, calling agent status. Be specific with IDs and timestamps.',
    }
    
    instruction = query_instructions.get(query_type, 'Provide a helpful, data-driven response.')
    
    prompt = f"""You are the PDIE Agent. A user asked: "{query}"

The agentic loop has already executed the following tools and collected results:
{tools_context}

**Your task:** {instruction}

Synthesize all the tool results above into a single, coherent, actionable answer.
Use markdown formatting. Be concise but comprehensive. Cite specific numbers from the tool results.
"""
    
    return generate_response(prompt, customer_data)


def generate_draft_message_ai(customer_data: dict, risk_level: str, channel: str) -> Optional[str]:
    """
    Use Groq to generate a professional, empathetic intervention message for Barclays.
    """
    if not config.is_configured('groq'):
        return None
    
    emi = float(customer_data.get('emi_amount', 15000))
    customer_id = customer_data.get('customer_id', 'Customer')
    full_name = customer_data.get('full_name', str(customer_id).replace('CUST', '')[:4])
    
    char_limit = 160 if channel == 'SMS' else 1000
    
    prompt = f"""Generate an official, professional, and highly empathetic pre-delinquency message from Barclays.

Context:
Customer: {full_name}
Risk Level: {risk_level}
EMI Amount: ₹{emi:,.0f}
Channel: {channel} (Barclays Official Channel)
Signal: {'Salary delay' if customer_data.get('salary_delay_days', 0) > 0 else 'Recent account activity'}

Tone Guidelines:
- Barclays Official: Professional, calm, and reassuring.
- Empathetic: Acknowledge that financial stress is tough.
- Solution-Oriented: Focus on HELPING, not collecting.
- No Threatening Language: Absolutely no mention of 'legal', 'late fees', or 'defaults'.

Content Requirements:
- Greet the customer by name.
- Mention that we've noticed some changes and want to provide support.
- Offer exactly 3 numbered options:
  1. Skip upcoming 2 EMIs (no penalty)
  2. Restructure/Reduce EMI amount
  3. Speak with a dedicated Relationship Manager
- Include a clear, soft Call to Action (CTA).
- {f'STRICT LIMIT: {char_limit} characters' if channel == 'SMS' else ''}

Output ONLY the message text. No subject lines, no explanations, no quotes."""
    
    result = generate_response(prompt)
    if result.get('success') and result.get('response'):
        msg = result['response'].strip()
        # Clean up any markdown artifacts
        msg = msg.strip('`').strip('"').strip("'").strip()
        if msg.startswith('```'):
            msg = msg.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        return msg
    
    return None


def generate_call_script_ai(customer_data: dict) -> Optional[Dict[str, Any]]:
    """
    Use Groq to generate a professional, empathetic call script for Barclays Analysts.
    """
    if not config.is_configured('groq'):
        return None
    
    emi = float(customer_data.get('emi_amount', 15000))
    risk_score = float(customer_data.get('risk_score', 50))
    full_name = customer_data.get('full_name', 'our customer')
    salary_delay = customer_data.get('salary_delay_days', 0)
    
    prompt = f"""Create a professional yet deeply empathetic call script for a Barclays Relationship Manager.
Target Customer: {full_name}
Risk Score: {risk_score}/100
EMI: ₹{emi:,.0f}
Signal: {f'Salary delay of {salary_delay} days' if salary_delay > 0 else 'Shift in spending patterns'}

The goal is to provide a "safety net" call. Use behavioral psychology like 'Choice Architecture' and 'Empathy Loops'.

Structure the output as JSON with exactly these keys:
"opening": A warm greeting focusing on account wellness.
"empathy_hook": A non-intrusive acknowledgment of potential stress.
"offer": A clear explanation of the 'Payment Holiday' (skip 2 EMIs, no penalty).
"close": A soft wrap-up checking if they want this set up.
"objection_handlers": A dictionary with at least 2 common concerns (e.g., credit score impact).

Rules:
- Professional Barclays tone.
- Empathetic and curious, not accusatory.
- Focus on "Financial Wellbeing".
- Respond in RAW JSON format only."""
    
    result = generate_response(prompt)
    if result.get('success') and result.get('response'):
        resp_text = result['response'].strip()
        # Clean markdown
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()
            
        try:
            return json.loads(resp_text)
        except Exception:
            return None
            
    return None
