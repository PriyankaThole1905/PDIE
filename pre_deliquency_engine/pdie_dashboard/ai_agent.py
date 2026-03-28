"""
PDIE AI Communication Agent
Advanced LLM-powered message generation for at-risk customers

This module implements:
1. Real Groq API calls (llama-3.3-70b-versatile)
2. Sophisticated prompt engineering for personalized messages
3. Multi-tier messaging strategy (Low/Medium/High/Critical risk)
4. Channel-specific optimization (SMS/WhatsApp/Email)
5. SHAP-based context injection
6. Behavioral psychology integration
7. A/B variant generation with distinct tones
8. Batch processing for full portfolio outreach
9. MessageLog tracking with generation source

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import re
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ===== ENUMS =====

class RiskTier(Enum):
    """Customer risk classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageChannel(Enum):
    """Communication channel"""
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"


# ===== DATACLASSES =====

@dataclass
class CustomerContext:
    """Context for message generation"""
    customer_name: str
    risk_score: float
    risk_tier: RiskTier
    top_risk_factors: List[Tuple[str, float]]  # (factor_name, shap_value)
    loan_emi: float
    loan_due_date: int   # Day of month
    days_until_due: int
    monthly_income: float
    recovery_options: List[str]  # Available pathways


@dataclass
class GeneratedMessage:
    """Generated message with metadata"""
    message_text: str
    character_count: int
    tone: str
    channel: MessageChannel
    cta_options: List[str]  # Call-to-action options
    expected_response_rate: float


@dataclass
class MessageLog:
    """Audit log entry for every generated message"""
    customer_id: str
    timestamp: str
    channel: str
    risk_tier: str
    message_text: str
    character_count: int
    variant_id: int
    generated_by: str   # "groq_llm" or "template_fallback"


# ===== MAIN AGENT CLASS =====

class AICommunicationAgent:
    """
    AI-powered communication agent that generates personalized,
    empathetic messages for at-risk customers using Groq's
    llama-3.3-70b-versatile model.

    Falls back gracefully to template-based messages if the Groq
    API is unavailable, so the dashboard never crashes.

    Behavioral psychology principles applied:
    - Pre-commitment (offer options before default)
    - Choice architecture (2-3 concrete options)
    - Loss aversion framing (help before problem occurs)
    - Social proof (normalize financial stress)
    - Urgency without threat
    """

    def __init__(self, use_real_llm: bool = True, api_key: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            use_real_llm: If True, call Groq API. Falls back to templates on failure.
            api_key: Groq API key. If not supplied, reads GROQ_API_KEY from env.
        """
        self.use_real_llm = use_real_llm
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        # Character limits by channel
        self.char_limits = {
            MessageChannel.SMS: 160,
            MessageChannel.WHATSAPP: 300,
            MessageChannel.EMAIL: 1000
        }

        # In-memory log of all generated messages (supports batch_generate)
        self.message_logs: List[MessageLog] = []


    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _map_risk_score_to_tier(self, risk_score: float) -> RiskTier:
        """Map numeric risk score to risk tier."""
        if risk_score >= 80:
            return RiskTier.CRITICAL
        elif risk_score >= 70:
            return RiskTier.HIGH
        elif risk_score >= 50:
            return RiskTier.MEDIUM
        else:
            return RiskTier.LOW


    def _translate_risk_factor(self, factor_name: str, shap_value: float) -> str:
        """
        Translate technical feature name to customer-friendly language.

        Args:
            factor_name: Technical feature name
            shap_value: SHAP contribution value

        Returns:
            Human-readable explanation
        """
        translations = {
            'salary_delay_days': 'your salary came in late recently',
            'savings_drawdown_rate_4w': 'your savings have been falling',
            'upi_lending_app_txn_count_30d': 'you have been using digital lending apps',
            'emergency_fund_days': 'your emergency fund is running low',
            'utility_payment_delay_avg': 'some utility bills were paid late',
            'discretionary_spend_pct_change': 'your spending pattern has changed',
            'emi_to_income_ratio': 'your EMI is high relative to your income',
            'atm_withdrawal_spike_30d': 'cash withdrawals have increased recently',
            'bill_payment_delay_max': 'one or more bills were paid late',
            'essential_spend_ratio': 'spending has shifted toward essentials'
        }
        return translations.get(factor_name, factor_name.replace('_', ' '))


    def _get_tone_guidance(self, risk_tier: RiskTier) -> str:
        """Get tone guidance based on risk tier."""
        tones = {
            RiskTier.LOW: "Friendly and informational. Light touch.",
            RiskTier.MEDIUM: "Supportive and proactive. Caring but not urgent.",
            RiskTier.HIGH: "Empathetic and solution-focused. Show genuine concern.",
            RiskTier.CRITICAL: "Urgent but supportive. Emphasize 'we want to help BEFORE it's too late.'"
        }
        return tones.get(risk_tier, "Professional and helpful.")


    def _build_system_prompt(self, risk_tier: RiskTier, channel: MessageChannel,
                              tone_override: Optional[str] = None) -> str:
        """
        Build the system prompt for the Groq LLM.

        Args:
            risk_tier: Customer risk level
            channel: Communication channel
            tone_override: Optional tone instruction that replaces the default tone guidance
                           (used for A/B variant generation)

        Returns:
            System prompt string
        """
        char_limit = self.char_limits[channel]
        tone = tone_override if tone_override else self._get_tone_guidance(risk_tier)

        system_prompt = f"""You are a professional Relationship Manager at Barclays India.
Your task is to draft an official, empathetic, and solution-focused message to a customer regarding their upcoming loan payment.

Barclays Communication Principles:
- Tone: "Official Personal" — Professional, calm, and reassuring.
- Language: Clear, modern English. No banking jargon.
- No Threats: Never use language that sounds like collections (no mention of late fees, penalty, or default).
- Partnership: Frame the bank as a supportive partner helping the customer through a tough patch.
- Social Proof: Normalize that many customers face temporary financial shifts.

Strict Formatting Rules:
- Under {char_limit} characters STRICTLY for {channel.value.upper()}.
- Greet them with "Hi [Name],"
- State clearly that we've noticed a shift and WANT TO HELP.
- Offer exactly 3 NUMBERED options:
  1. Skip next 2 EMIs (no penalty)
  2. Reduce/Restructure your monthly payment
  3. Speak with a dedicated Relationship Manager
- Sign off consistently: "- Barclays"

Tone Directive: {tone}
"""
        return system_prompt


    def _build_user_prompt(self, context: CustomerContext, channel: MessageChannel) -> str:
        """
        Build the user prompt with customer-specific context.

        Args:
            context: Customer context
            channel: Communication channel

        Returns:
            User prompt string
        """
        # Format risk factors in plain English
        risk_factors_text = ""
        for i, (factor, shap_value) in enumerate(context.top_risk_factors[:3], 1):
            readable_factor = self._translate_risk_factor(factor, shap_value)
            risk_factors_text += f"{i}. {readable_factor}\n"

        # Format recovery options
        options_text = "\n".join(f"- {opt}" for opt in context.recovery_options)

        prompt = f"""Generate a {channel.value.upper()} message for this customer:

Customer: {context.customer_name}
Risk Score: {context.risk_score:.0f}/100 ({context.risk_tier.value.upper()})

Loan Details:
- EMI: ₹{context.loan_emi:,.0f}/month
- Due date: {context.loan_due_date}th of every month
- Next EMI: {context.days_until_due} days from now

Top Risk Signals Detected:
{risk_factors_text}
Available Recovery Options:
{options_text}

Generate the message now. Remember: max {self.char_limits[channel]} characters, empathetic tone, 2-3 numbered options."""

        return prompt


    # ------------------------------------------------------------------
    # GROQ API INTEGRATION
    # ------------------------------------------------------------------

    def _call_real_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the Groq API using llama-3.3-70b-versatile.

        Falls back to template-based message if the API call fails for
        any reason (network, quota, key missing, etc.) so the dashboard
        never crashes.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            Generated message text
        """
        if not self.api_key:
            return self._generate_template_message(user_prompt)

        try:
            from groq import Groq  # type: ignore

            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception:
            # Silent fallback — never let an API error surface to the UI
            return self._generate_template_message(user_prompt)


    def generate_message_with_groq(self, context: CustomerContext,
                                    channel: MessageChannel = MessageChannel.WHATSAPP,
                                    tone_override: Optional[str] = None,
                                    variant_id: int = 0,
                                    customer_id: Optional[str] = None) -> GeneratedMessage:
        """
        Generate a personalized message via the Groq LLM, with logging.

        Builds a rich system prompt (RBI-compliant, channel-aware, tone-aware)
        and a user prompt with customer name, EMI, risk factors (in plain English),
        and recovery options.  Falls back to templates if Groq is unavailable.

        Args:
            context: Customer context
            channel: Communication channel
            tone_override: Override the default tone (used in A/B testing)
            variant_id: 0-indexed variant number (for MessageLog)
            customer_id: Optional customer ID for audit log

        Returns:
            GeneratedMessage object
        """
        system_prompt = self._build_system_prompt(context.risk_tier, channel, tone_override)
        user_prompt = self._build_user_prompt(context, channel)

        generated_by = "template_fallback"

        if self.use_real_llm and self.api_key:
            try:
                from groq import Groq  # type: ignore

                client = Groq(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                message_text = response.choices[0].message.content
                generated_by = "groq_llm"

            except Exception:
                message_text = self._generate_template_message(user_prompt)
        else:
            message_text = self._generate_template_message(user_prompt)

        # Enforce character limit
        char_limit = self.char_limits[channel]
        if len(message_text) > char_limit:
            message_text = self._trim_message(message_text, char_limit)

        cta_options = self._extract_cta_options(message_text)
        expected_response_rate = self._estimate_response_rate(context.risk_tier, channel)

        # Append to audit log
        log_entry = MessageLog(
            customer_id=customer_id or context.customer_name.replace(" ", "_").lower(),
            timestamp=datetime.now().isoformat(),
            channel=channel.value,
            risk_tier=context.risk_tier.value,
            message_text=message_text,
            character_count=len(message_text),
            variant_id=variant_id,
            generated_by=generated_by
        )
        self.message_logs.append(log_entry)

        tone_label = tone_override if tone_override else self._get_tone_guidance(context.risk_tier)

        return GeneratedMessage(
            message_text=message_text,
            character_count=len(message_text),
            tone=tone_label,
            channel=channel,
            cta_options=cta_options,
            expected_response_rate=expected_response_rate
        )


    # ------------------------------------------------------------------
    # TEMPLATE FALLBACK
    # ------------------------------------------------------------------

    def _generate_template_message(self, context_text: str) -> str:
        """
        Generate a message using smart templates (fallback when no LLM available).

        Args:
            context_text: Context information (user prompt text)

        Returns:
            Generated message string
        """
        # Extract key information using regex
        name_match = re.search(r'Customer: (.+?)\n', context_text)
        name = name_match.group(1) if name_match else "Customer"

        risk_match = re.search(r'Risk Score: (\d+)/100 \((\w+)\)', context_text)
        risk_score = int(risk_match.group(1)) if risk_match else 70
        risk_tier = risk_match.group(2).lower() if risk_match else "high"

        emi_match = re.search(r'EMI: ₹([\d,]+)', context_text)
        emi = emi_match.group(1) if emi_match else "12,500"

        days_match = re.search(r'Next EMI: (\d+) days', context_text)
        days = days_match.group(1) if days_match else "12"

        factors_match = re.search(r'Top Risk Signals Detected:\n1\. (.+?)\n', context_text)
        top_factor = factors_match.group(1) if factors_match else "some financial stress signals"

        if risk_tier == 'critical' or risk_score >= 80:
            template = (
                f"Hi {name}, we see you're under financial pressure. "
                f"We want to help BEFORE your ₹{emi} EMI bounces in {days} days.\n"
                f"Options:\n"
                f"1. Skip next 2 EMIs (no penalty)\n"
                f"2. Reduce EMI amount\n"
                f"3. Talk to advisor NOW\n"
                f"Reply 1/2/3 - Barclays"
            )
        elif risk_tier == 'high' or risk_score >= 70:
            template = (
                f"Hi {name}, we noticed {top_factor}. "
                f"If your ₹{emi} EMI in {days} days will be tough, we can help:\n"
                f"1. Payment holiday (skip next EMI)\n"
                f"2. Reduce EMI\n"
                f"3. Talk to advisor\n"
                f"Reply 1/2/3 - Barclays"
            )
        elif risk_tier == 'medium' or risk_score >= 50:
            template = (
                f"Hi {name}, your ₹{emi} EMI is due in {days} days. "
                f"If you need help:\n"
                f"1. Payment holiday option\n"
                f"2. Restructure loan\n"
                f"Reply or call 1800-XXX-XXXX - Barclays"
            )
        else:
            template = (
                f"Hi {name}, your ₹{emi} EMI is due soon. "
                f"We're here if you need help with payments. "
                f"Call 1800-XXX-XXXX - Barclays"
            )

        return template


    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate_message(self, context: CustomerContext,
                         channel: MessageChannel = MessageChannel.SMS) -> GeneratedMessage:
        """
        Generate a personalized message for a customer.

        Routes to Groq if use_real_llm=True and api_key is available,
        otherwise uses the template fallback.

        Args:
            context: Customer context
            channel: Communication channel

        Returns:
            GeneratedMessage object
        """
        return self.generate_message_with_groq(context, channel)


    def generate_multiple_variants(self, context: CustomerContext,
                                   channel: MessageChannel = MessageChannel.SMS,
                                   n_variants: int = 3) -> List[GeneratedMessage]:
        """
        Generate multiple genuinely different message variants for A/B testing.

        Variant 0 – Empathetic: "We understand finances can be tough..."
        Variant 1 – Direct:     "Your EMI is due in X days. Here are your options:"
        Variant 2 – Friendly:   "Hey [Name], just a heads-up from Barclays..."

        Each variant sends a different system-level tone instruction to the LLM,
        so the messages are substantively different rather than superficially varied.

        Args:
            context: Customer context
            channel: Communication channel
            n_variants: Number of variants to return (max 3)

        Returns:
            List of GeneratedMessage objects
        """
        tone_overrides = [
            (
                "Empathetic and warm. Open with understanding: acknowledge that managing finances "
                "can be hard and that Barclays is there as a friend. Use gentle, supportive language."
            ),
            (
                "Direct and clear. Lead with the key fact: when the EMI is due and how much it is. "
                "Be concise and action-oriented. No fluff — just options and a clear next step."
            ),
            (
                "Friendly and casual. Start with 'Hey [Name]!' to sound approachable. "
                "Keep it conversational, positive, and encouraging — like a helpful friend at the bank."
            )
        ]

        variants: List[GeneratedMessage] = []
        for i in range(min(n_variants, len(tone_overrides))):
            variant = self.generate_message_with_groq(
                context=context,
                channel=channel,
                tone_override=tone_overrides[i],
                variant_id=i
            )
            variants.append(variant)

        return variants


    def batch_generate(self, customers: List[CustomerContext],
                       channel: MessageChannel = MessageChannel.WHATSAPP,
                       customer_ids: Optional[List[str]] = None) -> List[GeneratedMessage]:
        """
        Generate personalized messages for a full portfolio of at-risk customers.

        Args:
            customers: List of CustomerContext objects
            channel: Communication channel to use for all customers
            customer_ids: Optional list of customer IDs for audit logging
                          (must be same length as customers if provided)

        Returns:
            List of GeneratedMessage objects (one per customer, same order)
        """
        results: List[GeneratedMessage] = []

        for i, ctx in enumerate(customers):
            cid = customer_ids[i] if customer_ids and i < len(customer_ids) else None
            msg = self.generate_message_with_groq(
                context=ctx,
                channel=channel,
                variant_id=0,
                customer_id=cid
            )
            results.append(msg)

        return results


    # ------------------------------------------------------------------
    # UTILITY HELPERS
    # ------------------------------------------------------------------

    def _trim_message(self, message: str, max_chars: int) -> str:
        """Intelligently trim message to fit character limit."""
        if len(message) <= max_chars:
            return message

        lines = message.split('\n')

        if len(lines) >= 3:
            trimmed = lines[0] + '\n'
            for line in lines[1:-1]:
                if len(trimmed) + len(line) + len(lines[-1]) + 2 <= max_chars:
                    trimmed += line + '\n'
                else:
                    break
            trimmed += lines[-1]

            if len(trimmed) <= max_chars:
                return trimmed

        return message[:max_chars - 3] + "..."


    def _extract_cta_options(self, message: str) -> List[str]:
        """Extract call-to-action options from message."""
        options = []
        pattern = r'(\d+)[\.:\)]\s*(.+?)(?=\n|$)'
        matches = re.findall(pattern, message)

        for num, text in matches:
            option = text.strip()
            if option:
                options.append(option)

        return options


    def _estimate_response_rate(self, risk_tier: RiskTier, channel: MessageChannel) -> float:
        """
        Estimate expected response rate based on risk tier and channel.

        Based on industry benchmarks and behavioral studies.

        Args:
            risk_tier: Customer risk level
            channel: Communication channel

        Returns:
            Expected response rate (0-1)
        """
        base_rates = {
            MessageChannel.SMS: 0.15,
            MessageChannel.WHATSAPP: 0.25,
            MessageChannel.EMAIL: 0.08
        }

        base = base_rates.get(channel, 0.15)

        tier_multipliers = {
            RiskTier.LOW: 0.6,
            RiskTier.MEDIUM: 1.0,
            RiskTier.HIGH: 1.4,
            RiskTier.CRITICAL: 1.8
        }

        multiplier = tier_multipliers.get(risk_tier, 1.0)
        return min(base * multiplier, 0.65)


# ===== HELPER FUNCTIONS =====

def create_context_from_customer(customer_data: Dict,
                                 shap_values: Dict,
                                 recovery_pathways: List[str]) -> CustomerContext:
    """
    Create a CustomerContext from a customer data dictionary.

    Args:
        customer_data: Customer information
        shap_values: SHAP explanation values
        recovery_pathways: Available recovery options

    Returns:
        CustomerContext object
    """
    risk_score = customer_data.get('risk_score', 70.0)

    if risk_score >= 80:
        tier = RiskTier.CRITICAL
    elif risk_score >= 70:
        tier = RiskTier.HIGH
    elif risk_score >= 50:
        tier = RiskTier.MEDIUM
    else:
        tier = RiskTier.LOW

    top_factors = []
    if shap_values:
        sorted_features = sorted(shap_values.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)
        top_factors = sorted_features[:3]

    return CustomerContext(
        customer_name=customer_data.get('customer_name', 'Customer'),
        risk_score=risk_score,
        risk_tier=tier,
        top_risk_factors=top_factors,
        loan_emi=customer_data.get('emi_amount', 15000),
        loan_due_date=customer_data.get('emi_day', 15),
        days_until_due=customer_data.get('days_until_due', 12),
        monthly_income=customer_data.get('monthly_income', 75000),
        recovery_options=recovery_pathways
    )


# ===== MAIN TEST BLOCK =====

if __name__ == "__main__":
    print("=" * 70)
    print("  PDIE AI Communication Agent — Groq API Test")
    print("  Model: llama-3.3-70b-versatile")
    print("=" * 70)
    print()

    # --- Sample Indian banking customer: Priya Sharma ---
    test_customer = {
        'customer_name': 'Priya Sharma',
        'risk_score': 83,
        'emi_amount': 22000,
        'emi_day': 5,
        'days_until_due': 7,
        'monthly_income': 95000
    }

    test_shap = {
        'salary_delay_days': 0.22,
        'savings_drawdown_rate_4w': 0.17,
        'upi_lending_app_txn_count_30d': 0.13
    }

    test_options = [
        "Payment holiday — skip next 2 EMIs (no penalty, interest waived)",
        "EMI reduction — reduce to ₹16,500/month for 6 months",
        "Speak with a Barclays relationship manager right now"
    ]

    context = create_context_from_customer(test_customer, test_shap, test_options)

    # Initialise agent — reads GROQ_API_KEY from environment automatically
    agent = AICommunicationAgent(use_real_llm=True)

    if not agent.api_key:
        print("⚠️  GROQ_API_KEY not set in environment. Running template fallback mode.\n")

    # --- Single message test (WhatsApp) ---
    print("── SINGLE MESSAGE (WhatsApp) ────────────────────────────────────────")
    msg = agent.generate_message_with_groq(context, MessageChannel.WHATSAPP,
                                            customer_id="CUST_PRIYA_001")
    print(f"\n{msg.message_text}")
    print(f"\nCharacters : {msg.character_count} / {agent.char_limits[MessageChannel.WHATSAPP]}")
    print(f"Generated  : {agent.message_logs[-1].generated_by}")
    print(f"Response % : {msg.expected_response_rate * 100:.1f}%")
    print(f"CTAs       : {msg.cta_options}")

    # --- A/B variant test ---
    print()
    print("── A/B VARIANTS (SMS) ───────────────────────────────────────────────")
    variant_names = ["Empathetic", "Direct", "Friendly/Casual"]
    variants = agent.generate_multiple_variants(context, MessageChannel.SMS, n_variants=3)

    for i, (name, v) in enumerate(zip(variant_names, variants)):
        print(f"\n[Variant {i+1} — {name}]")
        print(v.message_text)
        print(f"({v.character_count} chars)")

    # --- Batch generate test ---
    print()
    print("── BATCH GENERATE (2 customers) ─────────────────────────────────────")

    second_customer = {
        'customer_name': 'Arjun Mehta',
        'risk_score': 61,
        'emi_amount': 14500,
        'emi_day': 10,
        'days_until_due': 14,
        'monthly_income': 60000
    }
    second_shap = {
        'utility_payment_delay_avg': 0.14,
        'emergency_fund_days': 0.11,
        'emi_to_income_ratio': 0.09
    }
    second_options = [
        "EMI restructuring — extend tenure by 6 months",
        "Part payment — pay 50% now, rest in 15 days",
        "Call our helpline at 1800-102-4567"
    ]
    context2 = create_context_from_customer(second_customer, second_shap, second_options)

    batch_results = agent.batch_generate(
        [context, context2],
        channel=MessageChannel.WHATSAPP,
        customer_ids=["CUST_PRIYA_001", "CUST_ARJUN_002"]
    )

    for i, result in enumerate(batch_results, 1):
        print(f"\n[Customer {i}]")
        print(result.message_text)
        print(f"({result.character_count} chars | {agent.message_logs[-(len(batch_results) - i + 1)].generated_by})")

    # --- Audit log summary ---
    print()
    print("── MESSAGE LOG SUMMARY ──────────────────────────────────────────────")
    for entry in agent.message_logs:
        print(f"  {entry.timestamp[:19]}  {entry.customer_id:<22} "
              f"ch={entry.channel:<10} tier={entry.risk_tier:<8} "
              f"v{entry.variant_id}  [{entry.generated_by}]")

    print()
    print("✅  PDIE AI Communication Agent test complete!")
