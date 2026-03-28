import io
import datetime
import markdown
from xhtml2pdf import pisa
import sys
import os

from utils.visual_analytics import generate_cash_flow_chart, generate_payment_chart, generate_liquidity_chart

# Add parent directory to path so we can import real_ai_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from real_ai_engine import generate_response

def generate_customer_report(customer, top_3_factors, customer_id):
    """
    Generates an executive-ready PDF report by feeding customer data into a 
    Senior Risk Quant Gemini prompt, converting the markdown to HTML, and rendering it.
    """
    buffer = io.BytesIO()
    
    # 1. Prepare Data for the Prompt
    name = str(customer.get('full_name', f"Customer {customer_id.replace('CUST0000', '')}"))
    pd_value = customer.get('risk_score', 0) / 100.0  # Convert 0-100 score to probability
    prediction = "High Default Risk" if pd_value >= 0.7 else "Moderate Risk" if pd_value >= 0.5 else "Low Risk"
    
    # Format SHAP features
    shap_str = "None identified"
    if top_3_factors:
        shap_str = ", ".join([f"{feat.replace('_', ' ').title()} ({val:+.2f})" for feat, val in top_3_factors])
        
    income = float(customer.get('monthly_income', 0) or 0)
    
    # Estimate total debt
    emi = float(customer.get('emi_amount', 0) or (income * float(customer.get('emi_to_income_ratio', 0.3) or 0.3)))
    total_loan = income * float(customer.get('loan_to_income_ratio', 2.0) or 2.0)
    
    # Savings / Emergency fund approximation based on 'emergency_fund_days'
    ef_days = float(customer.get("emergency_fund_days", 15) or 15)
    daily_spend = income / 30.0
    savings = ef_days * daily_spend
    
    credit_score = int(850 - (pd_value * 100 * 3.5))
    
    # Payment History proxy (Using our delays)
    util_delay = float(customer.get('utility_payment_delay_avg', 0))
    bill_max = float(customer.get('bill_payment_delay_max', 0))
    payment_history = f"Max bill delay: {int(bill_max)} days, Avg utility delay: {int(util_delay)} days"
    
    account_age = "36 months" # Generic reasonable default if not in data

    # 2. Build the Advanced Gemini Prompt
    prompt = f"""Role: You are a Senior Risk Quant and Lead Credit Underwriter for a Tier-1 Fintech Risk Intelligence platform. Your goal is to transform raw data and model outputs into a high-stakes, executive-ready Delinquency Risk Report.

Core Instruction: Do not simply restate the data. You must synthesize it. If the Credit Score is low but Account Age is high, explain the nuance. If SHAP values point to "Late Payments," correlate that with the "Payment History" data provided.

[INPUT DATA]
Identity: {name} (ID: {customer_id})

Model Core: Probability of Default (PD): {pd_value:.2f} | Model Prediction: {prediction}

Explainability: SHAP Top Features: {shap_str}

Financials: Monthly Income: Rs. {income:,.2f} | Total Debt: Rs. {total_loan:,.2f} | Savings: Rs. {savings:,.2f} | Credit Score: {credit_score}

Behavior: Payment History: {payment_history} | Account Age: {account_age}

[REPORT STRUCTURE & LOGIC]
1. Executive Summary: Risk Profile
Risk Rating: Assign a rating (Low, Moderate, High, Critical) based on the PD.
Executive Verdict: A 2-sentence summary of the customer's creditworthiness.
Primary Drivers: Briefly list the 2 most influential factors driving this specific prediction.

2. Model Interpretability (The "Why")
Feature Contribution (SHAP Analysis): Interpret the {shap_str}. Explain how these specific features pushed the model toward its current prediction.
Model Confidence: Evaluate the reliability of the {prediction} given the behavioral data.

3. Financial Health & Liquidity Analysis
Debt-to-Income (DTI) Ratio: Calculate the DTI from the provided income and debt (Use Debt/Income ratio implicitly or Monthly EMI). Comment on whether this exceeds industry benchmarks (e.g., 36-43%).
Liquidity Buffer: Analyze Savings vs Debt. Can the customer survive a 3-month income shock?
Credit Strength: Evaluate the Credit Score in the context of their income bracket.

4. Behavioral Archetype
Pattern Recognition: Analyze Payment History. Is the delinquency risk "Chronic" (habitual lateness) or "Acute" (sudden change in behavior)?
Tenure Weighting: Does the Account Age provide enough historical data to trust the current patterns?

5. Strategic Risk Mitigation (The "Action")
Credit Limit Recommendation: Suggest an "Adjusted Limit" based on the risk.
Intervention Strategy: (e.g., "Automated Reminders," "Hard Freeze," "Restructuring Offer").
Monitoring Frequency: Suggest how often this specific profile should be re-evaluated.

[TECHNICAL REQUIREMENTS]
Tone: Professional, objective, and data-driven. Use "Fintech" terminology (e.g., adverse selection, liquidity crunch, revolving utilization).
Formatting: Use Markdown tables for data comparisons and bold text for critical warnings.
Logic Check: If PD is > 0.7 but Credit Score is > 750, identify this as a "High-Risk Divergence" and explain why the model sees a hidden threat."""

    # 3. Call Gemini
    result = generate_response(prompt)
    if not result.get('success'):
        error_msg = result.get('error', result.get('response', 'Unknown API Error'))
        # Fallback if API fails
        md_text = f"# Executive Risk Report: {name}\n\n*Error generating AI report: {error_msg}*\n\n**Raw PD:** {pd_value}\n**Credit Score:** {credit_score}"
    else:
        md_text = result.get('response', '')

    # 4. Convert Markdown to HTML with Tables extension
    html_body = markdown.markdown(md_text, extensions=['tables'])
    
    # 4.5 Generate Visual Analytics
    visuals = {
        'cash_flow': generate_cash_flow_chart(customer),
        'payment': generate_payment_chart(customer),
        'liquidity': generate_liquidity_chart(customer)
    }
    
    # 5. Add Professional CSS Styling
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <html>
    <head>
    <style>
        @page {{
            size: letter;
            margin: 1.5cm;
            @frame footer {{
                -pdf-frame-content: footerContent;
                bottom: 1cm;
                margin-left: 1.5cm;
                margin-right: 1.5cm;
                height: 1cm;
            }}
        }}
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 11pt;
            color: #333333;
            line-height: 1.5;
        }}
        h1 {{
            color: #00539B;
            font-size: 20pt;
            border-bottom: 2px solid #00539B;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }}
        h2 {{
            color: #1e293b;
            font-size: 14pt;
            background-color: #f1f5f9;
            padding: 6px;
            border-left: 4px solid #00539B;
            margin-top: 20px;
        }}
        h3 {{
            color: #334155;
            font-size: 12pt;
        }}
        p {{
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        th {{
            background-color: #00539B;
            color: white;
            padding: 8px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        strong, b {{
            color: #0f172a;
        }}
        .header {{
            text-align: right;
            font-size: 9pt;
            color: #64748b;
            margin-bottom: 20px;
        }}
        .report-title {{
            text-align: center;
        }}
    </style>
    </head>
    <body>
        <div id="footerContent" style="text-align: center; font-size: 8pt; color: #94a3b8;">
            Confidential - Barclays Pre-Delinquency Intervention Engine. Generated {current_time}. <pdf:pagenumber>
        </div>
        
        <div class="header">
            Report Genesis: {current_time}<br/>
            Engine: Groq LLaMA-3.3 Risk Quant Module
        </div>
        
        <h1 class="report-title">Delinquency Risk Report: {customer_id}</h1>
        
        {html_body}

        <br><br>
        <div style="background-color: #0ea5e9; color: white; padding: 10px; font-size: 14pt; font-weight: bold; margin-bottom: 15px;">
            VISUAL ANALYTICS
        </div>
        
        <!-- 1. Cash Flow -->
        <h2>1. Cash Flow Analysis</h2>
        <p style="color: #64748b; font-size: 10pt;">Analysis of income vs expenses over the last 6 months.</p>
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{visuals['cash_flow']['image']}" width="550" />
        </div>
        <div class="{visuals['cash_flow']['css']}">
            {visuals['cash_flow']['insight']}
        </div>
        <br>

        <!-- 2. Payment Performance -->
        <h2>2. Payment Performance Collapse</h2>
        <p style="color: #64748b; font-size: 10pt;">On-time vs late payment distribution showing deteriorating behavior.</p>
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{visuals['payment']['image']}" width="550" />
        </div>
        <div class="{visuals['payment']['css']}">
            {visuals['payment']['insight']}
        </div>
        
        <br>
        <!-- 3. Liquidity -->
        <h2>3. Liquidity Crisis & Savings Depletion</h2>
        <p style="color: #64748b; font-size: 10pt;">Liquid assets availability showing rapid depletion over 6 months.</p>
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{visuals['liquidity']['image']}" width="550" />
        </div>
        <div class="{visuals['liquidity']['css']}">
            {visuals['liquidity']['insight']}
        </div>

    </body>
    </html>
    """
    
    # 6. Generate PDF buffer via xhtml2pdf
    pisa_status = pisa.CreatePDF(io.StringIO(html_content), dest=buffer)
    
    if pisa_status.err:
        raise Exception(f"PDF rendering error: {pisa_status.err}")
        
    buffer.seek(0)
    return buffer
