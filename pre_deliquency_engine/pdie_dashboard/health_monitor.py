"""
PDIE Financial Health Monitor
Real-time customer financial health scoring system

This module implements:
1. Composite health score (0-100) from 5 vital signs
2. Alert threshold detection
3. Trend analysis (30-day trajectory)
4. Automated intervention triggers
5. Real-time monitoring simulation

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class AlertLevel(Enum):
    """Health alert levels"""
    GREEN = "green"      # Score >= 70
    YELLOW = "yellow"    # Score 60-69
    ORANGE = "orange"    # Score 50-59
    RED = "red"          # Score 40-49
    CRITICAL = "critical"  # Score < 40


@dataclass
class VitalSign:
    """Individual vital sign measurement"""
    name: str
    score: float  # 0-100
    alert_level: AlertLevel
    current_value: any
    threshold_breached: bool
    alert_message: str


@dataclass
class HealthSnapshot:
    """Complete health assessment snapshot"""
    customer_id: str
    timestamp: datetime
    composite_score: float
    alert_level: AlertLevel
    vital_signs: List[VitalSign]
    trend_30d: List[float]  # Historical scores
    recommended_action: str
    urgency: int  # 1-5 scale


class FinancialHealthMonitor:
    """
    Monitor customer financial health using 5 vital signs:
    
    1. Income Stability (30% weight)
       - Salary delay tracking
       - Income variance
       
    2. Savings Cushion (25% weight)
       - Emergency fund days
       - Savings drawdown rate
       
    3. External Debt Signals (20% weight)
       - UPI to lending apps
       - Credit card utilization
       
    4. Bill Payment Stress (15% weight)
       - Utility payment delays
       - Essential bill coverage
       
    5. Spending Pattern Shift (10% weight)
       - Discretionary spend change
       - Essential vs discretionary ratio
    """
    
    def __init__(self):
        # Vital sign weights (must sum to 1.0)
        self.weights = {
            'income_stability': 0.30,
            'savings_cushion': 0.25,
            'external_debt': 0.20,
            'bill_payment': 0.15,
            'spending_pattern': 0.10
        }
        
        # Alert thresholds
        self.thresholds = {
            'green': 70,
            'yellow': 60,
            'orange': 50,
            'red': 40,
            'critical': 0
        }
    
    
    def _map_score_to_alert_level(self, score: float) -> AlertLevel:
        """Map numeric score to alert level."""
        if score >= 70:
            return AlertLevel.GREEN
        elif score >= 60:
            return AlertLevel.YELLOW
        elif score >= 50:
            return AlertLevel.ORANGE
        elif score >= 40:
            return AlertLevel.RED
        else:
            return AlertLevel.CRITICAL
    
    
    def assess_income_stability(self, customer_data: Dict) -> Tuple[float, VitalSign]:
        """
        Assess income stability vital sign.
        
        Factors:
        - Salary delay days (critical)
        - Salary amount variance
        - Employment type stability
        
        Args:
            customer_data: Customer information
            
        Returns:
            (score, VitalSign object)
        """
        score = 100  # Start at perfect
        
        # Get data
        salary_delay = customer_data.get('salary_delay_days', 0)
        salary_variance = customer_data.get('salary_amount_variance', 0.0)
        
        # Penalty for salary delay
        if salary_delay > 10:
            score -= 40  # Severe
            alert_msg = f"⚠️ CRITICAL: Salary delayed {salary_delay} days"
        elif salary_delay > 7:
            score -= 30  # High risk
            alert_msg = f"⚠️ WARNING: Salary delayed {salary_delay} days"
        elif salary_delay > 3:
            score -= 15  # Moderate risk
            alert_msg = f"⚠️ ALERT: Salary delayed {salary_delay} days"
        else:
            alert_msg = "✓ Salary on time"
        
        # Penalty for income variance
        if salary_variance > 0.25:
            score -= 20  # High volatility
        elif salary_variance > 0.15:
            score -= 10  # Moderate volatility
        
        score = max(0, min(100, score))
        alert_level = self._map_score_to_alert_level(score)
        
        vital = VitalSign(
            name="Income Stability",
            score=score,
            alert_level=alert_level,
            current_value=f"{salary_delay} days delay",
            threshold_breached=salary_delay > 7,
            alert_message=alert_msg
        )
        
        return score, vital
    
    
    def assess_savings_cushion(self, customer_data: Dict) -> Tuple[float, VitalSign]:
        """
        Assess savings cushion vital sign.
        
        Factors:
        - Emergency fund days remaining
        - Savings drawdown rate
        - Absolute savings level
        
        Args:
            customer_data: Customer information
            
        Returns:
            (score, VitalSign object)
        """
        score = 100
        
        # Get data
        emergency_fund_days = customer_data.get('emergency_fund_days', 30)
        savings_drawdown = customer_data.get('savings_drawdown_rate_4w', 0.0)
        current_savings = customer_data.get('current_savings', 50000)
        
        # Penalty for low emergency fund
        if emergency_fund_days < 10:
            score -= 30  # Critical
            alert_msg = f"🔴 CRITICAL: Only {emergency_fund_days:.0f} days runway"
        elif emergency_fund_days < 20:
            score -= 20  # High risk
            alert_msg = f"🟠 WARNING: Only {emergency_fund_days:.0f} days runway"
        elif emergency_fund_days < 30:
            score -= 10  # Moderate
            alert_msg = f"🟡 ALERT: {emergency_fund_days:.0f} days runway"
        else:
            alert_msg = f"✓ {emergency_fund_days:.0f} days runway"
        
        # Penalty for savings depletion
        if savings_drawdown < -0.30:  # Losing >30% per month
            score -= 25  # Severe burn rate
        elif savings_drawdown < -0.15:
            score -= 15  # High burn rate
        elif savings_drawdown < -0.05:
            score -= 5   # Moderate burn
        
        score = max(0, min(100, score))
        alert_level = self._map_score_to_alert_level(score)
        
        vital = VitalSign(
            name="Savings Cushion",
            score=score,
            alert_level=alert_level,
            current_value=f"₹{current_savings:,.0f} ({emergency_fund_days:.0f} days)",
            threshold_breached=emergency_fund_days < 15,
            alert_message=alert_msg
        )
        
        return score, vital
    
    
    def assess_external_debt(self, customer_data: Dict) -> Tuple[float, VitalSign]:
        """
        Assess external debt signals vital sign.
        
        Factors:
        - UPI transactions to lending apps
        - Credit card utilization
        - New credit inquiries
        
        Args:
            customer_data: Customer information
            
        Returns:
            (score, VitalSign object)
        """
        score = 100
        
        # Get data
        lending_app_txns = customer_data.get('upi_lending_app_txn_count_30d', 0)
        lending_app_amount = customer_data.get('upi_lending_app_amount_30d', 0)
        
        # Penalty for lending app usage
        if lending_app_txns >= 5:
            score -= 35  # Critical - taking high-cost debt
            alert_msg = f"🔴 CRITICAL: {lending_app_txns} transactions to lending apps (₹{lending_app_amount:,.0f})"
        elif lending_app_txns >= 3:
            score -= 25  # High risk
            alert_msg = f"🟠 WARNING: {lending_app_txns} transactions to lending apps"
        elif lending_app_txns >= 1:
            score -= 12  # Moderate
            alert_msg = f"🟡 ALERT: Using digital lending apps"
        else:
            alert_msg = "✓ No high-cost debt detected"
        
        score = max(0, min(100, score))
        alert_level = self._map_score_to_alert_level(score)
        
        vital = VitalSign(
            name="External Debt Signals",
            score=score,
            alert_level=alert_level,
            current_value=f"{lending_app_txns} lending app txns",
            threshold_breached=lending_app_txns >= 3,
            alert_message=alert_msg
        )
        
        return score, vital
    
    
    def assess_bill_payment_stress(self, customer_data: Dict) -> Tuple[float, VitalSign]:
        """
        Assess bill payment stress vital sign.
        
        Factors:
        - Utility payment delays
        - Rent/mortgage delays
        - Bill payment coverage ratio
        
        Args:
            customer_data: Customer information
            
        Returns:
            (score, VitalSign object)
        """
        score = 100
        
        # Get data
        utility_delay = customer_data.get('utility_payment_delay_avg', 0)
        bill_delay_max = customer_data.get('bill_payment_delay_max', 0)
        
        # Penalty for bill delays
        if utility_delay > 10:
            score -= 25  # Severe stress
            alert_msg = f"🔴 CRITICAL: Bills {utility_delay:.0f} days late on average"
        elif utility_delay > 5:
            score -= 15  # High stress
            alert_msg = f"🟠 WARNING: Bills {utility_delay:.0f} days late"
        elif utility_delay > 2:
            score -= 8   # Moderate
            alert_msg = f"🟡 ALERT: Bills {utility_delay:.0f} days late"
        else:
            alert_msg = "✓ Bills paid on time"
        
        # Additional penalty for worst-case delay
        if bill_delay_max > 15:
            score -= 10
        
        score = max(0, min(100, score))
        alert_level = self._map_score_to_alert_level(score)
        
        vital = VitalSign(
            name="Bill Payment Stress",
            score=score,
            alert_level=alert_level,
            current_value=f"{utility_delay:.0f} days avg delay",
            threshold_breached=utility_delay > 7,
            alert_message=alert_msg
        )
        
        return score, vital
    
    
    def assess_spending_pattern(self, customer_data: Dict) -> Tuple[float, VitalSign]:
        """
        Assess spending pattern shift vital sign.
        
        Factors:
        - Discretionary spend change
        - Essential vs discretionary ratio
        - ATM withdrawal spikes
        
        Args:
            customer_data: Customer information
            
        Returns:
            (score, VitalSign object)
        """
        score = 100
        
        # Get data
        discretionary_change = customer_data.get('discretionary_spend_pct_change', 0.0)
        essential_ratio = customer_data.get('essential_spend_ratio', 0.5)
        atm_spike = customer_data.get('atm_withdrawal_spike_30d', 0)
        
        # Belt-tightening (discretionary spend drop) is a stress signal
        if discretionary_change < -0.50:  # >50% drop
            score -= 15  # Severe belt-tightening
            alert_msg = f"🟡 ALERT: Discretionary spending down {abs(discretionary_change)*100:.0f}%"
        elif discretionary_change < -0.30:  # >30% drop
            score -= 10  # Moderate cutback
            alert_msg = f"🟡 Spending down {abs(discretionary_change)*100:.0f}%"
        else:
            alert_msg = "✓ Spending patterns stable"
        
        # High essential ratio indicates stress
        if essential_ratio > 0.80:  # 80%+ on essentials
            score -= 10
        
        # ATM spikes indicate cash hoarding / anxiety
        if atm_spike > 5:
            score -= 8
        
        score = max(0, min(100, score))
        alert_level = self._map_score_to_alert_level(score)
        
        vital = VitalSign(
            name="Spending Pattern Shift",
            score=score,
            alert_level=alert_level,
            current_value=f"{discretionary_change*100:+.0f}% discretionary change",
            threshold_breached=discretionary_change < -0.40,
            alert_message=alert_msg
        )
        
        return score, vital
    
    
    def compute_composite_score(self, vital_scores: Dict[str, float]) -> float:
        """
        Compute weighted composite health score.
        
        Args:
            vital_scores: Dict of vital sign scores
            
        Returns:
            Composite score (0-100)
        """
        composite = 0.0
        
        for vital_name, weight in self.weights.items():
            score = vital_scores.get(vital_name, 100)  # Default to 100 if missing
            composite += weight * score
        
        return round(composite, 1)
    
    
    def assess_customer_health(self, customer_data: Dict, 
                              historical_scores: List[float] = None) -> HealthSnapshot:
        """
        Perform complete health assessment for a customer.
        
        Args:
            customer_data: Customer information
            historical_scores: Optional 30-day history of scores
            
        Returns:
            HealthSnapshot object
        """
        # Assess all vital signs
        income_score, income_vital = self.assess_income_stability(customer_data)
        savings_score, savings_vital = self.assess_savings_cushion(customer_data)
        debt_score, debt_vital = self.assess_external_debt(customer_data)
        bills_score, bills_vital = self.assess_bill_payment_stress(customer_data)
        spending_score, spending_vital = self.assess_spending_pattern(customer_data)
        
        # Compute composite score
        vital_scores = {
            'income_stability': income_score,
            'savings_cushion': savings_score,
            'external_debt': debt_score,
            'bill_payment': bills_score,
            'spending_pattern': spending_score
        }
        
        composite = self.compute_composite_score(vital_scores)
        alert_level = self._map_score_to_alert_level(composite)
        
        # Compile vital signs
        vitals = [income_vital, savings_vital, debt_vital, bills_vital, spending_vital]
        
        # Determine recommended action
        action = self._determine_action(composite, alert_level, vitals)
        urgency = self._calculate_urgency(composite, historical_scores)
        
        # Create snapshot
        snapshot = HealthSnapshot(
            customer_id=customer_data.get('customer_id', 'UNKNOWN'),
            timestamp=datetime.now(),
            composite_score=composite,
            alert_level=alert_level,
            vital_signs=vitals,
            trend_30d=historical_scores or [composite],
            recommended_action=action,
            urgency=urgency
        )
        
        return snapshot
    
    
    def _determine_action(self, score: float, alert_level: AlertLevel, 
                         vitals: List[VitalSign]) -> str:
        """Determine recommended intervention action."""
        if score < 40:
            return "🔴 INTERVENE IMMEDIATELY - Trigger AI Communication Agent with URGENT message"
        elif score < 50:
            return "🟠 INTERVENE NOW - Trigger AI Communication Agent, offer payment holiday"
        elif score < 60:
            return "🟡 MONITOR CLOSELY - Prepare intervention, daily checks"
        elif score < 70:
            return "🟢 WATCH - Add to watch list, weekly monitoring"
        else:
            return "✅ HEALTHY - Standard monitoring"
    
    
    def _calculate_urgency(self, current_score: float, 
                          historical_scores: List[float] = None) -> int:
        """
        Calculate urgency level (1-5 scale).
        
        Factors:
        - Absolute score level
        - Rate of decline
        - Volatility
        
        Args:
            current_score: Current health score
            historical_scores: 30-day history
            
        Returns:
            Urgency level (1=low, 5=critical)
        """
        urgency = 1
        
        # Base urgency from score
        if current_score < 40:
            urgency = 5  # Critical
        elif current_score < 50:
            urgency = 4  # High
        elif current_score < 60:
            urgency = 3  # Medium
        elif current_score < 70:
            urgency = 2  # Low
        else:
            urgency = 1  # Minimal
        
        # Adjust for trend
        if historical_scores and len(historical_scores) >= 7:
            # Calculate 7-day change
            week_ago_score = historical_scores[-7]
            weekly_change = current_score - week_ago_score
            
            # Rapid deterioration increases urgency
            if weekly_change < -15:  # Lost >15 points in a week
                urgency = min(5, urgency + 2)
            elif weekly_change < -10:
                urgency = min(5, urgency + 1)
        
        return urgency
    
    
    def generate_trend_analysis(self, snapshot: HealthSnapshot) -> Dict:
        """
        Generate trend analysis from historical data.
        
        Args:
            snapshot: Health snapshot with trend data
            
        Returns:
            Trend analysis dict
        """
        if len(snapshot.trend_30d) < 2:
            return {
                'direction': 'STABLE',
                'velocity': 0.0,
                'forecast_7d': snapshot.composite_score
            }
        
        scores = snapshot.trend_30d
        
        # Linear regression for trend
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Slope (points per day)
        slope = np.polyfit(x, y, 1)[0]
        
        # Direction
        if slope < -1.0:
            direction = "↓↓ RAPIDLY DETERIORATING"
        elif slope < -0.3:
            direction = "↓ DECLINING"
        elif slope > 1.0:
            direction = "↑↑ RAPIDLY IMPROVING"
        elif slope > 0.3:
            direction = "↑ IMPROVING"
        else:
            direction = "→ STABLE"
        
        # Forecast 7 days ahead
        forecast = snapshot.composite_score + (slope * 7)
        forecast = max(0, min(100, forecast))
        
        return {
            'direction': direction,
            'velocity': slope,
            'forecast_7d': round(forecast, 1)
        }


# ===== HELPER FUNCTIONS =====

def create_mock_historical_trend(current_score: float, trend_type: str = 'declining') -> List[float]:
    """
    Create mock 30-day historical scores for demo purposes.
    
    Args:
        current_score: Current score
        trend_type: 'declining', 'improving', 'stable', 'volatile'
        
    Returns:
        List of 30 daily scores
    """
    scores = []
    
    if trend_type == 'declining':
        # Gradual decline from ~80 to current_score
        start_score = min(80, current_score + 30)
        for i in range(30):
            score = start_score - (start_score - current_score) * (i / 29)
            score += np.random.normal(0, 2)  # Add noise
            scores.append(max(0, min(100, score)))
    
    elif trend_type == 'improving':
        # Gradual improvement
        start_score = max(40, current_score - 20)
        for i in range(30):
            score = start_score + (current_score - start_score) * (i / 29)
            score += np.random.normal(0, 2)
            scores.append(max(0, min(100, score)))
    
    elif trend_type == 'volatile':
        # Random walk
        score = current_score
        for i in range(30):
            score += np.random.normal(0, 5)
            score = max(0, min(100, score))
            scores.append(score)
    
    else:  # stable
        # Stable around current score
        for i in range(30):
            score = current_score + np.random.normal(0, 3)
            scores.append(max(0, min(100, score)))
    
    return scores


if __name__ == "__main__":
    # Test the monitor
    print("Testing Financial Health Monitor...\n")
    
    # Sample customer data
    test_customer = {
        'customer_id': 'CUST00182947',
        'salary_delay_days': 10,
        'salary_amount_variance': 0.12,
        'emergency_fund_days': 12,
        'savings_drawdown_rate_4w': -0.42,
        'current_savings': 42000,
        'upi_lending_app_txn_count_30d': 5,
        'upi_lending_app_amount_30d': 28000,
        'utility_payment_delay_avg': 8,
        'bill_payment_delay_max': 15,
        'discretionary_spend_pct_change': -0.65,
        'essential_spend_ratio': 0.78,
        'atm_withdrawal_spike_30d': 6
    }
    
    # Create monitor
    monitor = FinancialHealthMonitor()
    
    # Create mock historical trend
    historical = create_mock_historical_trend(42, 'declining')
    
    # Assess health
    snapshot = monitor.assess_customer_health(test_customer, historical)
    
    # Generate trend analysis
    trend = monitor.generate_trend_analysis(snapshot)
    
    # Display results
    print("=" * 80)
    print("FINANCIAL HEALTH ASSESSMENT")
    print("=" * 80)
    print(f"\nCustomer: {snapshot.customer_id}")
    print(f"Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🏥 COMPOSITE HEALTH SCORE: {snapshot.composite_score:.1f}/100")
    print(f"Alert Level: {snapshot.alert_level.value.upper()}")
    print(f"Urgency: {snapshot.urgency}/5")
    print(f"\n📉 30-DAY TREND: {trend['direction']}")
    print(f"Velocity: {trend['velocity']:.2f} points/day")
    print(f"7-day forecast: {trend['forecast_7d']:.1f}")
    
    print(f"\n🔍 VITAL SIGNS BREAKDOWN:\n")
    
    for vital in snapshot.vital_signs:
        icon = {
            AlertLevel.GREEN: "🟢",
            AlertLevel.YELLOW: "🟡",
            AlertLevel.ORANGE: "🟠",
            AlertLevel.RED: "🔴",
            AlertLevel.CRITICAL: "🔴🔴"
        }.get(vital.alert_level, "⚪")
        
        print(f"{icon} {vital.name:<25} Score: {vital.score:>5.1f}/100")
        print(f"   {vital.alert_message}")
        print()
    
    print(f"\n⚡ RECOMMENDED ACTION:")
    print(f"   {snapshot.recommended_action}")
    
    print("\n" + "=" * 80)
    print("✅ Financial Health Monitor test complete!")
