import { useEffect, useState } from 'react';
import { Users, AlertTriangle, TrendingUp, DollarSign, Activity, Shield } from 'lucide-react';
import { KPICard } from '../components/ui/KPICard';
import { RiskDonutChart, RiskHistogram } from '../components/charts';
import { customerService } from '../api/services/customerService';
import type { CustomerSummary, PortfolioStats, RiskCategory } from '../types';

export function PortfolioOverview() {
  const [summary, setSummary] = useState<CustomerSummary | null>(null);
  const [stats, setStats] = useState<PortfolioStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [summaryData, statsData] = await Promise.all([
          customerService.getPortfolioSummary(),
          customerService.getPortfolioStats()
        ]);
        setSummary(summaryData);
        setStats(statsData);
      } catch (err) {
        console.error('Failed to fetch portfolio data:', err);
        setError('Failed to load portfolio data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatCurrency = (value: number) => {
    if (!value) return '₹0';
    if (value >= 10000000) {
      return `₹${(value / 10000000).toFixed(1)}Cr`;
    }
    if (value >= 100000) {
      return `₹${(value / 100000).toFixed(1)}L`;
    }
    return `₹${value.toLocaleString()}`;
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="page-header">
          <h1 className="text-3xl font-bold mb-2">Portfolio Command Center</h1>
          <p className="text-blue-100">Loading portfolio data...</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="kpi-card animate-pulse">
              <div className="h-8 w-8 bg-gray-200 rounded mb-2"></div>
              <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
              <div className="h-8 bg-gray-200 rounded w-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="space-y-6">
        <div className="page-header">
          <h1 className="text-3xl font-bold mb-2">Portfolio Command Center</h1>
        </div>
        <div className="card text-center py-12">
          <AlertTriangle className="w-12 h-12 text-orange-500 mx-auto mb-4" />
          <p className="text-gray-600">{error || 'No data available. Please start the API server.'}</p>
          <p className="text-gray-500 text-sm mt-2">Run: cd pdie_dashboard && python pdie_api.py</p>
        </div>
      </div>
    );
  }

  const riskDistribution = stats?.risk_distribution || { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="text-3xl font-bold mb-2">Portfolio Command Center</h1>
        <p className="text-blue-100">
          Real-time risk intelligence across {summary.total_customers.toLocaleString()}+ retail customers
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          icon={<Users className="text-blue-600" />}
          label="Total Customers"
          value={summary.total_customers.toLocaleString()}
          delta="100% portfolio"
          deltaType="neutral"
        />
        <KPICard
          icon={<AlertTriangle className="text-red-600" />}
          label="Critical Risk (≥80)"
          value={summary.critical_count.toLocaleString()}
          delta={`${(summary.critical_count / summary.total_customers * 100).toFixed(1)}% of portfolio`}
          deltaType="up"
        />
        <KPICard
          icon={<TrendingUp className="text-orange-600" />}
          label="High Risk (≥70)"
          value={summary.high_risk_count.toLocaleString()}
          delta={`${(summary.high_risk_count / summary.total_customers * 100).toFixed(1)}% of portfolio`}
          deltaType="up"
        />
        <KPICard
          icon={<Shield className="text-yellow-600" />}
          label="At Risk (≥50)"
          value={summary.at_risk_count.toLocaleString()}
          delta={`${(summary.at_risk_count / summary.total_customers * 100).toFixed(1)}% of portfolio`}
          deltaType="neutral"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          icon={<Activity className="text-purple-600" />}
          label="Avg Risk Score"
          value={summary.avg_risk_score.toFixed(1)}
          delta="Portfolio mean"
          deltaType="neutral"
        />
        <KPICard
          icon={<DollarSign className="text-green-600" />}
          label="Total Exposure (Annual)"
          value={formatCurrency(summary.total_exposure)}
          delta={`${summary.at_risk_count.toLocaleString()} at-risk accounts`}
          deltaType="neutral"
        />
        <KPICard
          icon={<Users className="text-indigo-600" />}
          label="Avg Income (High Risk)"
          value={formatCurrency(summary.avg_income_high_risk)}
          delta={`${summary.high_risk_count.toLocaleString()} customers`}
          deltaType="neutral"
        />
        <KPICard
          icon={<Shield className="text-teal-600" />}
          label="Est. Intervention Savings"
          value={formatCurrency(summary.estimated_savings)}
          delta={`If ${summary.critical_count.toLocaleString()} critical intervened`}
          deltaType="down"
        />
      </div>

      {summary.critical_count > summary.total_customers * 0.05 ? (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-r-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
            <div>
              <p className="font-semibold text-red-800">Elevated Alert</p>
              <p className="text-sm text-red-700 mt-1">
                {summary.critical_count.toLocaleString()} customers ({((summary.critical_count / summary.total_customers) * 100).toFixed(1)}%) 
                are in the CRITICAL zone (risk ≥80). These accounts require immediate intervention via the AI Communication Agent.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-green-50 border-l-4 border-green-500 rounded-r-lg p-4">
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-green-600 mt-0.5" />
            <div>
              <p className="font-semibold text-green-800">Portfolio Healthy</p>
              <p className="text-sm text-green-700 mt-1">
                Only {((summary.high_risk_count / summary.total_customers) * 100).toFixed(1)}% of customers 
                are in high-risk zones. Continue standard monitoring with weekly reviews.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RiskDonutChart 
          data={riskDistribution as Record<RiskCategory, number>} 
          total={summary.total_customers} 
        />
        <RiskHistogram data={[]} />
      </div>
    </div>
  );
}

export default PortfolioOverview;
