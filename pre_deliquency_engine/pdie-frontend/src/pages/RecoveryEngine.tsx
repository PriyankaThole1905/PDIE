import { Activity, TrendingUp, DollarSign, CheckCircle } from 'lucide-react';

export function RecoveryEngine() {
  const pathways = [
    {
      name: 'EMI Holiday',
      npv: 125000,
      recovery_rate: 0.85,
      acceptance_prob: 0.72,
      new_emi: 0,
      new_tenure_months: 30,
      monthly_savings: 18500,
      description: 'Temporary EMI relief for 3 months',
    },
    {
      name: 'Graduated EMI',
      npv: 98000,
      recovery_rate: 0.78,
      acceptance_prob: 0.65,
      new_emi: 14800,
      new_tenure_months: 36,
      monthly_savings: 3700,
      description: 'Step-up EMI starting low and increasing over time',
    },
    {
      name: 'Interest Concession',
      npv: 145000,
      recovery_rate: 0.82,
      acceptance_prob: 0.58,
      new_emi: 16800,
      new_tenure_months: 24,
      monthly_savings: 1700,
      description: 'Reduced interest rate for remainder of loan',
    },
    {
      name: 'Loan Restructuring',
      npv: 112000,
      recovery_rate: 0.75,
      acceptance_prob: 0.45,
      new_emi: 15500,
      new_tenure_months: 42,
      monthly_savings: 3000,
      description: 'Extended tenure with reduced EMI',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="text-3xl font-bold mb-2">Recovery Decision Engine</h1>
        <p className="text-blue-100">
          AI-powered pathway simulation · Monte Carlo analysis · Policy-compliant offers
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="card bg-gradient-to-br from-blue-50 to-cyan-50">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <span className="text-sm font-medium text-gray-600">Pathways Tested</span>
          </div>
          <p className="text-3xl font-black text-gray-800">5</p>
        </div>
        <div className="card bg-gradient-to-br from-green-50 to-emerald-50">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-gray-600">Best Recovery</span>
          </div>
          <p className="text-3xl font-black text-gray-800">85%</p>
        </div>
        <div className="card bg-gradient-to-br from-purple-50 to-violet-50">
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-purple-600" />
            <span className="text-sm font-medium text-gray-600">NPV Impact</span>
          </div>
          <p className="text-3xl font-black text-gray-800">₹1.45L</p>
        </div>
        <div className="card bg-gradient-to-br from-orange-50 to-amber-50">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="w-5 h-5 text-orange-600" />
            <span className="text-sm font-medium text-gray-600">Policy Checks</span>
          </div>
          <p className="text-3xl font-black text-gray-800">4/4</p>
        </div>
      </div>

      <div className="card">
        <h3 className="text-lg font-bold text-gray-800 mb-4">Recommended Recovery Pathways</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-xs font-bold text-gray-500 uppercase">Pathway</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">NPV</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">Recovery Rate</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">Acceptance Prob</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">New EMI</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">Tenure</th>
                <th className="text-right py-3 px-4 text-xs font-bold text-gray-500 uppercase">Monthly Savings</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-gray-500 uppercase">Action</th>
              </tr>
            </thead>
            <tbody>
              {pathways.map((pathway, idx) => (
                <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-4 px-4">
                    <p className="font-semibold text-gray-800">{pathway.name}</p>
                    <p className="text-sm text-gray-500">{pathway.description}</p>
                  </td>
                  <td className="py-4 px-4 text-right font-bold text-green-600">
                    ₹{pathway.npv.toLocaleString()}
                  </td>
                  <td className="py-4 px-4 text-right">
                    <span className={`font-semibold ${
                      pathway.recovery_rate >= 0.8 ? 'text-green-600' : 
                      pathway.recovery_rate >= 0.7 ? 'text-yellow-600' : 'text-orange-600'
                    }`}>
                      {pathway.recovery_rate * 100}%
                    </span>
                  </td>
                  <td className="py-4 px-4 text-right">
                    <span className={`font-semibold ${
                      pathway.acceptance_prob >= 0.6 ? 'text-green-600' : 
                      pathway.acceptance_prob >= 0.4 ? 'text-yellow-600' : 'text-orange-600'
                    }`}>
                      {pathway.acceptance_prob * 100}%
                    </span>
                  </td>
                  <td className="py-4 px-4 text-right font-medium">
                    ₹{pathway.new_emi.toLocaleString()}
                  </td>
                  <td className="py-4 px-4 text-right font-medium">
                    {pathway.new_tenure_months} months
                  </td>
                  <td className="py-4 px-4 text-right font-medium text-green-600">
                    ₹{pathway.monthly_savings.toLocaleString()}
                  </td>
                  <td className="py-4 px-4 text-center">
                    <button className="btn-primary text-sm py-2 px-4">
                      Create Offer
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default RecoveryEngine;
