import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertTriangle } from 'lucide-react';
import { RiskBadge } from '../components/ui/RiskBadge';
import { CashFlowChart } from '../components/charts/CashFlowChart';
import { customerService } from '../api/services/customerService';
import type { Customer, RiskCategory } from '../types';

export function CustomerDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const [customer, setCustomer] = useState<Customer | null>(null);
  const [cashFlow, setCashFlow] = useState<{ month: string; income: number; expenses: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!id) return;
      
      try {
        setLoading(true);
        const [customerData, txnData] = await Promise.all([
          customerService.getCustomerById(id),
          customerService.getCustomerTransactions(id)
        ]);
        setCustomer(customerData);
        setCashFlow(txnData.cash_flow || []);
      } catch (err) {
        console.error('Failed to fetch customer:', err);
        setError('Failed to load customer data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id]);

  if (loading) {
    return (
      <div className="space-y-6">
        <button 
          onClick={() => navigate('/customers')}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Customers
        </button>
        <div className="page-header">
          <h1 className="text-3xl font-bold mb-2">Loading...</h1>
        </div>
        <div className="card animate-pulse">
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error || !customer) {
    return (
      <div className="space-y-6">
        <button 
          onClick={() => navigate('/customers')}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Customers
        </button>
        <div className="card text-center py-12">
          <AlertTriangle className="w-12 h-12 text-orange-500 mx-auto mb-4" />
          <p className="text-gray-600">{error || 'Customer not found'}</p>
          <button 
            onClick={() => navigate('/customers')}
            className="mt-4 text-blue-600 hover:underline"
          >
            Back to Customers
          </button>
        </div>
      </div>
    );
  }

  const initials = customer.full_name 
    ? customer.full_name.split(' ').map(n => n[0]).join('').toUpperCase() 
    : 'NA';
  
  const creditScore = Math.round(850 - (Number(customer.risk_score) * 3.5));
  const monthlyIncome = Number(customer.monthly_income) || 85000;
  const emiAmount = Number(customer.emi_amount) || 18500;

  const cashFlowData = cashFlow.length > 0 ? cashFlow : [
    { month: 'Jan', income: monthlyIncome * 0.9, expenses: monthlyIncome * 0.65 },
    { month: 'Feb', income: monthlyIncome * 0.92, expenses: monthlyIncome * 0.7 },
    { month: 'Mar', income: monthlyIncome, expenses: monthlyIncome * 0.65 },
    { month: 'Apr', income: monthlyIncome * 0.95, expenses: monthlyIncome * 0.72 },
    { month: 'May', income: monthlyIncome, expenses: monthlyIncome * 0.68 },
    { month: 'Jun', income: monthlyIncome * 0.98, expenses: monthlyIncome * 0.75 },
  ];

  return (
    <div className="space-y-6">
      <button 
        onClick={() => navigate('/customers')}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-800"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Customers
      </button>

      <div className="page-header">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Customer Deep Analysis</h1>
            <p className="text-blue-100">
              Comprehensive 360° risk assessment · AI-powered intervention recommendations
            </p>
          </div>
          <RiskBadge 
            category={String(customer.risk_category || 'MEDIUM') as RiskCategory} 
            score={Number(customer.risk_score) || 0} 
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">Customer Profile</h3>
          
          <div className="flex items-center gap-4 mb-6">
            <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center text-white text-xl font-bold">
              {initials}
            </div>
            <div>
              <h4 className="text-xl font-bold text-gray-800">{String(customer.full_name || 'Unknown')}</h4>
              <p className="text-gray-500 text-sm font-mono">{customer.customer_id}</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">City</span>
              <span className="font-medium">{String(customer.city || 'N/A')}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">City Tier</span>
              <span className="font-medium">{String(customer.city_tier || 'N/A')}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Employment</span>
              <span className="font-medium">{String(customer.employment_type || 'N/A')}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-gray-500">Credit Score</span>
              <span className={`font-bold ${creditScore < 600 ? 'text-red-600' : 'text-green-600'}`}>
                {creditScore}
              </span>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-xs text-gray-500 uppercase">Monthly Income</p>
              <p className="text-lg font-bold text-gray-800">₹{monthlyIncome.toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-xs text-gray-500 uppercase">EMI / Month</p>
              <p className="text-lg font-bold text-gray-800">₹{emiAmount.toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-xs text-gray-500 uppercase">Outstanding Principal</p>
              <p className="text-lg font-bold text-gray-800">₹{Number(customer.outstanding_principal || 0).toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-xs text-gray-500 uppercase">EMI Ratio</p>
              <p className="text-lg font-bold text-gray-800">
                {((emiAmount / monthlyIncome) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <CashFlowChart data={cashFlowData} />

          <div className="card">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Risk Factors</h3>
            <div className="space-y-3">
              {Number(customer.risk_score) >= 80 && (
                <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                  <span className="text-red-800 font-medium">Critical risk level requires immediate intervention</span>
                </div>
              )}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase mb-1">EMI to Income</p>
                  <p className="text-xl font-bold text-gray-800">
                    {((emiAmount / monthlyIncome) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase mb-1">Loan to Income</p>
                  <p className="text-xl font-bold text-gray-800">
                    {((Number(customer.outstanding_principal) / monthlyIncome)).toFixed(1)}x
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase mb-1">Debt Service Ratio</p>
                  <p className="text-xl font-bold text-gray-800">High</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CustomerDetail;
