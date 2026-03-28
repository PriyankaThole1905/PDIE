import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, AlertTriangle } from 'lucide-react';
import { DataTable } from '../components/ui/DataTable';
import type { Column } from '../components/ui/DataTable';
import { RiskBadge } from '../components/ui/RiskBadge';
import { customerService } from '../api/services/customerService';
import type { Customer, RiskCategory } from '../types';

export function AtRiskCustomers() {
  const navigate = useNavigate();
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTiers, setSelectedTiers] = useState<RiskCategory[]>(['CRITICAL', 'HIGH']);
  const [selectedCityTier, setSelectedCityTier] = useState<string>('All');

  const fetchCustomers = async () => {
    try {
      setLoading(true);
      const response = await customerService.getCustomers({
        search: searchTerm,
        risk_tiers: selectedTiers.length > 0 && selectedTiers.length < 4 ? selectedTiers : undefined,
        city_tier: selectedCityTier !== 'All' ? selectedCityTier : undefined,
        page,
        page_size: 10
      });
      setCustomers(response.customers);
    } catch (err) {
      console.error('Failed to fetch customers:', err);
      setError('Failed to load customers. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers();
  }, [page, selectedTiers, selectedCityTier]);

  useEffect(() => {
    const debounce = setTimeout(() => {
      setPage(1);
      fetchCustomers();
    }, 500);
    return () => clearTimeout(debounce);
  }, [searchTerm]);

  const columns: Column<Customer>[] = [
    {
      key: 'customer_id',
      header: 'ID',
      render: (customer) => (
        <span className="font-mono text-sm text-gray-500">
          {String(customer.customer_id).replace('CUST0000', 'USR-')}
        </span>
      ),
    },
    {
      key: 'full_name',
      header: 'Name',
      render: (customer) => (
        <span className="font-semibold text-gray-800">{String(customer.full_name || 'N/A')}</span>
      ),
    },
    {
      key: 'risk_category',
      header: 'Risk',
      render: (customer) => (
        <RiskBadge 
          category={String(customer.risk_category || 'MEDIUM') as RiskCategory} 
          score={Number(customer.risk_score) || 0} 
        />
      ),
    },
    {
      key: 'city',
      header: 'City',
      render: (customer) => String(customer.city || 'N/A'),
    },
    {
      key: 'city_tier',
      header: 'City Tier',
      render: (customer) => String(customer.city_tier || 'N/A'),
    },
    {
      key: 'employment_type',
      header: 'Employment',
      render: (customer) => String(customer.employment_type || 'N/A'),
    },
    {
      key: 'monthly_income',
      header: 'Income',
      render: (customer) => (
        <span className="font-medium">₹{Number(customer.monthly_income || 0).toLocaleString()}</span>
      ),
    },
    {
      key: 'emi_amount',
      header: 'EMI',
      render: (customer) => (
        <span className="font-medium">₹{Number(customer.emi_amount || 0).toLocaleString()}</span>
      ),
    },
  ];

  const tierCounts = {
    CRITICAL: customers.filter(c => String(c.risk_category) === 'CRITICAL').length,
    HIGH: customers.filter(c => String(c.risk_category) === 'HIGH').length,
    MEDIUM: customers.filter(c => String(c.risk_category) === 'MEDIUM').length,
    LOW: customers.filter(c => String(c.risk_category) === 'LOW').length,
  };

  if (error) {
    return (
      <div className="space-y-6">
        <div className="page-header">
          <h1 className="text-3xl font-bold mb-2">At-Risk Customer Intelligence</h1>
        </div>
        <div className="card text-center py-12">
          <AlertTriangle className="w-12 h-12 text-orange-500 mx-auto mb-4" />
          <p className="text-gray-600">{error}</p>
          <p className="text-gray-500 text-sm mt-2">Run: cd pdie_dashboard && python pdie_api.py</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="text-3xl font-bold mb-2">At-Risk Customer Intelligence</h1>
        <p className="text-blue-100">
          Prioritized intervention list · Customers predicted to default within 21 days
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <div className={`px-4 py-2 rounded-lg border cursor-pointer transition-colors ${
          selectedTiers.includes('CRITICAL') ? 'bg-red-100 border-red-300' : 'bg-white border-gray-200'
        }`}
             onClick={() => setSelectedTiers(prev => prev.includes('CRITICAL') ? prev.filter(t => t !== 'CRITICAL') : [...prev, 'CRITICAL'])}>
          <span className="font-bold text-red-700">🔴 {tierCounts.CRITICAL}</span>
          <span className="text-red-600 text-sm ml-2">CRITICAL</span>
        </div>
        <div className={`px-4 py-2 rounded-lg border cursor-pointer transition-colors ${
          selectedTiers.includes('HIGH') ? 'bg-orange-100 border-orange-300' : 'bg-white border-gray-200'
        }`}
             onClick={() => setSelectedTiers(prev => prev.includes('HIGH') ? prev.filter(t => t !== 'HIGH') : [...prev, 'HIGH'])}>
          <span className="font-bold text-orange-700">🟠 {tierCounts.HIGH}</span>
          <span className="text-orange-600 text-sm ml-2">HIGH RISK</span>
        </div>
        <div className={`px-4 py-2 rounded-lg border cursor-pointer transition-colors ${
          selectedTiers.includes('MEDIUM') ? 'bg-yellow-100 border-yellow-300' : 'bg-white border-gray-200'
        }`}
             onClick={() => setSelectedTiers(prev => prev.includes('MEDIUM') ? prev.filter(t => t !== 'MEDIUM') : [...prev, 'MEDIUM'])}>
          <span className="font-bold text-yellow-700">🔵 {tierCounts.MEDIUM}</span>
          <span className="text-yellow-600 text-sm ml-2">MEDIUM</span>
        </div>
        <div className={`px-4 py-2 rounded-lg border cursor-pointer transition-colors ${
          selectedTiers.includes('LOW') ? 'bg-green-100 border-green-300' : 'bg-white border-gray-200'
        }`}
             onClick={() => setSelectedTiers(prev => prev.includes('LOW') ? prev.filter(t => t !== 'LOW') : [...prev, 'LOW'])}>
          <span className="font-bold text-green-700">👥 {tierCounts.LOW}</span>
          <span className="text-green-600 text-sm ml-2">TOTAL</span>
        </div>
      </div>

      <div className="card p-4">
        <div className="flex flex-wrap gap-4 mb-4">
          <div className="relative flex-1 min-w-[250px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by name or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <select
            value={selectedCityTier}
            onChange={(e) => setSelectedCityTier(e.target.value)}
            className="px-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="All">All City Tiers</option>
            <option value="Tier 1">Tier 1</option>
            <option value="Tier 2">Tier 2</option>
            <option value="Tier 3">Tier 3</option>
          </select>
        </div>

        <DataTable
          data={customers}
          columns={columns}
          pageSize={10}
          loading={loading}
          onRowClick={(customer) => navigate(`/customer/${customer.customer_id}`)}
          emptyMessage="No customers match your filters"
        />
      </div>
    </div>
  );
}

export default AtRiskCustomers;
