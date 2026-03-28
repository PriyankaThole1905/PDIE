import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface CashFlowChartProps {
  data: { month: string; income: number; expenses: number }[];
}

export function CashFlowChart({ data }: CashFlowChartProps) {
  return (
    <div className="card">
      <h3 className="text-lg font-bold text-gray-800 mb-4">Cash Flow (Last 6 Months)</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="incomeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="expenseGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#dc2626" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#dc2626" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
            <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#6b7280' }} axisLine={{ stroke: '#e5e7eb' }} />
            <YAxis 
              tick={{ fontSize: 11, fill: '#6b7280' }} 
              axisLine={false}
              tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
              formatter={(value) => [`₹${Number(value).toLocaleString()}`, '']}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="income"
              stroke="#22c55e"
              strokeWidth={2}
              fill="url(#incomeGradient)"
              name="Income/Credit"
            />
            <Area
              type="monotone"
              dataKey="expenses"
              stroke="#dc2626"
              strokeWidth={2}
              fill="url(#expenseGradient)"
              name="Expenses/Debit"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default CashFlowChart;
