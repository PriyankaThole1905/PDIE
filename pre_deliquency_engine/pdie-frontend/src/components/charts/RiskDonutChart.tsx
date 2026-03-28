import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { RiskCategory } from '../../types';

interface RiskDonutChartProps {
  data: Record<RiskCategory, number>;
  total: number;
}

const COLORS: Record<RiskCategory, string> = {
  LOW: '#22c55e',
  MEDIUM: '#eab308',
  HIGH: '#f97316',
  CRITICAL: '#dc2626',
};

export function RiskDonutChart({ data, total }: RiskDonutChartProps) {
  const chartData = Object.entries(data).map(([name, value]) => ({
    name,
    value,
    color: COLORS[name as RiskCategory],
  }));

  return (
    <div className="card">
      <h3 className="text-lg font-bold text-gray-800 mb-4">Risk Tier Breakdown</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
            />
            <Legend
              formatter={(value) => (
                <span className="text-sm text-gray-600">{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="text-center mt-2">
        <p className="text-3xl font-black text-gray-800">{total.toLocaleString()}</p>
        <p className="text-sm text-gray-500">Total Customers</p>
      </div>
    </div>
  );
}

export default RiskDonutChart;
