import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface RiskHistogramProps {
  data: number[];
}

export function RiskHistogram({ data }: RiskHistogramProps) {
  const histogramData = Array.from({ length: 20 }, (_, i) => {
    const rangeStart = i * 5;
    const rangeEnd = (i + 1) * 5;
    const count = data.filter((score) => score >= rangeStart && score < rangeEnd).length;
    return {
      range: `${rangeStart}-${rangeEnd}`,
      count,
      midPoint: rangeStart + 2.5,
    };
  });

  return (
    <div className="card">
      <h3 className="text-lg font-bold text-gray-800 mb-4">Risk Score Distribution</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
            <XAxis 
              dataKey="range" 
              tick={{ fontSize: 10, fill: '#6b7280' }}
              axisLine={{ stroke: '#e5e7eb' }}
            />
            <YAxis 
              tick={{ fontSize: 10, fill: '#6b7280' }}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
              labelStyle={{ fontWeight: 600 }}
            />
            <ReferenceLine x="50" stroke="#eab308" strokeDasharray="3 3" />
            <ReferenceLine x="70" stroke="#f97316" strokeDasharray="3 3" />
            <ReferenceLine x="80" stroke="#dc2626" strokeDasharray="3 3" />
            <Bar dataKey="count" fill="#00395D" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-center gap-4 mt-4 text-xs">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-yellow-500 dashed"></span>
          Medium (50)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-orange-500 dashed"></span>
          High (70)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-red-500 dashed"></span>
          Critical (80)
        </span>
      </div>
    </div>
  );
}

export default RiskHistogram;
