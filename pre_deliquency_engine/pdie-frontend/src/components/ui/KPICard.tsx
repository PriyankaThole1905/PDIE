import type { ReactNode } from 'react';

interface KPICardProps {
  icon: ReactNode;
  label: string;
  value: string;
  delta?: string;
  deltaType?: 'up' | 'down' | 'neutral';
}

export function KPICard({ icon, label, value, delta, deltaType = 'neutral' }: KPICardProps) {
  const deltaColors = {
    up: 'text-red-600 bg-red-50',
    down: 'text-green-600 bg-green-50',
    neutral: 'text-gray-600 bg-gray-50',
  };

  return (
    <div className="kpi-card">
      <div className="flex items-start justify-between">
        <div className="text-2xl mb-2">{icon}</div>
        {delta && (
          <span className={`text-xs font-semibold px-2 py-1 rounded-full ${deltaColors[deltaType]}`}>
            {delta}
          </span>
        )}
      </div>
      <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-3xl font-black text-slate-800 tracking-tight">{value}</p>
    </div>
  );
}

export default KPICard;
