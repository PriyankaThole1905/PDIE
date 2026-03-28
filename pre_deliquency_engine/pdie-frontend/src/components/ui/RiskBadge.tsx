import type { RiskCategory } from '../../types';

interface RiskBadgeProps {
  category: RiskCategory;
  score?: number;
  showScore?: boolean;
}

export function RiskBadge({ category, score, showScore = true }: RiskBadgeProps) {
  const config = {
    CRITICAL: {
      className: 'risk-critical',
      label: 'CRITICAL',
    },
    HIGH: {
      className: 'risk-high',
      label: 'HIGH',
    },
    MEDIUM: {
      className: 'risk-medium',
      label: 'MEDIUM',
    },
    LOW: {
      className: 'risk-low',
      label: 'LOW',
    },
  };

  const { className, label } = config[category];

  return (
    <span className={className}>
      {label} {showScore && score !== undefined && `(${Math.round(score)})`}
    </span>
  );
}

export default RiskBadge;
