import { Severity, IncidentOutcome } from '../../types';

// ── SeverityBadge ─────────────────────────────────────────────────────────
// Rounded pill: rounded-[6px] px-[10px] py-[4px] text-[13px] semibold
// Critical: bg #f2c0ba text #8c2a22
// Moderate: bg #f6d2b4 text #8f4413
// Minor:    bg #fbf2d3 text #7a6414

interface SeverityBadgeProps {
  severity: Severity;
  className?: string;
}

const SEVERITY_PILL: Record<Severity, { bg: string; color: string }> = {
  Critical: { bg: '#f2c0ba', color: '#8c2a22' },
  Moderate: { bg: '#f6d2b4', color: '#8f4413' },
  Minor:    { bg: '#fbf2d3', color: '#7a6414' },
};

export function SeverityBadge({ severity, className = '' }: SeverityBadgeProps) {
  const cfg = SEVERITY_PILL[severity];
  return (
    <span
      className={`inline-flex items-center font-semibold ${className}`}
      style={{
        backgroundColor: cfg.bg,
        color: cfg.color,
        borderRadius: '6px',
        padding: '4px 10px',
        fontSize: '13px',
        lineHeight: 1.4,
      }}
    >
      {severity}
    </span>
  );
}

// ── OutcomeDot ────────────────────────────────────────────────────────────
// 8px colored circle + text #111827 medium text-[14px]
// Escalated: #ef4444 dot
// Accepted:  #22c55e dot
// Dismissed: #9ca3af dot
// Pending:   #3b82f6 dot

interface OutcomeDotProps {
  outcome: IncidentOutcome;
  className?: string;
}

const OUTCOME_DOT: Record<IncidentOutcome, string> = {
  Escalated: '#ef4444',
  Accepted:  '#22c55e',
  Dismissed: '#9ca3af',
  Pending:   '#3b82f6',
};

export function OutcomeDot({ outcome, className = '' }: OutcomeDotProps) {
  const dotColor = OUTCOME_DOT[outcome];
  return (
    <span className={`inline-flex items-center gap-[6px] ${className}`}>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: dotColor,
          flexShrink: 0,
          display: 'inline-block',
        }}
      />
      <span style={{ fontSize: '14px', fontWeight: 500, color: '#111827' }}>{outcome}</span>
    </span>
  );
}

// Keep OutcomeBadge as an alias to OutcomeDot for backward compat
export const OutcomeBadge = OutcomeDot;

// ── OutcomeTabs ───────────────────────────────────────────────────────────
// White bg card with pill buttons
// Container: bg-white p-[4px] rounded-[10px] gap-[2px]
// Active:   bg-[#1f4fd1] px-[14px] py-[8px] rounded-[8px] white semibold, count in #dbe4fb
// Inactive: no bg, px-[14px] py-[8px] rounded-[8px] text #374151 medium, count in #6b7280

interface OutcomeTabsProps<T extends string> {
  tabs: T[];
  active: T;
  counts?: Partial<Record<T, number>>;
  onChange: (tab: T) => void;
}

export function OutcomeTabs<T extends string>({ tabs, active, counts, onChange }: OutcomeTabsProps<T>) {
  return (
    <div
      className="inline-flex"
      style={{
        backgroundColor: '#ffffff',
        padding: '4px',
        borderRadius: '10px',
        gap: '2px',
      }}
    >
      {tabs.map((tab) => {
        const isActive = tab === active;
        const count = counts?.[tab];
        return (
          <button
            key={tab}
            type="button"
            onClick={() => onChange(tab)}
            style={{
              backgroundColor: isActive ? '#1f4fd1' : 'transparent',
              color: isActive ? '#ffffff' : '#374151',
              padding: '8px 14px',
              borderRadius: '8px',
              fontWeight: isActive ? 600 : 500,
              fontSize: '14px',
              border: 'none',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              whiteSpace: 'nowrap',
            }}
          >
            {tab}
            {count !== undefined && (
              <span
                style={{
                  color: isActive ? '#dbe4fb' : '#6b7280',
                  fontSize: '13px',
                  fontWeight: isActive ? 600 : 400,
                }}
              >
                {count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
