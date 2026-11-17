import type { ReactNode } from 'react';
import { Severity, IncidentOutcome } from '../../types';
import { SEVERITY_TONE, OUTCOME_DOT, type PillTone } from './tones';

// ── StatusPill (Figma: v2/Status pill) ─────────────────────────────────────
// Rounded pill, 6px dot + 11px medium label.
// Tones: Blue (info), Red (critical), Amber (moderate / pending),
//        Yellow (minor), Green (accepted / live), Neutral.


const TONE: Record<PillTone, { bg: string; fg: string; dot: string }> = {
  blue:    { bg: 'var(--color-brand-soft)',   fg: 'var(--color-brand)',   dot: 'var(--color-brand)' },
  red:     { bg: 'var(--color-danger-soft)',  fg: 'var(--color-danger)',  dot: 'var(--color-danger)' },
  amber:   { bg: 'var(--color-warning-soft)', fg: 'var(--color-warning)', dot: 'var(--color-warning-bar)' },
  yellow:  { bg: 'var(--color-caution-soft)', fg: 'var(--color-caution)', dot: 'var(--color-caution-dot)' },
  green:   { bg: 'var(--color-success-soft)', fg: 'var(--color-success)', dot: 'var(--color-success)' },
  neutral: { bg: 'var(--color-neutral-soft)', fg: 'var(--color-neutral)', dot: 'var(--color-neutral)' },
};

interface StatusPillProps {
  tone?: PillTone;
  children: ReactNode;
  className?: string;
  dot?: boolean;
}

export function StatusPill({ tone = 'blue', children, className = '', dot = true }: StatusPillProps) {
  const t = TONE[tone];
  return (
    <span
      className={`inline-flex items-center gap-[6px] rounded-full py-[4px] pr-[10px] text-[11px] font-medium leading-none whitespace-nowrap ${dot ? 'pl-[8px]' : 'pl-[10px]'} ${className}`}
      style={{ backgroundColor: t.bg, color: t.fg }}
    >
      {dot && <span className="size-[6px] shrink-0 rounded-full" style={{ backgroundColor: t.dot }} />}
      {children}
    </span>
  );
}

// ── Severity ────────────────────────────────────────────────────────────────



export function SeverityBadge({ severity, className = '' }: { severity: Severity; className?: string }) {
  return <StatusPill tone={SEVERITY_TONE[severity]} className={className}>{severity}</StatusPill>;
}

// ── Outcome ─────────────────────────────────────────────────────────────────
// 7px dot + 12px regular ink-2 label (Figma: History table "status" cell)


export function OutcomeDot({ outcome, className = '' }: { outcome: IncidentOutcome; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-[6px] text-[12px] text-ink-2 whitespace-nowrap ${className}`}>
      <span className="size-[7px] shrink-0 rounded-full" style={{ backgroundColor: OUTCOME_DOT[outcome] }} />
      {outcome}
    </span>
  );
}

// Backward-compatible alias
export const OutcomeBadge = OutcomeDot;

// ── OutcomeTabs (Figma: "Outcome tabs") ─────────────────────────────────────
// Underline tabs: 13px medium label + 12px muted count; active = brand + 2px underline.

interface OutcomeTabsProps<T extends string> {
  tabs: T[];
  active: T;
  counts?: Partial<Record<T, number>>;
  onChange: (tab: T) => void;
}

export function OutcomeTabs<T extends string>({ tabs, active, counts, onChange }: OutcomeTabsProps<T>) {
  return (
    <div className="flex items-center gap-[20px]" role="tablist">
      {tabs.map((tab) => {
        const isActive = tab === active;
        const count = counts?.[tab];
        return (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab)}
            className={`-mb-px flex cursor-pointer items-center gap-[6px] border-b-2 bg-transparent py-[8px] text-[13px] font-medium whitespace-nowrap transition-colors ${
              isActive ? 'border-brand text-brand' : 'border-transparent text-ink-2 hover:text-ink'
            }`}
          >
            {tab}
            {count !== undefined && <span className="text-[12px] font-normal text-muted">{count}</span>}
          </button>
        );
      })}
    </div>
  );
}
