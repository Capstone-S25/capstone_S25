import type { Severity, IncidentOutcome } from '../../types';

export type PillTone = 'blue' | 'red' | 'amber' | 'yellow' | 'green' | 'neutral';

export const SEVERITY_TONE: Record<Severity, PillTone> = {
  Critical: 'red',
  Moderate: 'amber',
  Minor:    'yellow',
};

/** Solid dot colour used for compact severity markers (queue items etc.). */
export const SEVERITY_DOT: Record<Severity, string> = {
  Critical: 'var(--color-danger)',
  Moderate: 'var(--color-warning-bar)',
  Minor:    'var(--color-caution-dot)',
};

export const OUTCOME_DOT: Record<IncidentOutcome, string> = {
  Escalated: 'var(--color-danger)',
  Accepted:  'var(--color-success)',
  Dismissed: 'var(--color-neutral)',
  Pending:   'var(--color-warning-bar)',
};
