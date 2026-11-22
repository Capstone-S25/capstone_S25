// Monitoring rules shared by the admin and staff screens.
//
// There is no rules endpoint in the backend yet, so the rules live in this
// browser (localStorage) and every page reads them through useRules(): saving
// on Admin › Monitoring rules changes what staff see straight away, in this tab
// and in any other open tab.
import { useSyncExternalStore } from 'react';
import type { MonitoringRule, Severity } from '../types';

/** The behaviours a rule can watch for, in the order the dropdowns list them. */
export const RULE_TYPES = ['Littering', 'Smoking', 'Loitering'] as const;
export type RuleType = (typeof RULE_TYPES)[number];

export const SEVERITIES: Severity[] = ['Critical', 'Moderate', 'Minor'];

export const DEFAULT_RULES: MonitoringRule[] = [
  { id: 'rule-01', location: 'Meeting Room 1', cameraIds: ['cam-01'], detectType: 'Smoking', activeFrom: '00:00', activeTo: '23:59', severity: 'Moderate', minConfidence: 0.7, enabled: true },
  { id: 'rule-02', location: 'Meeting Room 1', cameraIds: ['cam-01'], detectType: 'Littering', activeFrom: '08:00', activeTo: '20:00', severity: 'Minor', minConfidence: 0.65, enabled: true },
  { id: 'rule-03', location: 'Main Entrance', cameraIds: ['cam-02'], detectType: 'Loitering', activeFrom: '22:00', activeTo: '06:00', severity: 'Minor', minConfidence: 0.7, enabled: true },
  { id: 'rule-04', location: 'Corridor 2', cameraIds: ['cam-03'], detectType: 'Loitering', activeFrom: '00:00', activeTo: '23:59', severity: 'Minor', minConfidence: 0.7, enabled: true },
  { id: 'rule-05', location: 'Carpark B2', cameraIds: ['cam-04'], detectType: 'Smoking', activeFrom: '22:00', activeTo: '06:00', severity: 'Critical', minConfidence: 0.7, enabled: true },
  { id: 'rule-06', location: 'Lift Lobby 1', cameraIds: ['cam-05'], detectType: 'Littering', activeFrom: '00:00', activeTo: '23:59', severity: 'Minor', minConfidence: 0.7, enabled: true },
  { id: 'rule-07', location: 'Staircase B', cameraIds: ['cam-06'], detectType: 'Smoking', activeFrom: '00:00', activeTo: '23:59', severity: 'Moderate', minConfidence: 0.7, enabled: false },
  { id: 'rule-08', location: 'Meeting Room 1', cameraIds: ['cam-01'], detectType: 'Loitering', activeFrom: '18:00', activeTo: '08:00', severity: 'Minor', minConfidence: 0.6, enabled: true },
  { id: 'rule-09', location: 'Staircase B', cameraIds: ['cam-06'], detectType: 'Littering', activeFrom: '06:00', activeTo: '22:00', severity: 'Minor', minConfidence: 0.65, enabled: true },
  { id: 'rule-10', location: 'Pantry L3', cameraIds: ['cam-07'], detectType: 'Littering', activeFrom: '07:00', activeTo: '21:00', severity: 'Minor', minConfidence: 0.6, enabled: false },
];

const KEY = 'insight.rules.v1';
const listeners = new Set<() => void>();
let cache: MonitoringRule[] | null = null;

function read(): MonitoringRule[] {
  if (cache) return cache;
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    cache = Array.isArray(saved) ? (saved as MonitoringRule[]) : DEFAULT_RULES;
  } catch {
    cache = DEFAULT_RULES;
  }
  return cache;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  // Another tab saved: drop the cache so the next read picks the change up.
  const onStorage = (e: StorageEvent) => { if (e.key === KEY) { cache = null; listener(); } };
  window.addEventListener('storage', onStorage);
  return () => { listeners.delete(listener); window.removeEventListener('storage', onStorage); };
}

/** Saves the full rule list and tells every page using useRules(). */
export function saveRules(rules: MonitoringRule[]) {
  cache = rules;
  try { localStorage.setItem(KEY, JSON.stringify(rules)); } catch { /* private mode: keep it for this tab */ }
  listeners.forEach((l) => l());
}

/** Puts the demo rules back. */
export function resetRules() {
  try { localStorage.removeItem(KEY); } catch { /* nothing saved */ }
  cache = null;
  listeners.forEach((l) => l());
}

export function useRules(): MonitoringRule[] {
  return useSyncExternalStore(subscribe, read, read);
}

const minutes = (hhmm: string) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m; };

/** True when a time ("HH:MM" or "HH:MM:SS") falls in the rule's hours; overnight hours wrap. */
export function isActiveAt(rule: Pick<MonitoringRule, 'activeFrom' | 'activeTo'>, time: string): boolean {
  const t = minutes(time.slice(0, 5));
  const from = minutes(rule.activeFrom);
  const to = minutes(rule.activeTo);
  return from <= to ? t >= from && t <= to : t >= from || t <= to;
}

/** The enabled rule that covers this alert (same place and behaviour, inside its hours), if any. */
export function matchRule(rules: MonitoringRule[], location: string, type: string, time: string): MonitoringRule | undefined {
  const place = location.trim().toLowerCase();
  return rules.find((r) => r.enabled && r.detectType === type && r.location.toLowerCase() === place && isActiveAt(r, time));
}

/** "10 PM – 6 AM", or "All day" for 00:00 – 23:59. */
export function formatHours(from: string, to: string): string {
  if (from === '00:00' && to === '23:59') return 'All day';
  const label = (hhmm: string) => {
    const [h, m] = hhmm.split(':').map(Number);
    const hour = h % 12 || 12;
    return `${hour}${m ? `:${String(m).padStart(2, '0')}` : ''} ${h < 12 ? 'AM' : 'PM'}`;
  };
  return `${label(from)} – ${label(to)}`;
}

/** The rule as one line, e.g. "Carpark B2 · Smoking · 10 PM – 6 AM". */
export function ruleLine(rule: MonitoringRule): string {
  return `${rule.location} · ${rule.detectType} · ${formatHours(rule.activeFrom, rule.activeTo)}`;
}

/** The behaviours a camera's enabled rules watch for. */
export function cameraLooksFor(rules: MonitoringRule[], camId: string): string[] {
  return [...new Set(rules.filter((r) => r.enabled && r.cameraIds.includes(camId)).map((r) => r.detectType))];
}
