import { useState, useEffect } from 'react';
import { Clock, MoreVertical, Check, Plus } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import AddRuleModal from '../../components/modals/AddRuleModal';
import { StatusPill } from '../../components/ui/Badge';
import { SEVERITY_TONE } from '../../components/ui/tones';
import { Toggle } from '../../components/ui/Toggle';
import { PageHeader } from '../../components/ui/Primitives';
import { MOCK_RULES } from '../../data/mock';
import type { MonitoringRule, Severity, IncidentType } from '../../types';

const EXTRA_RULES: MonitoringRule[] = [
  { id: 'rule-07', location: 'Meeting Room 1', cameraIds: ['cam-05'], detectType: 'Loitering', activeFrom: '18:00', activeTo: '08:00', severity: 'Minor', minConfidence: 0.6, enabled: true },
  { id: 'rule-08', location: 'Staircase B', cameraIds: ['cam-06'], detectType: 'Littering', activeFrom: '06:00', activeTo: '22:00', severity: 'Minor', minConfidence: 0.65, enabled: true },
  { id: 'rule-09', location: 'Pantry L3', cameraIds: ['cam-07'], detectType: 'Littering', activeFrom: '07:00', activeTo: '21:00', severity: 'Minor', minConfidence: 0.6, enabled: false },
];

const ALL_RULES: MonitoringRule[] = [...MOCK_RULES, ...EXTRA_RULES];

const SEVERITY_OPTIONS: Severity[] = ['Critical', 'Moderate', 'Minor'];
const DETECT_OPTIONS: IncidentType[] = ['Smoking', 'Littering', 'Loitering', 'Unattended item'];

interface RowState {
  enabled: boolean;
  activeFrom: string;
  activeTo: string;
  severity: Severity;
  minConfidence: number;
  detectType: IncidentType;
}

function buildInitial(rules: MonitoringRule[]): Record<string, RowState> {
  return Object.fromEntries(rules.map((r) => [r.id, {
    enabled: r.enabled,
    activeFrom: r.activeFrom,
    activeTo: r.activeTo,
    severity: r.severity,
    minConfidence: Math.round(r.minConfidence * 100),
    detectType: r.detectType,
  }]));
}

const camLabels = (ids: string[]) => ids.map((id) => 'Cam ' + id.replace('cam-', '').padStart(2, '0')).join(', ');

const CELL_INPUT =
  'rounded-[6px] border border-line bg-white px-[8px] py-[6px] text-[12px] text-ink outline-none transition-colors hover:border-faint focus:border-brand disabled:cursor-not-allowed disabled:bg-head disabled:text-faint';

// Figma table header style: 11px medium muted, 1px line under header, line-soft between rows
const COLS = 'grid grid-cols-[minmax(160px,1.2fr)_150px_220px_190px_120px_170px] items-center gap-[16px]';

export default function MonitoringRulesPage() {
  const [rows, setRows] = useState<Record<string, RowState>>(() => buildInitial(ALL_RULES));
  const [baseline, setBaseline] = useState(rows);
  const [showAddRule, setShowAddRule] = useState(false);
  const [saved, setSaved] = useState(false);

  const dirty = JSON.stringify(rows) !== JSON.stringify(baseline);

  function update<K extends keyof RowState>(id: string, key: K, value: RowState[K]) {
    setRows((prev) => ({ ...prev, [id]: { ...prev[id], [key]: value } }));
  }

  function handleSave() {
    setBaseline(rows);
    setSaved(true);
  }

  useEffect(() => {
    if (!saved) return;
    const t = setTimeout(() => setSaved(false), 2000);
    return () => clearTimeout(t);
  }, [saved]);

  const active = Object.values(rows).filter((r) => r.enabled).length;

  return (
    <AppLayout role="admin" activePath="/admin/rules">
      <PageHeader
        title="Monitoring rules"
        subtitle={`${active} of ${ALL_RULES.length} rules active · the AI flags matching behaviour during each rule's hours.`}
        actions={
          <>
            <Button variant="secondary" icon={<Plus size={16} />} onClick={() => setShowAddRule(true)}>Add rule</Button>
            <Button variant="primary" onClick={handleSave} disabled={!dirty && !saved} icon={saved ? <Check size={16} /> : undefined}>
              {saved ? 'Saved' : 'Save changes'}
            </Button>
          </>
        }
      />

      <div className="w-full overflow-x-auto" role="table" aria-label="Monitoring rules">
        <div className="min-w-[1070px]">
          <div role="row" className={`${COLS} border-b border-line pb-[8px] text-[11px] font-medium text-muted`}>
            <span role="columnheader">Location</span>
            <span role="columnheader">Detect</span>
            <span role="columnheader">Active hours</span>
            <span role="columnheader">Severity</span>
            <span role="columnheader">Min. confidence</span>
            <span role="columnheader">Status</span>
          </div>

          {ALL_RULES.map((rule) => {
            const row = rows[rule.id];
            const off = !row.enabled;
            return (
              <div key={rule.id} role="row" className={`${COLS} border-b border-line-soft py-[10px] transition-colors ${off ? 'bg-head/60' : 'hover:bg-head/60'}`}>
                <span role="cell" className="flex min-w-0 flex-col gap-[2px]">
                  <span className={`truncate text-[13px] font-medium ${off ? 'text-faint' : 'text-ink'}`}>{rule.location}</span>
                  <span className={`truncate text-[12px] ${off ? 'text-faint' : 'text-muted'}`}>{camLabels(rule.cameraIds)}</span>
                </span>

                <span role="cell">
                  <select
                    aria-label={`Detect type for ${rule.location}`}
                    value={row.detectType}
                    onChange={(e) => update(rule.id, 'detectType', e.target.value as IncidentType)}
                    disabled={off}
                    className={`${CELL_INPUT} w-full`}
                  >
                    {DETECT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </span>

                <span role="cell" className={`inline-flex w-fit items-center gap-[6px] rounded-[6px] border px-[8px] py-[6px] ${off ? 'border-line bg-head' : 'border-line bg-white focus-within:border-brand'}`}>
                  <Clock size={13} className={off ? 'text-faint' : 'text-muted'} />
                  <input
                    type="time"
                    aria-label="From"
                    value={row.activeFrom}
                    onChange={(e) => update(rule.id, 'activeFrom', e.target.value)}
                    disabled={off}
                    className={`w-[78px] border-none bg-transparent text-[12px] outline-none [&::-webkit-calendar-picker-indicator]:hidden ${off ? 'text-faint' : 'text-ink'}`}
                  />
                  <span className="text-[12px] text-faint">–</span>
                  <input
                    type="time"
                    aria-label="To"
                    value={row.activeTo}
                    onChange={(e) => update(rule.id, 'activeTo', e.target.value)}
                    disabled={off}
                    className={`w-[78px] border-none bg-transparent text-[12px] outline-none [&::-webkit-calendar-picker-indicator]:hidden ${off ? 'text-faint' : 'text-ink'}`}
                  />
                </span>

                <span role="cell" className="flex items-center gap-[8px]">
                  <StatusPill tone={off ? 'neutral' : SEVERITY_TONE[row.severity]} className="w-[76px] justify-start">{row.severity}</StatusPill>
                  <select
                    aria-label={`Severity for ${rule.location}`}
                    value={row.severity}
                    onChange={(e) => update(rule.id, 'severity', e.target.value as Severity)}
                    disabled={off}
                    className={CELL_INPUT}
                  >
                    {SEVERITY_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </span>

                <span role="cell" className={`inline-flex w-fit items-center overflow-hidden rounded-[6px] border border-line ${off ? 'bg-head' : 'bg-white focus-within:border-brand'}`}>
                  <input
                    type="number"
                    aria-label={`Minimum confidence for ${rule.location}`}
                    min={0}
                    max={100}
                    value={row.minConfidence}
                    onChange={(e) => update(rule.id, 'minConfidence', Number(e.target.value))}
                    disabled={off}
                    className={`w-[52px] border-none bg-transparent px-[8px] py-[6px] text-right text-[12px] outline-none ${off ? 'text-faint' : 'text-ink'}`}
                  />
                  <span className={`pr-[8px] text-[12px] ${off ? 'text-faint' : 'text-muted'}`}>%</span>
                </span>

                <span role="cell" className="flex items-center gap-[10px]">
                  <Toggle checked={row.enabled} onChange={(v) => update(rule.id, 'enabled', v)} label={`Enable rule for ${rule.location}`} />
                  <span className={`w-[52px] text-[12px] ${off ? 'text-faint' : 'text-ink-2'}`}>{row.enabled ? 'Enabled' : 'Disabled'}</span>
                  <button type="button" aria-label="More options" className="flex size-[28px] cursor-pointer items-center justify-center rounded-[6px] border-none bg-transparent text-muted hover:bg-head hover:text-ink">
                    <MoreVertical size={16} />
                  </button>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <AddRuleModal isOpen={showAddRule} onClose={() => setShowAddRule(false)} onAdd={() => setShowAddRule(false)} />
    </AppLayout>
  );
}
