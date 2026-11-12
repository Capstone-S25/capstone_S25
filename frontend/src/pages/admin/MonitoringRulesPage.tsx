import { useState, useEffect } from 'react';
import { Clock, MoreVertical, Check } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import AddRuleModal from '../../components/modals/AddRuleModal';
import { SeverityBadge } from '../../components/ui/Badge';
import { Toggle } from '../../components/ui/Toggle';
import { MOCK_RULES } from '../../data/mock';
import type { MonitoringRule, Severity, IncidentType } from '../../types';

// ── Extend to 9 rules ─────────────────────────────────────────────────────
const EXTRA_RULES: MonitoringRule[] = [
  {
    id: 'rule-07',
    location: 'Meeting Room 1',
    cameraIds: ['cam-05'],
    detectType: 'Loitering',
    activeFrom: '18:00',
    activeTo: '08:00',
    severity: 'Minor',
    minConfidence: 0.60,
    enabled: true,
  },
  {
    id: 'rule-08',
    location: 'Staircase B',
    cameraIds: ['cam-06'],
    detectType: 'Littering',
    activeFrom: '06:00',
    activeTo: '22:00',
    severity: 'Minor',
    minConfidence: 0.65,
    enabled: true,
  },
  {
    id: 'rule-09',
    location: 'Pantry L3',
    cameraIds: ['cam-07'],
    detectType: 'Littering',
    activeFrom: '07:00',
    activeTo: '21:00',
    severity: 'Minor',
    minConfidence: 0.60,
    enabled: false,
  },
];

const ALL_RULES: MonitoringRule[] = [...MOCK_RULES, ...EXTRA_RULES];

const SEVERITY_OPTIONS: Severity[]      = ['Critical', 'Moderate', 'Minor'];
const DETECT_OPTIONS:   IncidentType[]  = ['Smoking', 'Littering', 'Loitering', 'Unattended item'];

// ── Row state ─────────────────────────────────────────────────────────────
interface RowState {
  enabled: boolean;
  activeFrom: string;
  activeTo: string;
  severity: Severity;
  minConfidence: number;
  detectType: IncidentType;
}

function buildInitial(rules: MonitoringRule[]): Record<string, RowState> {
  return Object.fromEntries(rules.map((r) => [
    r.id,
    {
      enabled:       r.enabled,
      activeFrom:    r.activeFrom,
      activeTo:      r.activeTo,
      severity:      r.severity,
      minConfidence: Math.round(r.minConfidence * 100),
      detectType:    r.detectType,
    },
  ]));
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function MonitoringRulesPage() {
  const [rows, setRows] = useState<Record<string, RowState>>(buildInitial(ALL_RULES));
  const [showAddRule, setShowAddRule] = useState(false);
  const [saved, setSaved] = useState(false);

  function update<K extends keyof RowState>(id: string, key: K, value: RowState[K]) {
    setRows((prev) => ({ ...prev, [id]: { ...prev[id], [key]: value } }));
  }

  function handleSave() {
    setSaved(true);
  }

  useEffect(() => {
    if (!saved) return;
    const t = setTimeout(() => setSaved(false), 2000);
    return () => clearTimeout(t);
  }, [saved]);

  return (
    <AppLayout role="admin" activePath="/admin/rules">
      <div style={{ padding: '20px 28px 20px 20px' }}>

        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h1 className="text-xl font-bold text-gray-900">Monitoring Rules</h1>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={() => setShowAddRule(true)}>+ Add rule</Button>
            <Button variant="primary" onClick={handleSave}>
              {saved ? <><Check size={14} /> Saved</> : 'Save changes'}
            </Button>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Location</th>
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Detect</th>
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Active hours</th>
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Severity</th>
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Min. confidence</th>
                <th className="text-left text-xs font-semibold text-gray-500 px-4 py-3 uppercase tracking-wide">Status</th>
              </tr>
            </thead>
            <tbody>
              {ALL_RULES.map((rule) => {
                const row     = rows[rule.id];
                const disabled = !row.enabled;
                const dimText  = disabled ? 'text-gray-300' : 'text-gray-900';
                const dimSub   = disabled ? 'text-gray-300' : 'text-gray-400';

                return (
                  <tr
                    key={rule.id}
                    className={`border-b border-gray-100 last:border-0 transition-colors ${disabled ? 'bg-gray-50' : 'hover:bg-gray-50/50'}`}
                  >
                    {/* Location */}
                    <td className="px-4 py-3">
                      <p className={`font-semibold ${dimText}`}>{rule.location}</p>
                      <p className={`text-xs mt-0.5 ${dimSub}`}>{rule.cameraIds.join(', ')}</p>
                    </td>

                    {/* Detect */}
                    <td className="px-4 py-3">
                      <select
                        value={row.detectType}
                        onChange={(e) => update(rule.id, 'detectType', e.target.value as IncidentType)}
                        disabled={disabled}
                        className={`border border-gray-200 rounded-md px-2 py-1 text-sm bg-white focus:outline-none focus:ring-1 focus:ring-[#1f4fd1] ${disabled ? 'text-gray-300 cursor-not-allowed' : 'text-gray-700'}`}
                      >
                        {DETECT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    </td>

                    {/* Active hours */}
                    <td className="px-4 py-3">
                      <div className={`inline-flex items-center gap-1.5 border rounded-full px-3 py-1 ${disabled ? 'border-gray-200 bg-gray-50' : 'border-gray-300 bg-white'}`}>
                        <Clock size={13} className={disabled ? 'text-gray-300' : 'text-gray-500'} />
                        <input
                          type="text"
                          value={`${row.activeFrom} – ${row.activeTo}`}
                          onChange={(e) => {
                            const parts = e.target.value.split(' – ');
                            if (parts.length === 2) {
                              update(rule.id, 'activeFrom', parts[0]);
                              update(rule.id, 'activeTo',   parts[1]);
                            }
                          }}
                          disabled={disabled}
                          className={`w-28 text-xs bg-transparent outline-none ${disabled ? 'text-gray-300' : 'text-gray-700'}`}
                        />
                      </div>
                    </td>

                    {/* Severity */}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <SeverityBadge severity={row.severity} />
                        <select
                          value={row.severity}
                          onChange={(e) => update(rule.id, 'severity', e.target.value as Severity)}
                          disabled={disabled}
                          className={`border border-gray-200 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-1 focus:ring-[#1f4fd1] ${disabled ? 'text-gray-300 cursor-not-allowed' : 'text-gray-700'}`}
                        >
                          {SEVERITY_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                        </select>
                      </div>
                    </td>

                    {/* Min. confidence */}
                    <td className="px-4 py-3">
                      <div className={`inline-flex items-center border rounded-md overflow-hidden ${disabled ? 'border-gray-200 bg-gray-50' : 'border-gray-300 bg-white'}`}>
                        <input
                          type="number"
                          min={0}
                          max={100}
                          value={row.minConfidence}
                          onChange={(e) => update(rule.id, 'minConfidence', Number(e.target.value))}
                          disabled={disabled}
                          className={`w-14 px-2 py-1 text-sm text-right bg-transparent outline-none ${disabled ? 'text-gray-300' : 'text-gray-700'}`}
                        />
                        <span className={`pr-2 text-sm ${disabled ? 'text-gray-300' : 'text-gray-500'}`}>%</span>
                      </div>
                    </td>

                    {/* Status */}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <Toggle
                          checked={row.enabled}
                          onChange={(v) => update(rule.id, 'enabled', v)}
                          label={`Toggle rule ${rule.id}`}
                        />
                        <span className={`text-xs ${disabled ? 'text-gray-300' : 'text-gray-500'}`}>
                          {row.enabled ? 'Enabled' : 'Disabled'}
                        </span>
                        <button
                          type="button"
                          className="text-gray-400 hover:text-gray-600 ml-1"
                          aria-label="More options"
                        >
                          <MoreVertical size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <AddRuleModal
        isOpen={showAddRule}
        onClose={() => setShowAddRule(false)}
        onAdd={() => setShowAddRule(false)}
      />
    </AppLayout>
  );
}
