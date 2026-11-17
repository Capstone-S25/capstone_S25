import { useState } from 'react';
import { Download } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { SeverityBadge, OutcomeDot, OutcomeTabs } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { ClipThumb, EmptyState, PageHeader } from '../../components/ui/Primitives';
import { MOCK_INCIDENTS, MOCK_STAFF } from '../../data/mock';
import type { Incident, IncidentOutcome } from '../../types';
import ClipDetailDrawer from '../../components/drawers/ClipDetailDrawer';

type Tab = 'All' | IncidentOutcome;

const TABS: Tab[] = ['All', 'Accepted', 'Dismissed', 'Escalated'];

const STAFF_MAP: Record<string, string> = Object.fromEntries(MOCK_STAFF.map((s) => [s.id, s.name]));

const LOCATIONS = ['Carpark B2', 'Lift Lobby 1', 'Corridor 2', 'Main entrance', 'Meeting Room 1', 'Staircase B', 'Pantry L3'];

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }),
    time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }),
    reviewed: new Date(d.getTime() + 3 * 60 * 1000).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }),
  };
}

const camLabel = (camId: string) => 'Cam ' + camId.replace('cam-', '').padStart(2, '0');

// Column widths from Figma: History table
const COLS = 'grid grid-cols-[84px_minmax(0,1fr)_80px_170px_100px_100px_150px] gap-[16px]';

export default function ReviewHistoryPage() {
  const [activeTab, setActiveTab] = useState<Tab>('All');
  const [severity, setSeverity] = useState('all');
  const [location, setLocation] = useState('all');
  const [period, setPeriod] = useState('7d');
  const [selectedInc, setSelectedInc] = useState<Incident | null>(null);

  const reviewed = MOCK_INCIDENTS.filter((i) => i.outcome !== 'Pending');

  const tabCounts: Partial<Record<Tab, number>> = {
    All: reviewed.length,
    Accepted: reviewed.filter((i) => i.outcome === 'Accepted').length,
    Dismissed: reviewed.filter((i) => i.outcome === 'Dismissed').length,
    Escalated: reviewed.filter((i) => i.outcome === 'Escalated').length,
  };

  const filtered = reviewed.filter((inc) => {
    if (activeTab !== 'All' && inc.outcome !== activeTab) return false;
    if (severity !== 'all' && inc.severity !== severity) return false;
    if (location !== 'all' && inc.location !== location) return false;
    return true;
  });

  return (
    <AppLayout role="staff" activePath="/staff/review-history">
      <PageHeader
        title="Review history"
        subtitle="Every clip your team has reviewed, with its outcome."
        actions={<Button variant="secondary" icon={<Download size={16} strokeWidth={1.75} />} onClick={() => window.print()}>Export PDF</Button>}
      />

      {/* Filters */}
      <div className="-mt-[2px] flex w-full flex-wrap items-center gap-[8px]">
        <OutcomeTabs<Tab> tabs={TABS} active={activeTab} counts={tabCounts} onChange={setActiveTab} />
        <div className="flex-1" />
        <FilterSelect
          label="Severity"
          value={severity}
          onChange={setSeverity}
          options={[{ value: 'all', label: 'All' }, { value: 'Critical', label: 'Critical' }, { value: 'Moderate', label: 'Moderate' }, { value: 'Minor', label: 'Minor' }]}
        />
        <FilterSelect
          label="Location"
          value={location}
          onChange={setLocation}
          options={[{ value: 'all', label: 'All locations' }, ...LOCATIONS.map((l) => ({ value: l, label: l }))]}
        />
        <FilterSelect
          label="Date"
          value={period}
          onChange={setPeriod}
          options={[{ value: '7d', label: 'Last 7 days' }, { value: '30d', label: 'Last 30 days' }, { value: '90d', label: 'Last 90 days' }]}
        />
      </div>

      {/* Table */}
      <div className="w-full overflow-x-auto" role="table" aria-label="Review history">
        <div className="min-w-[900px]">
          <div role="row" className={`${COLS} border-b border-line pb-[8px] text-[11px] font-medium text-muted`}>
            <span role="columnheader">Clip</span>
            <span role="columnheader">Incident</span>
            <span role="columnheader">Camera</span>
            <span role="columnheader">Date &amp; time</span>
            <span role="columnheader">Severity</span>
            <span role="columnheader">Outcome</span>
            <span role="columnheader">Reviewed by</span>
          </div>

          {filtered.map((inc, idx) => {
            const { date, time, reviewed: reviewedAt } = formatDateTime(inc.dateTime);
            const assignee = inc.assignedTo ? STAFF_MAP[inc.assignedTo] ?? '—' : '—';
            return (
              <div
                key={inc.id}
                role="row"
                tabIndex={0}
                onClick={() => setSelectedInc(inc)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedInc(inc); } }}
                className={`${COLS} cursor-pointer items-center border-b border-line-soft py-[10px] transition-colors hover:bg-head focus-visible:bg-head ${selectedInc?.id === inc.id ? 'bg-head' : ''}`}
              >
                <span role="cell"><ClipThumb seed={idx} className="h-[44px] w-[72px] rounded-[4px]" iconSize={14} /></span>
                <span role="cell" className="flex min-w-0 flex-col gap-[2px]">
                  <span className="truncate text-[13px] font-medium text-ink">{inc.type} · {inc.location}</span>
                  <span className="truncate text-[12px] text-muted">{inc.description}</span>
                </span>
                <span role="cell" className="text-[12px] text-ink-2">{camLabel(inc.camId)}</span>
                <span role="cell" className="text-[12px] text-ink-2">{date}, {time}</span>
                <span role="cell"><SeverityBadge severity={inc.severity} /></span>
                <span role="cell"><OutcomeDot outcome={inc.outcome} /></span>
                <span role="cell" className="truncate text-[12px] text-ink-2">{assignee}{assignee !== '—' ? ` · ${reviewedAt}` : ''}</span>
              </div>
            );
          })}

          {filtered.length === 0 && <EmptyState>No reviewed clips match these filters.</EmptyState>}
        </div>
      </div>

      <ClipDetailDrawer inc={selectedInc} onClose={() => setSelectedInc(null)} />
    </AppLayout>
  );
}
