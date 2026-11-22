import { useState } from 'react';
import { ArrowUp, Download, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { BarMeter, PageHeader } from '../../components/ui/Primitives';

const DAILY_DATA = [
  { day: 'Tue', real: 9, false: 3 },
  { day: 'Wed', real: 10, false: 5 },
  { day: 'Thu', real: 7, false: 3 },
  { day: 'Fri', real: 13, false: 6 },
  { day: 'Sat', real: 10, false: 5 },
  { day: 'Sun', real: 8, false: 4 },
  { day: 'Mon', real: 6, false: 2 },
];

const BY_TYPE = [
  { type: 'Smoking', count: 36 },
  { type: 'Littering', count: 33 },
  { type: 'Loitering', count: 28 },
];
const MAX_COUNT = Math.max(...BY_TYPE.map((t) => t.count));

const ALERTS = { accepted: 68, dismissed: 29, pending: 7 };
const ALERTS_TOTAL = ALERTS.accepted + ALERTS.dismissed + ALERTS.pending;

// Hex values mirror the theme tokens (SVG attributes can't read CSS vars reliably)
const C_BRAND = '#0041c0';
const C_BRAND_MID = '#9db5ea';
const C_MUTED = '#667089';
const C_LINE = '#e8ecf3';

export default function WorkloadSummaryPage() {
  const [period, setPeriod] = useState('5-11');

  return (
    <AppLayout role="admin" activePath="/admin/workload">
      <PageHeader
        title="Workload summary"
        subtitle="How much footage the AI screened and how your team handled it."
        actions={
          <>
            <FilterSelect label="Period" value={period} onChange={setPeriod} options={[{ value: '5-11', label: '5–11 Oct' }, { value: '1-4', label: '1–4 Oct' }]} />
            <Button variant="secondary" icon={<Download size={16} strokeWidth={1.75} />} onClick={() => window.print()}>Export PDF</Button>
            <Button variant="secondary" icon={<Download size={16} strokeWidth={1.75} />}>Export CSV</Button>
          </>
        }
      />

      {/* Summary */}
      <div className="flex items-start gap-[10px] rounded-[8px] bg-brand-soft px-[16px] py-[12px] text-[13px] leading-[1.5] text-ink">
        <Sparkles size={16} className="mt-[2px] shrink-0 text-brand" />
        <p className="m-0">
          This week <strong className="font-semibold">22 of 1,344 hours</strong> were flagged for review (1.6%). 97 clips reviewed:
          <strong className="font-semibold"> 68 real incidents</strong>, 29 false alarms. Most false alarms were people waiting in Corridor 2.
          Median review time <strong className="font-semibold">42 s</strong>.
        </p>
      </div>

      {/* Stats */}
      <div className="flex w-full items-start border-b border-line-soft pb-[16px]">
        <Stat first label="Footage recorded" value="1,344 h" foot={<span className="text-[11px] text-muted">8 cameras · 7 days</span>} />
        <Stat
          label="Alerts raised"
          value={String(ALERTS_TOTAL)}
          foot={
            <span className="flex w-full flex-col gap-[6px]">
              <span className="flex h-[5px] w-full gap-[2px] overflow-hidden rounded-[2.5px]">
                <span className="bg-success" style={{ width: `${(ALERTS.accepted / ALERTS_TOTAL) * 100}%` }} />
                <span className="bg-neutral" style={{ width: `${(ALERTS.dismissed / ALERTS_TOTAL) * 100}%` }} />
                <span className="bg-warning-bar" style={{ width: `${(ALERTS.pending / ALERTS_TOTAL) * 100}%` }} />
              </span>
              <span className="flex flex-wrap gap-x-[10px] text-[11px] text-muted">
                <Dot c="var(--color-success)">{ALERTS.accepted} accepted</Dot>
                <Dot c="var(--color-neutral)">{ALERTS.dismissed} dismissed</Dot>
                <Dot c="var(--color-warning-bar)">{ALERTS.pending} pending</Dot>
              </span>
            </span>
          }
        />
        <Stat label="Escalated cases" value="12" foot={<Trend good={false}>2 more than last week</Trend>} />
        <Stat label="Confirmed real" value="70%" foot={<Trend good>Up from 55% last week</Trend>} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-[32px] lg:grid-cols-[3fr_2fr]">
        <div className="flex min-w-0 flex-col gap-[12px]">
          <div className="flex items-center gap-[16px]">
            <p className="m-0 text-[14px] font-medium text-ink">Alerts per day</p>
            <div className="flex-1" />
            <Dot c={C_BRAND} square>Real incident</Dot>
            <Dot c={C_BRAND_MID} square>False alarm</Dot>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={DAILY_DATA} barCategoryGap="32%">
              <CartesianGrid vertical={false} stroke={C_LINE} />
              <XAxis dataKey="day" tick={{ fontSize: 11, fill: C_MUTED }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: C_MUTED }} axisLine={false} tickLine={false} width={28} />
              <Tooltip
                cursor={{ fill: '#f4f6fb' }}
                contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #dde3ee', boxShadow: 'none', fontFamily: 'inherit' }}
                formatter={(v, name) => [v, name === 'real' ? 'Real incident' : 'False alarm']}
              />
              <Bar dataKey="real" stackId="a" fill={C_BRAND} />
              <Bar dataKey="false" stackId="a" fill={C_BRAND_MID} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="flex min-w-0 flex-col gap-[12px]">
          <p className="m-0 text-[14px] font-medium text-ink">Alerts by type</p>
          <div className="flex flex-col gap-[14px] pt-[6px]">
            {BY_TYPE.map(({ type, count }) => (
              <div key={type} className="flex items-center gap-[12px]">
                <span className="w-[110px] shrink-0 text-[12px] text-ink-2">{type}</span>
                <BarMeter pct={(count / MAX_COUNT) * 100} className="flex-1" />
                <span className="w-[24px] shrink-0 text-right text-[12px] font-medium text-ink">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

function Stat({ label, value, foot, first = false }: { label: string; value: string; foot?: React.ReactNode; first?: boolean }) {
  return (
    <div className={`flex min-w-0 flex-1 flex-col gap-[4px] pr-[16px] ${first ? '' : 'border-l border-line-soft pl-[20px]'}`}>
      <span className="text-[12px] font-medium text-ink-2">{label}</span>
      <span className="text-[24px] font-semibold text-ink">{value}</span>
      {foot}
    </div>
  );
}

function Trend({ good, children }: { good: boolean; children: React.ReactNode }) {
  return (
    <span className="flex items-center gap-[3px] text-[11px] text-muted">
      <ArrowUp size={10} strokeWidth={3} style={{ color: good ? 'var(--color-success)' : 'var(--color-danger)' }} />
      {children}
    </span>
  );
}

function Dot({ c, children, square = false }: { c: string; children: React.ReactNode; square?: boolean }) {
  return (
    <span className="flex items-center gap-[5px] text-[11px] text-muted">
      <span className={`size-[8px] ${square ? 'rounded-[2px]' : 'rounded-full'}`} style={{ backgroundColor: c }} />
      {children}
    </span>
  );
}
