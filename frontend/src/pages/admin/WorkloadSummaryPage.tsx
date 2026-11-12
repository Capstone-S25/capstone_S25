import { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Download } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';

// ── Static data matching the spec ─────────────────────────────────────────
const DAILY_DATA = [
  { day: 'Tue', real: 9,  false: 3 },
  { day: 'Wed', real: 10, false: 5 },
  { day: 'Thu', real: 7,  false: 3 },
  { day: 'Fri', real: 13, false: 6 },
  { day: 'Sat', real: 10, false: 5 },
  { day: 'Sun', real: 8,  false: 4 },
  { day: 'Mon', real: 6,  false: 2 },
];

const BY_TYPE = [
  { type: 'Smoking',         count: 30 },
  { type: 'Littering',       count: 26 },
  { type: 'Loitering',       count: 25 },
  { type: 'Unattended item', count: 16 },
];
const MAX_COUNT = 30;

// ── Stat card ──────────────────────────────────────────────────────────────
interface StatCardProps {
  title: string;
  subtitle: string;
  children?: ReactNode;
  trend?: 'up' | 'down';
  trendColor?: 'red' | 'green';
  trendText?: string;
}

function StatCard({ title, subtitle, children, trend, trendColor = 'green', trendText }: StatCardProps) {
  const trendCls = trendColor === 'red' ? 'text-red-500' : 'text-green-600';
  return (
    <div className="bg-white rounded-2xl px-5 py-4 flex-1">
      <p className="text-xs text-gray-500 mb-1">{title}</p>
      <p className="text-lg font-bold text-gray-900 leading-tight">{subtitle}</p>
      {children}
      {trendText && (
        <div className={`flex items-center gap-1 mt-2 text-xs font-medium ${trendCls}`}>
          {trend === 'up'
            ? <TrendingUp size={13} />
            : <TrendingDown size={13} />}
          {trendText}
        </div>
      )}
    </div>
  );
}

// ── Alerts progress bar ────────────────────────────────────────────────────
function AlertsBar() {
  const total = 104;
  const accepted  = 68;
  const dismissed = 29;
  const pending   = 7;
  return (
    <div className="mt-2">
      <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
        <div className="rounded-l-full bg-green-500" style={{ width: `${(accepted  / total) * 100}%` }} />
        <div className="bg-gray-300"                 style={{ width: `${(dismissed / total) * 100}%` }} />
        <div className="rounded-r-full bg-blue-400"  style={{ width: `${(pending   / total) * 100}%` }} />
      </div>
      <div className="flex gap-3 mt-1.5 flex-wrap">
        <span className="flex items-center gap-1 text-xs text-gray-500">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
          {accepted} accepted
        </span>
        <span className="flex items-center gap-1 text-xs text-gray-500">
          <span className="w-2 h-2 rounded-full bg-gray-300 inline-block" />
          {dismissed} dismissed
        </span>
        <span className="flex items-center gap-1 text-xs text-gray-500">
          <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
          {pending} pending
        </span>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function WorkloadSummaryPage() {
  return (
    <AppLayout role="admin" activePath="/admin/workload">
      <div style={{ padding: '20px 28px 20px 20px' }}>

        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-gray-900">Workload Summary</h1>
            <select
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #d0d5dd',
                borderRadius: '8px',
                paddingLeft: '12px',
                paddingRight: '10px',
                paddingTop: '9px',
                paddingBottom: '9px',
                fontSize: '14px',
                color: '#101828',
                boxShadow: '0px 1px 2px 0px rgba(16,24,40,0.05)',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              <option>Period: 5–11 Oct</option>
              <option>Period: 1–4 Oct</option>
            </select>
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" size="sm">
              <Download size={14} />
              Export PDF
            </Button>
            <Button variant="secondary" size="sm">
              <Download size={14} />
              Export CSV
            </Button>
          </div>
        </div>

        {/* Summary text */}
        <div className="rounded-xl px-5 py-3.5 mb-5 text-sm leading-relaxed" style={{ backgroundColor: '#eef2fb', border: '1px solid #c7d7f8', color: '#1f4fd1' }}>
          This week: <strong>22 of 1,344 hours</strong> flagged for review (1.6%). 97 clips reviewed:
          <strong> 68 real incidents</strong>, 29 false alarms. Most false alarms were people waiting in Corridor 2.
          Median review time <strong>42 s</strong>.
        </div>

        {/* 4 stat cards */}
        <div className="flex gap-4 mb-6">
          <StatCard
            title="Footage recorded"
            subtitle="1,344 h"
          >
            <p className="text-xs text-gray-400 mt-1">8 cameras · 7 days</p>
          </StatCard>

          <StatCard title="Alerts raised" subtitle="104">
            <AlertsBar />
          </StatCard>

          <StatCard
            title="Escalated cases"
            subtitle="12"
            trend="up"
            trendColor="red"
            trendText="2 more than last week"
          />

          <StatCard
            title="Confirmed real"
            subtitle="70%"
            trend="up"
            trendColor="green"
            trendText="Up from 55% last week"
          />
        </div>

        {/* Two charts */}
        <div className="flex gap-5">

          {/* Left: stacked bar chart (60%) */}
          <div className="bg-white rounded-2xl p-5" style={{ flex: '0 0 60%' }}>
            <p className="text-sm font-semibold text-gray-800 mb-4">Alerts per day</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={DAILY_DATA} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  cursor={{ fill: '#f9fafb' }}
                />
                <Legend
                  iconType="square"
                  iconSize={10}
                  formatter={(value) => (
                    <span style={{ fontSize: 12, color: '#6b7280' }}>
                      {value === 'real' ? 'Real incident' : 'False alarm'}
                    </span>
                  )}
                />
                <Bar dataKey="real"  name="real"  stackId="a" fill="#1f4fd1" radius={[0, 0, 0, 0]} />
                <Bar dataKey="false" name="false" stackId="a" fill="#bfdbfe" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Right: alerts by type (40%) */}
          <div className="bg-white rounded-2xl p-5 flex-1">
            <p className="text-sm font-semibold text-gray-800 mb-5">Alerts by type</p>
            <div className="flex flex-col gap-4">
              {BY_TYPE.map(({ type, count }) => (
                <div key={type}>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-sm text-gray-700">{type}</span>
                    <span className="text-sm font-semibold text-gray-900">{count}</span>
                  </div>
                  <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(count / MAX_COUNT) * 100}%`,
                        backgroundColor: '#1f4fd1',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </AppLayout>
  );
}
