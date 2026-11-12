import { useState } from 'react';
import { CheckCircle, ChevronRight, AlertTriangle } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import { MOCK_INCIDENTS, MOCK_STAFF } from '../../data/mock';

// ── Stat card ──────────────────────────────────────────────────────────────
interface StatCardProps {
  count: number;
  label: string;
  delta: string;
  accent: 'red' | 'orange' | 'green';
}

function StatCard({ count, label, delta, accent }: StatCardProps) {
  const colors: Record<string, { num: string; note: string }> = {
    red:    { num: 'text-red-600',    note: 'text-red-500'    },
    orange: { num: 'text-orange-500', note: 'text-orange-500' },
    green:  { num: 'text-green-600',  note: 'text-green-600'  },
  };
  const c = colors[accent];
  return (
    <div className="bg-white rounded-xl border border-gray-200 px-5 py-4 flex-1">
      <div className="flex items-baseline gap-2 mb-1">
        <span className={`text-3xl font-bold ${c.num}`}>{count}</span>
        <span className="text-sm font-medium text-gray-700">{label}</span>
      </div>
      <p className={`text-xs ${c.note}`}>{delta}</p>
    </div>
  );
}

// ── Notification item ──────────────────────────────────────────────────────
interface NotifItemProps {
  text: string;
  time: string;
  unread?: boolean;
}
function NotifItem({ text, time, unread = false }: NotifItemProps) {
  return (
    <div className="flex items-start gap-2.5 py-2.5 border-b border-gray-100 last:border-0">
      {unread
        ? <span className="mt-1.5 w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: '#1f4fd1' }} />
        : <span className="mt-1.5 w-2 h-2 rounded-full bg-transparent flex-shrink-0" />}
      <div className="min-w-0 flex-1">
        <p className="text-xs text-gray-800 leading-snug">{text}</p>
        <p className="text-xs text-gray-400 mt-0.5">{time}</p>
      </div>
    </div>
  );
}

// ── Escalated case row ─────────────────────────────────────────────────────
interface EscalatedRowProps {
  color: string;
  title: string;
  date: string;
  confidence: number;
}
function EscalatedRow({ color, title, date, confidence }: EscalatedRowProps) {
  return (
    <div className="flex items-center gap-2.5 py-2 border-b border-gray-100 last:border-0">
      <span className="w-1 h-10 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-gray-800 truncate">{title}</p>
        <p className="text-xs text-gray-400">{date}</p>
      </div>
      <span className="text-xs font-semibold px-2 py-0.5 rounded-full" style={{ color: '#1f4fd1', backgroundColor: '#dbe4fb' }}>
        {confidence}%
      </span>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function AdminHomePage() {
  const [assignedTo, setAssignedTo] = useState('');

  const escalated = MOCK_INCIDENTS.filter((i) => i.outcome === 'Escalated');
  const resolved  = MOCK_INCIDENTS.filter((i) => i.outcome === 'Accepted').slice(0, 2);

  const severityColor: Record<string, string> = {
    Critical: '#ef4444',
    Moderate: '#f97316',
    Minor:    '#eab308',
  };

  return (
    <AppLayout role="admin" activePath="/admin/home">
      <div className="p-6 max-w-screen-xl mx-auto">

        {/* ── Stat row ── */}
        <div className="flex gap-4 mb-6">
          <StatCard count={3}  label="Escalated to you"    delta="1 more than yesterday"     accent="red"    />
          <StatCard count={9}  label="Dismissed this week" delta="+2 more than last week"     accent="orange" />
          <StatCard count={6}  label="Cameras online"      delta="1 offline since 9AM"        accent="green"  />
        </div>

        {/* ── Two-column main ── */}
        <div className="flex gap-5">

          {/* ── Left: Incident detail ── */}
          <div className="flex-1 min-w-0">
            <div className="bg-white rounded-xl border border-gray-200 p-5">

              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div>
                  <span className="inline-block text-xs font-semibold text-white rounded-full px-2.5 py-0.5 mb-2" style={{ backgroundColor: '#1f4fd1' }}>
                    Critical
                  </span>
                  <h2 className="text-lg font-bold text-gray-900">Smoking · Carpark B2</h2>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2.5 py-1 rounded-full">
                    Unassigned
                  </span>
                  <Button size="sm">Mark resolved</Button>
                </div>
              </div>

              {/* Video placeholder */}
              <div className="bg-gray-200 w-full aspect-video rounded-lg flex items-center justify-center mb-4">
                <span className="text-gray-400 text-sm">Video feed unavailable</span>
              </div>

              {/* Metadata row */}
              <div className="flex items-center gap-4 text-xs text-gray-500 mb-4 flex-wrap">
                <span className="font-medium text-gray-700">Cam 04 · 10:47 PM</span>
                <span>Mon 11 Oct</span>
                <span>3 min 32 s</span>
                <span>1 adult</span>
              </div>

              {/* Description */}
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-700 mb-1">Description</p>
                <p className="text-sm text-gray-600 leading-relaxed">
                  Individual observed smoking near pillar P-04. Smoke clearly visible. Approx. 3 minutes duration before moving off.
                </p>
              </div>

              {/* AI confidence */}
              <div className="flex items-center gap-2 mb-5">
                <AlertTriangle size={14} className="text-orange-400" />
                <span className="text-xs text-gray-500">AI confidence: </span>
                <span className="text-xs font-semibold text-gray-700">94%</span>
              </div>

              {/* Assign dropdown */}
              <div className="border-t border-gray-100 pt-4">
                <label className="text-sm font-medium text-gray-700 mr-3">Assign to staff:</label>
                <select
                  value={assignedTo}
                  onChange={(e) => setAssignedTo(e.target.value)}
                  className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">— Select staff —</option>
                  {MOCK_STAFF.map((s) => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* ── Right: sidebar panels ── */}
          <div className="w-80 flex-shrink-0 flex flex-col gap-4">

            {/* Notifications */}
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-gray-900">
                  Notifications <span className="text-[#1f4fd1] font-normal">[2 new]</span>
                </span>
                <button type="button" className="text-xs text-[#1f4fd1] hover:underline">
                  Mark all read
                </button>
              </div>
              <NotifItem
                text="Cam 08: Pantry 2 is offline since 9:21 AM"
                time="9:21 AM"
                unread
              />
              <NotifItem
                text="Ben Lee received an escalation from Jane Tan"
                time="10:03 AM"
                unread
              />
              <NotifItem
                text="Weekly workload report is ready"
                time="Yesterday"
              />
            </div>

            {/* Escalated cases */}
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-gray-900">
                  Escalated cases <span className="text-[#1f4fd1] font-normal">[1 open]</span>
                </span>
                <button type="button" className="text-xs text-[#1f4fd1] hover:underline flex items-center gap-0.5">
                  View all <ChevronRight size={12} />
                </button>
              </div>
              {escalated.map((inc) => (
                <EscalatedRow
                  key={inc.id}
                  color={severityColor[inc.severity]}
                  title={`${inc.type} · ${inc.location}`}
                  date={new Date(inc.dateTime).toLocaleDateString('en-SG', { day: 'numeric', month: 'short' })}
                  confidence={Math.round(inc.confidence * 100)}
                />
              ))}
            </div>

            {/* Recently resolved */}
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-semibold text-gray-900">Recently resolved</span>
              </div>
              {resolved.map((inc) => (
                <div key={inc.id} className="flex items-center gap-2.5 py-2 border-b border-gray-100 last:border-0">
                  <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-gray-800 truncate">{inc.type} · {inc.location}</p>
                    <p className="text-xs text-gray-400">
                      {new Date(inc.dateTime).toLocaleDateString('en-SG', { day: 'numeric', month: 'short' })}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Recent workload reports */}
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <p className="text-sm font-semibold text-gray-900 mb-3">Recent workload reports</p>
              {[
                { label: '5–11 Oct', dates: '5 Oct – 11 Oct 2026' },
                { label: '1–4 Oct',  dates: '1 Oct – 4 Oct 2026'  },
              ].map((r) => (
                <div key={r.label} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                  <div>
                    <p className="text-xs font-medium text-gray-800">{r.label}</p>
                    <p className="text-xs text-gray-400">{r.dates}</p>
                  </div>
                  <button type="button" className="text-xs text-[#1f4fd1] hover:underline flex items-center gap-0.5">
                    View full report <ChevronRight size={12} />
                  </button>
                </div>
              ))}
            </div>

          </div>
        </div>
      </div>
    </AppLayout>
  );
}
