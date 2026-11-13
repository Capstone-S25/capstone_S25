import { useState } from 'react';
import { TrendingUp, ChevronRight } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';

// ── Data ──────────────────────────────────────────────────────────────────────

const BAR_DATA = [
  { name: 'Carpark B2', count: 22, pct: 100 },
  { name: 'Lift Lobby 1', count: 16, pct: 73 },
  { name: 'Corridor 2', count: 11, pct: 50 },
  { name: 'Main entrance', count: 7, pct: 32 },
  { name: 'Meeting Room 1', count: 5, pct: 23 },
  { name: 'Staircase B', count: 4, pct: 18 },
  { name: 'Pantry L3', count: 3, pct: 14 },
];

type DotLevel = 'high' | 'med' | 'low';

interface RoomCell {
  name: string;
  count: number;
  level: DotLevel;
  span?: number;
}

const FLOOR_PLAN: { label: string; rooms: RoomCell[] }[] = [
  {
    label: 'Level 1–3',
    rooms: [
      { name: 'Main entrance', count: 7, level: 'med' },
      { name: 'Pantry L3', count: 3, level: 'low' },
    ],
  },
  {
    label: '',
    rooms: [
      { name: 'Lift Lobby 1', count: 16, level: 'high' },
      { name: 'Corridor 2', count: 11, level: 'med' },
    ],
  },
  {
    label: 'Basement 2',
    rooms: [
      { name: 'Carpark B2', count: 22, level: 'high' },
      { name: 'Staircase B', count: 4, level: 'low' },
      { name: 'Meeting Room 1', count: 5, level: 'low' },
    ],
  },
];

const TIME_BARS = [
  { label: 'Morning 6AM–12PM', count: 0, pct: 0 },
  { label: 'Afternoon 12–6PM', count: 1, pct: 7 },
  { label: 'Evening 6–10PM', count: 6, pct: 40 },
  { label: 'Night 10PM–6AM', count: 15, pct: 100 },
];

const DOT_STYLE: Record<DotLevel, { bg: string; dot: string; dotLabel: string; dotColor?: string }> = {
  high: { bg: 'bg-[#eef2fb] border-[#c7d7f8]', dot: '', dotLabel: '10+', dotColor: '#1a40b0' },
  med:  { bg: 'bg-[#f4f7fe] border-[#dbe4fb]', dot: '', dotLabel: '5–9', dotColor: '#1f4fd1' },
  low:  { bg: 'bg-gray-50 border-gray-200', dot: 'bg-gray-400', dotLabel: 'Under 5', dotColor: '#9ca3af' },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function HotspotsPage() {
  const [typePeriod, setTypePeriod] = useState('all');
  const [period, setPeriod] = useState('7d');
  const [selectedPlace] = useState('Carpark B2');

  return (
    <AppLayout role="staff" activePath="/staff/hotspots">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h1 className="text-2xl font-bold text-gray-900">Hotspots</h1>
          <div className="flex items-center gap-3">
            <select
              value={typePeriod}
              onChange={(e) => setTypePeriod(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="all">Type: All types</option>
              <option value="Smoking">Type: Smoking</option>
              <option value="Loitering">Type: Loitering</option>
              <option value="Littering">Type: Littering</option>
              <option value="Unattended item">Type: Unattended item</option>
            </select>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="7d">Period: Last 7 days</option>
              <option value="30d">Period: Last 30 days</option>
              <option value="90d">Period: Last 90 days</option>
            </select>
          </div>
        </div>

        {/* Stat cards row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <StatCard
            label="Busiest place"
            main="Carpark B2"
            sub="4 more than last week, mostly smoking"
            deltaUp
          />
          <StatCard
            label="Busiest time"
            main="Evenings, 6–10 PM"
            sub="41% of incidents, up from 33% last week"
            deltaUp
          />
          <StatCard
            label="Biggest change"
            main="Lift Lobby 1"
            sub="35% more littering than last week"
            deltaUp
          />
        </div>

        {/* Main card: floor plan + bar list */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 mb-5">
          <div className="flex gap-8">
            {/* Left: floor plan */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-gray-900">Site map · Tower A</h2>
                {/* Legend */}
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#1a40b0' }} /> 10+
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#1f4fd1' }} /> 5–9
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-gray-400 inline-block" /> Under 5
                  </span>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                {FLOOR_PLAN.map((row, ri) => (
                  <div key={ri} className="flex items-stretch gap-3">
                    {/* Floor label */}
                    <div className="w-20 flex-shrink-0 flex items-center">
                      {row.label && (
                        <span className="text-xs text-gray-400 font-medium leading-tight">{row.label}</span>
                      )}
                    </div>
                    {/* Rooms */}
                    <div className="flex gap-3 flex-1 flex-wrap">
                      {row.rooms.map((room) => {
                        const style = DOT_STYLE[room.level];
                        return (
                          <div
                            key={room.name}
                            className={`flex-1 min-w-[120px] border rounded-lg p-3 cursor-pointer hover:shadow-sm transition-shadow ${style.bg}`}
                            style={{ borderColor: room.name === selectedPlace ? '#1f4fd1' : undefined }}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: style.dotColor ?? '#9ca3af' }} />
                              <span className="text-sm font-bold text-gray-900">{room.count}</span>
                            </div>
                            <p className="text-xs text-gray-700 font-medium leading-tight">{room.name}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Divider */}
            <div className="w-px bg-gray-100 self-stretch" />

            {/* Right: bar list */}
            <div className="w-56 flex-shrink-0">
              <h2 className="text-sm font-semibold text-gray-900 mb-4">Incidents by place</h2>
              <div className="flex flex-col gap-3">
                {BAR_DATA.map((item) => (
                  <div key={item.name}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-700 truncate pr-2">{item.name}</span>
                      <span className="text-xs font-semibold text-gray-900 flex-shrink-0">{item.count}</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${item.pct}%`, backgroundColor: item.pct >= 70 ? '#1a40b0' : item.pct >= 40 ? '#1f4fd1' : '#93c5fd' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom card: selected place detail */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex gap-8">
            {/* Left: detail */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <h2 className="text-sm font-semibold text-gray-900">Selected place: {selectedPlace}</h2>
              </div>
              <p className="text-sm text-gray-600 mb-3">
                Mostly smoking near staircase doors late at night. Up slightly from last week.
              </p>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-medium text-gray-700 bg-gray-100 px-2.5 py-1 rounded-full">Smoking</span>
                <span className="text-xs font-medium text-gray-700 bg-gray-100 px-2.5 py-1 rounded-full">Loitering</span>
              </div>
              <button className="text-sm font-medium flex items-center gap-1 hover:gap-2 transition-all" style={{ color: '#1f4fd1', background: 'none', border: 'none', cursor: 'pointer' }}>
                View clips <ChevronRight size={14} />
              </button>
            </div>

            {/* Divider */}
            <div className="w-px bg-gray-100 self-stretch" />

            {/* Right: when it happens */}
            <div className="w-64 flex-shrink-0">
              <h2 className="text-sm font-semibold text-gray-900 mb-4">When it happens</h2>
              <div className="flex flex-col gap-3">
                {TIME_BARS.map((tb) => (
                  <div key={tb.label}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-600">{tb.label}</span>
                      <span className="text-xs font-semibold text-gray-900">{tb.count}</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      {tb.pct > 0 && (
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${tb.pct}%`,
                            backgroundColor: tb.count >= 10 ? '#1a40b0' : '#1f4fd1',
                          }}
                        />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  main: string;
  sub: string;
  deltaUp?: boolean;
}

function StatCard({ label, main, sub, deltaUp = false }: StatCardProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs font-medium text-gray-500 mb-1">{label}</p>
      <p className="text-sm font-bold text-gray-900 mb-1">{main}</p>
      <div className="flex items-start gap-1.5">
        {deltaUp && (
          <TrendingUp size={13} className="flex-shrink-0 mt-0.5" style={{ color: '#ef4444' }} />
        )}
        <p className="text-xs text-gray-500 leading-snug">{sub}</p>
      </div>
    </div>
  );
}
