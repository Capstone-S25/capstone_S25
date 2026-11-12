import { useState } from 'react';
import { Search } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import { OutcomeBadge } from '../../components/ui/Badge';
import { MOCK_INCIDENTS } from '../../data/mock';

const CLIP_RESULTS = MOCK_INCIDENTS.slice(0, 6);

function formatCardTime(iso: string) {
  const d = new Date(iso);
  return {
    time: d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
    date: d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
    camLabel: `CAM ${d.getHours() % 9 + 1 < 10 ? '0' : ''}${d.getHours() % 9 + 1} · ${d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`,
  };
}

// Gray shades for placeholder variety
const PLACEHOLDER_GRAYS = ['bg-gray-200', 'bg-gray-300', 'bg-gray-200', 'bg-slate-200', 'bg-gray-300', 'bg-zinc-200'];

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [typeFilter, setTypeFilter] = useState('all');
  const [locationFilter, setLocationFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');

  const handleSearch = () => setSubmitted(true);

  const results = CLIP_RESULTS.filter((inc) => {
    if (severityFilter !== 'all' && inc.severity !== severityFilter) return false;
    if (typeFilter !== 'all' && inc.type !== typeFilter) return false;
    if (locationFilter !== 'all' && inc.location !== locationFilter) return false;
    return true;
  });

  return (
    <AppLayout role="staff" activePath="/staff/search">
      <div className="p-6">
        {/* Page title */}
        <h1 className="text-2xl font-bold text-gray-900 mb-5">Search</h1>

        {/* Search bar */}
        <div className="flex gap-3 mb-4">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search your incidents..."
              className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-xl text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:ring-2 focus:border-transparent"
              style={{ '--tw-ring-color': '#1f4fd1' } as React.CSSProperties}
            />
          </div>
          <button
            onClick={handleSearch}
            className="px-5 py-2.5 text-sm font-medium text-white rounded-xl transition-colors hover:opacity-90"
            style={{ backgroundColor: '#1f4fd1' }}
          >
            Search
          </button>
        </div>

        {/* Filter bar */}
        <div className="flex items-center gap-3 mb-6 flex-wrap">
          <SearchFilter value={typeFilter} onChange={setTypeFilter}>
            <option value="all">All types</option>
            <option value="Smoking">Smoking</option>
            <option value="Littering">Littering</option>
            <option value="Loitering">Loitering</option>
            <option value="Unattended item">Unattended item</option>
          </SearchFilter>
          <SearchFilter value={locationFilter} onChange={setLocationFilter}>
            <option value="all">All locations</option>
            <option value="Carpark B2">Carpark B2</option>
            <option value="Lift Lobby 1">Lift Lobby 1</option>
            <option value="Corridor 2">Corridor 2</option>
            <option value="Main entrance">Main entrance</option>
            <option value="Meeting Room 1">Meeting Room 1</option>
            <option value="Staircase B">Staircase B</option>
            <option value="Pantry L3">Pantry L3</option>
          </SearchFilter>
          <SearchFilter value={dateFilter} onChange={setDateFilter}>
            <option value="all">Any date</option>
            <option value="today">Today</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </SearchFilter>
          <SearchFilter value={severityFilter} onChange={setSeverityFilter}>
            <option value="all">All severities</option>
            <option value="Critical">Critical</option>
            <option value="Moderate">Moderate</option>
            <option value="Minor">Minor</option>
          </SearchFilter>
        </div>

        {/* Results header */}
        <p className="text-sm text-gray-500 mb-4">
          <span className="font-semibold text-gray-900">{results.length} results</span>
          {query && (
            <> for <span style={{ color: '#1f4fd1' }}>"{query}"</span></>
          )}
        </p>

        {/* Results grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.map((inc, idx) => {
            const { time, date } = formatCardTime(inc.dateTime);
            const camNum = inc.camId.replace('cam-', '').padStart(2, '0');
            const camLabel = `CAM ${camNum} · ${time}`;

            return (
              <div
                key={inc.id}
                className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
              >
                {/* Thumbnail */}
                <div className={`relative w-full ${PLACEHOLDER_GRAYS[idx % PLACEHOLDER_GRAYS.length]}`} style={{ aspectRatio: '16/9' }}>
                  {/* Camera badge top-left */}
                  <span className="absolute top-2 left-2 text-xs font-semibold text-white bg-black/60 px-2 py-0.5 rounded">
                    {camLabel}
                  </span>
                  {/* Duration badge bottom-right */}
                  {inc.duration && (
                    <span className="absolute bottom-2 right-2 text-xs font-semibold text-white bg-black/60 px-2 py-0.5 rounded">
                      {inc.duration}
                    </span>
                  )}
                </div>

                {/* Card body */}
                <div className="p-3">
                  <p className="font-semibold text-gray-900 text-sm leading-snug line-clamp-2">
                    {inc.type} — {inc.location}
                  </p>
                  <p className="text-xs text-gray-400 mt-1 mb-2">
                    {inc.location} · {date}
                  </p>
                  <OutcomeBadge outcome={inc.outcome} />
                </div>
              </div>
            );
          })}
        </div>

        {results.length === 0 && (
          <div className="text-center py-16 text-gray-400 text-sm">
            No incidents match the selected filters.
          </div>
        )}
      </div>
    </AppLayout>
  );
}

// ── Sub-component ────────────────────────────────────────────────────────────

function SearchFilter({ value, onChange, children }: { value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
    >
      {children}
    </select>
  );
}
