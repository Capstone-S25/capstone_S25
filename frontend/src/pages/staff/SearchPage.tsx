import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AppLayout } from '../../components/layout/AppLayout';
import { OutcomeDot } from '../../components/ui/Badge';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { ClipId, ClipThumb, EmptyState, PageHeader } from '../../components/ui/Primitives';
import ClipDetailsModal from '../../components/modals/ClipDetailsModal';
import { useIncidents } from '../../data/useIncidents';
import { clipLabel, matchesClipId } from '../../data/clipId';
import { RULE_TYPES } from '../../data/rules';
import type { Incident } from '../../types';

const LOCATIONS = ['Carpark B2', 'Lift Lobby 1', 'Corridor 2', 'Main entrance', 'Meeting Room 1', 'Staircase B', 'Pantry L3'];

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
}

export default function SearchPage() {
  const { incidents } = useIncidents();
  const [params] = useSearchParams();
  const query = params.get('q') ?? '';
  const [typeFilter, setTypeFilter] = useState('all');
  const [locationFilter, setLocationFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('7d');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

  const q = query.trim().toLowerCase();
  const results = incidents.filter((inc) => {
    if (severityFilter !== 'all' && inc.severity !== severityFilter) return false;
    if (typeFilter !== 'all' && inc.type !== typeFilter) return false;
    if (locationFilter !== 'all' && inc.location !== locationFilter) return false;
    if (q) {
      const matches =
        inc.type.toLowerCase().includes(q) ||
        inc.location.toLowerCase().includes(q) ||
        inc.description.toLowerCase().includes(q) ||
        inc.camId.toLowerCase().includes(q) ||
        clipLabel(inc.id).includes(q) ||
        matchesClipId(inc.id, q);
      if (!matches) return false;
    }
    return true;
  });

  const count = `${results.length} clip${results.length === 1 ? '' : 's'}`;

  return (
    <AppLayout role="staff" activePath="/staff/search">
      <PageHeader
        title={query.trim() ? <>Results for “{query.trim()}”</> : 'All clips'}
        subtitle={query.trim() ? `${count} match your search` : `${count} · use the search bar above to narrow down`}
      />

      <div className="-mt-[4px] flex flex-wrap items-center gap-[8px]">
        <FilterSelect
          label="Type"
          value={typeFilter}
          onChange={setTypeFilter}
          options={[{ value: 'all', label: 'All types' }, ...RULE_TYPES.map((t) => ({ value: t, label: t }))]}
        />
        <FilterSelect
          label="Location"
          value={locationFilter}
          onChange={setLocationFilter}
          options={[{ value: 'all', label: 'All locations' }, ...LOCATIONS.map((l) => ({ value: l, label: l }))]}
        />
        <FilterSelect
          label="Date"
          value={dateFilter}
          onChange={setDateFilter}
          options={[{ value: 'today', label: 'Today' }, { value: '7d', label: 'Last 7 days' }, { value: '30d', label: 'Last 30 days' }, { value: 'all', label: 'Any date' }]}
        />
        <FilterSelect
          label="Severity"
          value={severityFilter}
          onChange={setSeverityFilter}
          options={[{ value: 'all', label: 'All' }, { value: 'Critical', label: 'Critical' }, { value: 'Moderate', label: 'Moderate' }, { value: 'Minor', label: 'Minor' }]}
        />
      </div>

      <div className="grid grid-cols-1 gap-x-[20px] gap-y-[20px] sm:grid-cols-2 lg:grid-cols-3">
        {results.map((inc, idx) => (
          <button
            key={inc.id}
            type="button"
            onClick={() => setSelectedIncident(inc)}
            className="group flex min-w-0 cursor-pointer flex-col gap-[10px] border-none bg-transparent p-0 text-left"
          >
            <ClipThumb
              seed={idx}
              src={inc.thumbnailUrl}
              className="aspect-[16/9] w-full rounded-[6px] outline-2 outline-offset-2 outline-transparent transition-[outline-color] group-hover:outline-brand-mid"
              iconSize={28}
            >
              <ClipId id={inc.id} variant="overlay" className="absolute top-[8px] left-[8px]" />
              {inc.duration && (
                <span className="absolute right-[8px] bottom-[8px] rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[6px] py-[2px] text-[11px] font-medium text-white">
                  {inc.duration}
                </span>
              )}
            </ClipThumb>
            <span className="flex w-full flex-col gap-[3px]">
              <span className="truncate text-[13px] font-medium text-ink">{inc.description}</span>
              <span className="flex w-full items-center gap-[8px]">
                <span className="truncate text-[12px] text-muted">{formatDate(inc.dateTime)} · {inc.location}</span>
                <span className="flex-1" />
                <OutcomeDot outcome={inc.outcome} />
              </span>
            </span>
          </button>
        ))}
      </div>

      {results.length === 0 && <EmptyState>No clips match your search. Try a different place or incident type.</EmptyState>}

      <ClipDetailsModal isOpen={selectedIncident !== null} onClose={() => setSelectedIncident(null)} incident={selectedIncident} />
    </AppLayout>
  );
}
