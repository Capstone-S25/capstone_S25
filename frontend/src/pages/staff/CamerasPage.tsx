import { useState } from 'react';
import { AppLayout } from '../../components/layout/AppLayout';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { ClipThumb, EmptyState, PageHeader } from '../../components/ui/Primitives';
import { MOCK_CAMERAS, MOCK_INCIDENTS } from '../../data/mock';
import type { Camera } from '../../types';
import CameraDetailDrawer from '../../components/drawers/CameraDetailDrawer';

// Camera rules lookup (mocked — in real app would join via MOCK_RULES)
const CAMERA_RULE_LABELS: Record<string, string[]> = {
  'cam-01': ['Smoking', 'Littering'],
  'cam-02': ['Loitering'],
  'cam-03': ['Unattended item'],
  'cam-04': ['Unattended item'],
  'cam-05': ['Unattended item'],
  'cam-06': ['Smoking'],
  'cam-07': ['Littering'],
  'cam-09': ['Unattended item', 'Smoking'],
};

const DISPLAY_CAMERAS = MOCK_CAMERAS.slice(0, 8);

function formatTime(iso?: string) {
  if (!iso) return '';
  return new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

export function CameraCard({ cam, seed, onSelect }: { cam: Camera; seed: number; onSelect: (c: Camera) => void }) {
  const isOffline = cam.status === 'offline';
  const rules = CAMERA_RULE_LABELS[cam.id] ?? [];
  const today = MOCK_INCIDENTS.filter((i) => i.camId === cam.id).length;
  return (
    <button
      type="button"
      onClick={() => onSelect(cam)}
      className="group flex min-w-0 cursor-pointer flex-col gap-[10px] border-none bg-transparent p-0 text-left"
    >
      <ClipThumb
        seed={seed}
        offline={isOffline}
        offlineLabel={isOffline ? `No signal since ${formatTime(cam.lastSeen) || 'earlier'}` : undefined}
        className="h-[180px] w-full rounded-[6px] outline-2 outline-offset-2 outline-transparent transition-[outline-color] group-hover:outline-brand-mid"
        iconSize={28}
      />
      <span className="flex w-full flex-col gap-[3px]">
        <span className="flex w-full items-center gap-[8px]">
          <span className="size-[7px] shrink-0 rounded-full" style={{ backgroundColor: isOffline ? 'var(--color-line)' : 'var(--color-success)' }} />
          <span className="truncate text-[13px] font-medium text-ink">{cam.name} · {cam.location}</span>
          <span className="flex-1" />
          <span className="shrink-0 text-[12px] text-muted">{isOffline ? 'Offline' : `${today} today`}</span>
        </span>
        <span className="truncate text-[12px] text-muted">
          {rules.length ? `Looks for: ${rules.join(', ')}` : 'No active rules'}
        </span>
      </span>
    </button>
  );
}

export default function CamerasPage() {
  const [locationFilter, setLocationFilter] = useState('all');
  const [selectedCam, setSelectedCam] = useState<Camera | null>(null);

  const filtered = DISPLAY_CAMERAS.filter((cam) => locationFilter === 'all' || cam.location === locationFilter);
  const uniqueLocations = Array.from(new Set(DISPLAY_CAMERAS.map((c) => c.location)));
  const online = DISPLAY_CAMERAS.filter((c) => c.status === 'online').length;
  const offline = DISPLAY_CAMERAS.length - online;

  return (
    <AppLayout role="staff" activePath="/staff/cameras">
      <PageHeader
        title="Cameras"
        subtitle={`${online} of ${DISPLAY_CAMERAS.length} online${offline ? ` · ${offline} offline` : ''}`}
        actions={
          <FilterSelect
            label="Location"
            value={locationFilter}
            onChange={setLocationFilter}
            options={[{ value: 'all', label: 'All locations' }, ...uniqueLocations.map((l) => ({ value: l, label: l }))]}
          />
        }
      />

      <div className="grid grid-cols-1 gap-[20px] sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((cam, idx) => (
          <CameraCard key={cam.id} cam={cam} seed={idx} onSelect={setSelectedCam} />
        ))}
      </div>

      {filtered.length === 0 && <EmptyState>No cameras found for the selected location.</EmptyState>}

      <CameraDetailDrawer cam={selectedCam} onClose={() => setSelectedCam(null)} />
    </AppLayout>
  );
}
