import { useState } from 'react';
import { Plus } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import { MOCK_CAMERAS, MOCK_INCIDENTS, MOCK_RULES } from '../../data/mock';
import type { Camera } from '../../types';
import AddCameraModal from '../../components/modals/AddCameraModal';
import CameraDetailDrawer from '../../components/drawers/CameraDetailDrawer';

// ── Helpers ────────────────────────────────────────────────────────────────
function todayIncidentCount(camId: string): number {
  const today = new Date().toISOString().slice(0, 10);
  return MOCK_INCIDENTS.filter(
    (i) => i.camId === camId && i.dateTime.startsWith(today),
  ).length;
}

function cameraRuleSummary(cam: Camera): string {
  const types = cam.rules
    .map((rid) => MOCK_RULES.find((r) => r.id === rid)?.detectType)
    .filter(Boolean) as string[];
  const unique = Array.from(new Set(types));
  return unique.length > 0 ? `Looks for: ${unique.join(', ')}` : 'No active rules';
}

const ALL_LOCATIONS = ['All locations', ...Array.from(new Set(MOCK_CAMERAS.map((c) => c.location)))];

// ── Camera card ────────────────────────────────────────────────────────────
function CameraCard({ cam, onSelect }: { cam: Camera; onSelect: (cam: Camera) => void }) {
  const online        = cam.status === 'online';
  const incidentCount = todayIncidentCount(cam.id);
  const ruleSummary   = cameraRuleSummary(cam);
  const label         = `${cam.name} · ${cam.location}`;

  return (
    <div
      className="bg-white rounded-2xl overflow-hidden cursor-pointer"
      onClick={() => onSelect(cam)}
      style={{ transition: 'box-shadow 0.15s' }}
      onMouseEnter={(e) => (e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.10)')}
      onMouseLeave={(e) => (e.currentTarget.style.boxShadow = 'none')}
    >

      {/* Thumbnail */}
      <div className="relative w-full aspect-video bg-gray-200 flex items-center justify-center">

        {/* Offline overlay */}
        {!online && (
          <div className="absolute inset-0 bg-black/40 z-10" />
        )}

        {/* Placeholder icon */}
        <span className="text-gray-400 text-xs relative z-0">No feed</span>

        {/* Top-left: cam label */}
        <div
          className="absolute top-2 left-2 z-20 rounded px-2 py-0.5"
          style={{ backgroundColor: 'rgba(0,0,0,0.6)' }}
        >
          <span className="text-white text-xs font-medium">{label}</span>
        </div>

        {/* Top-right: status badge */}
        <div className="absolute top-2 right-2 z-20">
          {online
            ? (
              <span className="text-xs font-semibold text-white bg-green-500 rounded-full px-2 py-0.5">
                Today
              </span>
            )
            : (
              <span className="text-xs font-semibold text-white bg-red-500 rounded-full px-2 py-0.5">
                Offline
              </span>
            )}
        </div>
      </div>

      {/* Below thumbnail */}
      <div className="px-3 py-3">
        <p className="text-sm font-bold text-gray-900 mb-1 truncate">{label}</p>

        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          {/* Incident count pill */}
          <span className="text-xs font-medium rounded-full px-2 py-0.5" style={{ color: '#1f4fd1', border: '1px solid #c7d7f8' }}>
            {incidentCount} Today
          </span>
        </div>

        {/* Rules summary */}
        <p className="text-xs text-gray-400 leading-snug truncate">{ruleSummary}</p>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function AdminCamerasPage() {
  const [locationFilter, setLocationFilter] = useState('All locations');
  const [showAddCamera, setShowAddCamera] = useState(false);
  const [selectedCam, setSelectedCam] = useState<Camera | null>(null);

  const filtered = locationFilter === 'All locations'
    ? MOCK_CAMERAS
    : MOCK_CAMERAS.filter((c) => c.location === locationFilter);

  return (
    <AppLayout role="admin" activePath="/admin/cameras">
      <div style={{ padding: '20px 28px 20px 20px' }}>

        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-gray-900">Cameras</h1>
            <select
              value={locationFilter}
              onChange={(e) => setLocationFilter(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-[#1f4fd1]"
            >
              {ALL_LOCATIONS.map((loc) => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>

          <Button variant="primary" onClick={() => setShowAddCamera(true)}>
            <Plus size={15} />
            Add camera
          </Button>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-4 gap-4">
          {filtered.map((cam) => (
            <CameraCard key={cam.id} cam={cam} onSelect={setSelectedCam} />
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-16 text-gray-400 text-sm">
            No cameras found for this location.
          </div>
        )}
      </div>

      <AddCameraModal
        isOpen={showAddCamera}
        onClose={() => setShowAddCamera(false)}
        onAdd={() => setShowAddCamera(false)}
      />

      <CameraDetailDrawer cam={selectedCam} onClose={() => setSelectedCam(null)} />
    </AppLayout>
  );
}
