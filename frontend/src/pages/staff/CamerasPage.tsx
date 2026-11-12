import { useState } from 'react';
import { WifiOff, Camera as CameraIcon } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { MOCK_CAMERAS } from '../../data/mock';

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

// Gray shades for thumbnail variety
const THUMB_SHADES = [
  'bg-gray-200',
  'bg-gray-300',
  'bg-slate-200',
  'bg-zinc-200',
  'bg-gray-200',
  'bg-neutral-200',
  'bg-gray-300',
  'bg-slate-300',
];

// Show 8 cameras (use all MOCK_CAMERAS which has 8 entries)
const DISPLAY_CAMERAS = MOCK_CAMERAS.slice(0, 8);

export default function CamerasPage() {
  const [locationFilter, setLocationFilter] = useState('all');

  const filtered = DISPLAY_CAMERAS.filter((cam) => {
    if (locationFilter !== 'all' && cam.location !== locationFilter) return false;
    return true;
  });

  const uniqueLocations = Array.from(new Set(DISPLAY_CAMERAS.map((c) => c.location)));

  return (
    <AppLayout role="staff" activePath="/staff/cameras">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Cameras</h1>
          <select
            value={locationFilter}
            onChange={(e) => setLocationFilter(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value="all">Locations: All locations</option>
            {uniqueLocations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>
        </div>

        {/* Camera grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filtered.map((cam, idx) => {
            const isOffline = cam.status === 'offline';
            const ruleLabels = CAMERA_RULE_LABELS[cam.id] ?? [];

            return (
              <div
                key={cam.id}
                className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
              >
                {/* Thumbnail */}
                <div
                  className={`relative w-full ${THUMB_SHADES[idx % THUMB_SHADES.length]}`}
                  style={{ aspectRatio: '16/9' }}
                >
                  {/* Offline overlay */}
                  {isOffline && (
                    <div className="absolute inset-0 bg-gray-900/40 flex items-center justify-center">
                      <WifiOff size={24} className="text-white/80" />
                    </div>
                  )}

                  {/* Camera name badge — top left */}
                  <span className="absolute top-2 left-2 text-xs font-semibold text-white bg-black/60 px-2 py-0.5 rounded whitespace-nowrap max-w-[calc(100%-16px)] truncate">
                    {cam.name} · {cam.location}
                  </span>

                  {/* Status badge — top right */}
                  {isOffline ? (
                    <span className="absolute top-2 right-2 text-xs font-semibold text-white bg-red-500 px-2 py-0.5 rounded">
                      Offline
                    </span>
                  ) : (
                    <span className="absolute top-2 right-2 text-xs font-semibold text-white bg-green-500/90 px-2 py-0.5 rounded">
                      Today
                    </span>
                  )}
                </div>

                {/* Card body */}
                <div className="p-3">
                  <div className="flex items-center gap-1.5 mb-1">
                    <CameraIcon size={12} className="text-gray-400 flex-shrink-0" />
                    <p className="text-sm font-semibold text-gray-900 truncate">
                      {cam.name} · {cam.location}
                    </p>
                  </div>
                  {ruleLabels.length > 0 ? (
                    <p className="text-xs text-gray-500 leading-snug">
                      Looks for: {ruleLabels.join(', ')}
                    </p>
                  ) : (
                    <p className="text-xs text-gray-400 italic">No active rules</p>
                  )}
                  {isOffline && cam.lastSeen && (
                    <p className="text-xs text-red-500 mt-1">
                      Last seen: {new Date(cam.lastSeen).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })} {new Date(cam.lastSeen).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-16 text-gray-400 text-sm">
            No cameras found for the selected location.
          </div>
        )}
      </div>
    </AppLayout>
  );
}
