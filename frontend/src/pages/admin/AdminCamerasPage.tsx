import { useState } from 'react';
import { Plus } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { EmptyState, PageHeader } from '../../components/ui/Primitives';
import { MOCK_CAMERAS } from '../../data/mock';
import type { Camera } from '../../types';
import AddCameraModal from '../../components/modals/AddCameraModal';
import CameraDetailDrawer from '../../components/drawers/CameraDetailDrawer';
import { CameraCard } from '../staff/CamerasPage';

const LOCATIONS = Array.from(new Set(MOCK_CAMERAS.map((c) => c.location)));

export default function AdminCamerasPage() {
  const [locationFilter, setLocationFilter] = useState('all');
  const [showAddCamera, setShowAddCamera] = useState(false);
  const [selectedCam, setSelectedCam] = useState<Camera | null>(null);

  const filtered = locationFilter === 'all' ? MOCK_CAMERAS : MOCK_CAMERAS.filter((c) => c.location === locationFilter);
  const online = MOCK_CAMERAS.filter((c) => c.status === 'online').length;

  return (
    <AppLayout role="admin" activePath="/admin/cameras">
      <PageHeader
        title="Cameras"
        subtitle={`${online} of ${MOCK_CAMERAS.length} online · select a camera to edit its rules`}
        actions={
          <>
            <FilterSelect
              label="Location"
              value={locationFilter}
              onChange={setLocationFilter}
              options={[{ value: 'all', label: 'All locations' }, ...LOCATIONS.map((l) => ({ value: l, label: l }))]}
            />
            <Button variant="primary" icon={<Plus size={16} />} onClick={() => setShowAddCamera(true)}>Add camera</Button>
          </>
        }
      />

      <div className="grid grid-cols-1 gap-[20px] sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((cam, idx) => <CameraCard key={cam.id} cam={cam} seed={idx} onSelect={setSelectedCam} />)}
      </div>

      {filtered.length === 0 && <EmptyState>No cameras found for this location.</EmptyState>}

      <AddCameraModal isOpen={showAddCamera} onClose={() => setShowAddCamera(false)} onAdd={() => setShowAddCamera(false)} />
      <CameraDetailDrawer cam={selectedCam} onClose={() => setSelectedCam(null)} />
    </AppLayout>
  );
}
