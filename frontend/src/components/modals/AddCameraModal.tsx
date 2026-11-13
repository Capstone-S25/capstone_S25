import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';

interface AddCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (data: { name: string; location: string; streamUrl: string; looksFor: string[] }) => void;
}

const LOCATIONS = [
  'Carpark B2',
  'Lift Lobby 1',
  'Corridor 2',
  'Main entrance',
  'Meeting Room 1',
  'Staircase B',
  'Pantry L3',
];

const LOOKS_FOR_OPTIONS = ['Smoking', 'Littering', 'Loitering', 'Unattended item'];

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

const LABEL_CLASS = 'block text-sm font-medium text-gray-700 mb-1.5';

export default function AddCameraModal({ isOpen, onClose, onAdd }: AddCameraModalProps) {
  const [name, setName] = useState('');
  const [location, setLocation] = useState('');
  const [streamUrl, setStreamUrl] = useState('');
  const [looksFor, setLooksFor] = useState<string[]>([]);

  const toggleLooksFor = (option: string) => {
    setLooksFor((prev) =>
      prev.includes(option) ? prev.filter((o) => o !== option) : [...prev, option]
    );
  };

  const handleAdd = () => {
    if (!name.trim() || !location || !streamUrl.trim()) return;
    onAdd({ name: name.trim(), location, streamUrl: streamUrl.trim(), looksFor });
    handleClose();
  };

  const handleClose = () => {
    setName('');
    setLocation('');
    setStreamUrl('');
    setLooksFor([]);
    onClose();
  };

  const isValid = Boolean(name.trim() && location && streamUrl.trim());

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Add camera" width="480px">
      <div className="space-y-4">
        {/* Subtitle */}
        <p className="text-sm text-gray-500 -mt-2">
          Connect a camera to start AI monitoring.
        </p>

        {/* Camera name + Location row */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className={LABEL_CLASS}>Camera name</label>
            <input
              type="text"
              className={INPUT_CLASS}
              placeholder="e.g. Cam 09"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="flex-1">
            <label className={LABEL_CLASS}>Location</label>
            <select
              className={INPUT_CLASS}
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            >
              <option value="">Select…</option>
              {LOCATIONS.map((loc) => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Stream address */}
        <div>
          <label className={LABEL_CLASS}>Stream address</label>
          <input
            type="text"
            className={INPUT_CLASS}
            placeholder="rtsp://..."
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
          />
        </div>

        {/* Looks for — chip multi-select */}
        <div>
          <label className={LABEL_CLASS}>Looks for</label>
          <div className="flex flex-wrap gap-2">
            {LOOKS_FOR_OPTIONS.map((option) => {
              const selected = looksFor.includes(option);
              return (
                <button
                  key={option}
                  type="button"
                  onClick={() => toggleLooksFor(option)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '20px',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    border: selected ? '1.5px solid #1f4fd1' : '1.5px solid #d0d5dd',
                    backgroundColor: selected ? '#dbe4fb' : '#ffffff',
                    color: selected ? '#1f4fd1' : '#374151',
                    transition: 'all 0.15s',
                  }}
                >
                  {option}
                </button>
              );
            })}
          </div>
        </div>

        {/* Footer buttons */}
        <div className="flex gap-3 pt-2">
          <Button variant="secondary" className="flex-1" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            className="flex-1"
            onClick={handleAdd}
            disabled={!isValid}
          >
            Add camera
          </Button>
        </div>
      </div>
    </Modal>
  );
}
