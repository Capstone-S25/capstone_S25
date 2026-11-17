import { useState } from 'react';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
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
    <Modal isOpen={isOpen} onClose={handleClose} title="Add camera" width="480px"
      subtitle="Connect a camera to start AI monitoring."
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
          Cancel
          </Button>
          <Button
          variant="primary"
          onClick={handleAdd}
          disabled={!isValid}
          >
          Add camera
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-[16px]">

        {/* Camera name + Location row */}
        <div className="flex gap-[12px]">
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
          <div className="flex flex-wrap gap-[8px]">
            {LOOKS_FOR_OPTIONS.map((option) => {
              const selected = looksFor.includes(option);
              return (
                <button
                  key={option}
                  type="button"
                  onClick={() => toggleLooksFor(option)}
                  aria-pressed={selected}
                  className={`cursor-pointer rounded-full border px-[12px] py-[6px] text-[12px] font-medium transition-colors ${
                    selected ? 'border-brand bg-brand-soft text-brand' : 'border-line bg-white text-ink-2 hover:bg-head'
                  }`}
                >
                  {option}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </Modal>
  );
}
