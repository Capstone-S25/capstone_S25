import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { MOCK_RULES } from '../../data/mock';

interface AddCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (data: { name: string; location: string; rtspUrl: string }) => void;
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

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

const LABEL_CLASS = 'block text-sm font-medium text-gray-700 mb-1.5';

export default function AddCameraModal({ isOpen, onClose, onAdd }: AddCameraModalProps) {
  const [name, setName] = useState('');
  const [location, setLocation] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [selectedRules, setSelectedRules] = useState<string[]>([]);

  const toggleRule = (ruleId: string) => {
    setSelectedRules((prev) =>
      prev.includes(ruleId) ? prev.filter((id) => id !== ruleId) : [...prev, ruleId]
    );
  };

  const handleAdd = () => {
    if (!name.trim() || !location || !rtspUrl.trim()) return;
    onAdd({ name: name.trim(), location, rtspUrl: rtspUrl.trim() });
    handleClose();
  };

  const handleClose = () => {
    setName('');
    setLocation('');
    setRtspUrl('');
    setSelectedRules([]);
    onClose();
  };

  const isValid = Boolean(name.trim() && location && rtspUrl.trim());

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="+ Add camera" width="480px">
      <div className="space-y-4">
        {/* Camera name */}
        <div>
          <label className={LABEL_CLASS}>Camera name</label>
          <input
            type="text"
            className={INPUT_CLASS}
            placeholder="e.g. Cam 09"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {/* Location */}
        <div>
          <label className={LABEL_CLASS}>Location</label>
          <select
            className={INPUT_CLASS}
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          >
            <option value="">Select location…</option>
            {LOCATIONS.map((loc) => (
              <option key={loc} value={loc}>{loc}</option>
            ))}
          </select>
        </div>

        {/* RTSP URL */}
        <div>
          <label className={LABEL_CLASS}>RTSP stream URL</label>
          <input
            type="text"
            className={INPUT_CLASS}
            placeholder="rtsp://..."
            value={rtspUrl}
            onChange={(e) => setRtspUrl(e.target.value)}
          />
        </div>

        {/* Assign rules */}
        <div>
          <label className={LABEL_CLASS}>Assign rules</label>
          <div className="border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-48 overflow-y-auto">
            {MOCK_RULES.map((rule) => (
              <label
                key={rule.id}
                className="flex items-center gap-3 px-3 py-2.5 cursor-pointer hover:bg-gray-50 transition-colors"
              >
                <input
                  type="checkbox"
                  className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  checked={selectedRules.includes(rule.id)}
                  onChange={() => toggleRule(rule.id)}
                />
                <div className="min-w-0">
                  <p className="text-sm text-gray-800 font-medium truncate">
                    {rule.detectType} — {rule.location}
                  </p>
                  <p className="text-xs text-gray-400 truncate">
                    {rule.activeFrom}–{rule.activeTo} · {rule.severity}
                  </p>
                </div>
              </label>
            ))}
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
