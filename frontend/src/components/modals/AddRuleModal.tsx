import { useState } from 'react';
import { Clock } from 'lucide-react';
import { MonitoringRule, IncidentType, Severity } from '../../types/index';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Toggle } from '../ui/Toggle';

interface AddRuleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (rule: Partial<MonitoringRule>) => void;
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

const DETECT_TYPES: IncidentType[] = ['Smoking', 'Littering', 'Loitering', 'Unattended item'];
const SEVERITIES: Severity[] = ['Critical', 'Moderate', 'Minor'];

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

const LABEL_CLASS = 'block text-sm font-medium text-gray-700 mb-1.5';

export default function AddRuleModal({ isOpen, onClose, onAdd }: AddRuleModalProps) {
  const [location, setLocation] = useState('');
  const [cameras, setCameras] = useState('');
  const [detectType, setDetectType] = useState<IncidentType | ''>('');
  const [activeFrom, setActiveFrom] = useState('00:00');
  const [activeTo, setActiveTo] = useState('23:59');
  const [severity, setSeverity] = useState<Severity | ''>('');
  const [minConfidence, setMinConfidence] = useState<number>(75);
  const [enabled, setEnabled] = useState(true);

  const handleAdd = () => {
    if (!location || !detectType || !severity) return;
    const cameraIds = cameras
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    onAdd({
      location,
      cameraIds,
      detectType: detectType as IncidentType,
      activeFrom,
      activeTo,
      severity: severity as Severity,
      minConfidence: minConfidence / 100,
      enabled,
    });
    handleClose();
  };

  const handleClose = () => {
    setLocation('');
    setCameras('');
    setDetectType('');
    setActiveFrom('00:00');
    setActiveTo('23:59');
    setSeverity('');
    setMinConfidence(75);
    setEnabled(true);
    onClose();
  };

  const isValid = Boolean(location && detectType && severity);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="+ Add rule" width="480px">
      <div className="space-y-4">
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

        {/* Camera(s) */}
        <div>
          <label className={LABEL_CLASS}>Camera(s)</label>
          <input
            type="text"
            className={INPUT_CLASS}
            placeholder="e.g. cam-01, cam-02"
            value={cameras}
            onChange={(e) => setCameras(e.target.value)}
          />
          <p className="mt-1 text-xs text-gray-400">Separate multiple IDs with commas.</p>
        </div>

        {/* Detect type */}
        <div>
          <label className={LABEL_CLASS}>Detect</label>
          <select
            className={INPUT_CLASS}
            value={detectType}
            onChange={(e) => setDetectType(e.target.value as IncidentType)}
          >
            <option value="">Select type…</option>
            {DETECT_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Active hours */}
        <div>
          <label className={LABEL_CLASS}>Active hours</label>
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Clock
                size={14}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
              />
              <input
                type="time"
                className={`${INPUT_CLASS} pl-8`}
                value={activeFrom}
                onChange={(e) => setActiveFrom(e.target.value)}
              />
            </div>
            <span className="text-sm text-gray-400 flex-shrink-0">to</span>
            <div className="relative flex-1">
              <Clock
                size={14}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
              />
              <input
                type="time"
                className={`${INPUT_CLASS} pl-8`}
                value={activeTo}
                onChange={(e) => setActiveTo(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Severity */}
        <div>
          <label className={LABEL_CLASS}>Severity</label>
          <select
            className={INPUT_CLASS}
            value={severity}
            onChange={(e) => setSeverity(e.target.value as Severity)}
          >
            <option value="">Select severity…</option>
            {SEVERITIES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Min. confidence */}
        <div>
          <label className={LABEL_CLASS}>Min. confidence</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              className={`${INPUT_CLASS} flex-1`}
              min={0}
              max={100}
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
            />
            <span className="text-sm text-gray-500 flex-shrink-0">%</span>
          </div>
        </div>

        {/* Status toggle */}
        <div className="flex items-center justify-between py-1">
          <label className="text-sm font-medium text-gray-700">Status</label>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">{enabled ? 'Enabled' : 'Disabled'}</span>
            <Toggle checked={enabled} onChange={setEnabled} label="Rule status" />
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
            Add rule
          </Button>
        </div>
      </div>
    </Modal>
  );
}
