import { useState } from 'react';
import { MonitoringRule, IncidentType, Severity } from '../../types/index';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';

interface AddRuleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (rule: Partial<MonitoringRule>) => void;
}

const LOCATION_CAMERAS = [
  { label: 'Carpark B2 · Cam 01',       location: 'Carpark B2',      camId: 'cam-01' },
  { label: 'Lift Lobby 1 · Cam 02',     location: 'Lift Lobby 1',    camId: 'cam-02' },
  { label: 'Corridor 2 · Cam 03',       location: 'Corridor 2',      camId: 'cam-03' },
  { label: 'Main entrance · Cam 04',    location: 'Main entrance',   camId: 'cam-04' },
  { label: 'Meeting Room 1 · Cam 05',   location: 'Meeting Room 1',  camId: 'cam-05' },
  { label: 'Staircase B · Cam 06',      location: 'Staircase B',     camId: 'cam-06' },
  { label: 'Pantry L3 · Cam 07',        location: 'Pantry L3',       camId: 'cam-07' },
];

const DETECT_TYPES: IncidentType[] = ['Smoking', 'Littering', 'Loitering', 'Unattended item'];
const SEVERITIES: Severity[] = ['Critical', 'Moderate', 'Minor'];

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

const LABEL_CLASS = 'block text-sm font-medium text-gray-700 mb-1.5';

export default function AddRuleModal({ isOpen, onClose, onAdd }: AddRuleModalProps) {
  const [locationCam, setLocationCam] = useState('');
  const [detectType, setDetectType] = useState<IncidentType | ''>('');
  const [activeHours, setActiveHours] = useState('00:00 – 23:59');
  const [severity, setSeverity] = useState<Severity | ''>('');
  const [minConfidence, setMinConfidence] = useState<number>(75);
  const [status, setStatus] = useState<'On' | 'Off'>('On');

  const handleAdd = () => {
    if (!locationCam || !detectType || !severity) return;
    const entry = LOCATION_CAMERAS.find((lc) => lc.camId === locationCam);
    const parts = activeHours.split(' – ');
    onAdd({
      location:      entry?.location ?? '',
      cameraIds:     entry ? [entry.camId] : [],
      detectType:    detectType as IncidentType,
      activeFrom:    parts[0]?.trim() ?? '00:00',
      activeTo:      parts[1]?.trim() ?? '23:59',
      severity:      severity as Severity,
      minConfidence: minConfidence / 100,
      enabled:       status === 'On',
    });
    handleClose();
  };

  const handleClose = () => {
    setLocationCam('');
    setDetectType('');
    setActiveHours('00:00 – 23:59');
    setSeverity('');
    setMinConfidence(75);
    setStatus('On');
    onClose();
  };

  const isValid = Boolean(locationCam && detectType && severity);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Add monitoring rule" width="520px">
      <div className="space-y-4">
        {/* Subtitle */}
        <p className="text-sm text-gray-500 -mt-2">
          The AI will flag matching behaviour on these cameras.
        </p>

        {/* Location + Detect row */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className={LABEL_CLASS}>Location</label>
            <select
              className={INPUT_CLASS}
              value={locationCam}
              onChange={(e) => setLocationCam(e.target.value)}
            >
              <option value="">Select…</option>
              {LOCATION_CAMERAS.map((lc) => (
                <option key={lc.camId} value={lc.camId}>{lc.label}</option>
              ))}
            </select>
          </div>
          <div className="flex-1">
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
        </div>

        {/* Active hours + Severity row */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className={LABEL_CLASS}>Active hours</label>
            <input
              type="text"
              className={INPUT_CLASS}
              placeholder="e.g. 22:00 – 06:00"
              value={activeHours}
              onChange={(e) => setActiveHours(e.target.value)}
            />
          </div>
          <div className="flex-1">
            <label className={LABEL_CLASS}>Severity</label>
            <select
              className={INPUT_CLASS}
              value={severity}
              onChange={(e) => setSeverity(e.target.value as Severity)}
            >
              <option value="">Select…</option>
              {SEVERITIES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Min. confidence + Status row */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className={LABEL_CLASS}>Min. confidence</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                className={INPUT_CLASS}
                min={0}
                max={100}
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
              />
              <span className="text-sm text-gray-500 flex-shrink-0">%</span>
            </div>
          </div>
          <div className="flex-1">
            <label className={LABEL_CLASS}>Status</label>
            <select
              className={INPUT_CLASS}
              value={status}
              onChange={(e) => setStatus(e.target.value as 'On' | 'Off')}
            >
              <option value="On">On</option>
              <option value="Off">Off</option>
            </select>
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
