import { useState } from 'react';
import { Incident } from '../../types/index';
import { MOCK_STAFF } from '../../data/mock';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { SeverityBadge } from '../ui/Badge';

interface EscalateModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
  onConfirm: (remark: string) => void;
}

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

export default function EscalateModal({ isOpen, onClose, incident, onConfirm }: EscalateModalProps) {
  const [note, setNote] = useState('');
  const [assignTo, setAssignTo] = useState('');

  const handleConfirm = () => {
    if (!note.trim()) return;
    onConfirm(note.trim());
    setNote('');
    setAssignTo('');
    onClose();
  };

  const handleClose = () => {
    setNote('');
    setAssignTo('');
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Escalate incident" width="460px">
      {incident && (
        <div className="space-y-5">
          {/* Incident preview */}
          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-200">
            <div className="w-16 h-12 rounded-lg bg-gray-200 flex-shrink-0 flex items-center justify-center overflow-hidden">
              {incident.thumbnailUrl ? (
                <img
                  src={incident.thumbnailUrl}
                  alt="Incident thumbnail"
                  className="w-full h-full object-cover"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
              ) : (
                <svg className="text-gray-400" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="2" y="3" width="20" height="14" rx="2" />
                  <path d="M8 21h8M12 17v4" />
                </svg>
              )}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <SeverityBadge severity={incident.severity} />
              </div>
              <p className="text-sm font-semibold text-gray-900 truncate">{incident.type}</p>
              <p className="text-xs text-gray-500 truncate">{incident.location}</p>
            </div>
          </div>

          {/* Note textarea */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-gray-700">
              Add a note <span className="text-red-500">*</span>
              <span className="text-gray-400 font-normal ml-1">(required)</span>
            </label>
            <textarea
              className={INPUT_CLASS}
              rows={3}
              placeholder="Describe what you observed or why this needs escalation..."
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
          </div>

          {/* Assign to dropdown */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-gray-700">Assign to</label>
            <select
              className={INPUT_CLASS}
              value={assignTo}
              onChange={(e) => setAssignTo(e.target.value)}
            >
              <option value="">Select staff member…</option>
              {MOCK_STAFF.map((staff) => (
                <option key={staff.id} value={staff.id}>
                  {staff.name}
                </option>
              ))}
            </select>
          </div>

          {/* Footer buttons */}
          <div className="flex gap-3 pt-1">
            <Button variant="secondary" className="flex-1" onClick={handleClose}>
              Cancel
            </Button>
            <Button
              className="flex-1 text-white border-transparent"
              style={{ backgroundColor: '#dc2626' }}
              onClick={handleConfirm}
              disabled={!note.trim()}
            >
              Escalate
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
}
