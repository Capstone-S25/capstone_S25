import { useState } from 'react';
import { Incident } from '../../types/index';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { SeverityBadge } from '../ui/Badge';

interface DismissModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
  onConfirm: (reason: string) => void;
}

const DISMISS_REASONS = [
  'False alarm — no actual violation',
  'Person left before action needed',
  'Known authorized activity',
  'Other',
];

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

export default function DismissModal({ isOpen, onClose, incident, onConfirm }: DismissModalProps) {
  const [reason, setReason] = useState('');
  const [notes, setNotes] = useState('');

  const handleConfirm = () => {
    if (!reason) return;
    onConfirm(reason);
    setReason('');
    setNotes('');
    onClose();
  };

  const handleClose = () => {
    setReason('');
    setNotes('');
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Dismiss incident" width="460px">
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

          {/* Reason select */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-gray-700">
              Reason <span className="text-red-500">*</span>
            </label>
            <select
              className={INPUT_CLASS}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            >
              <option value="">Select a reason…</option>
              {DISMISS_REASONS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {/* Optional notes */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium text-gray-700">
              Notes <span className="text-gray-400 font-normal">(optional)</span>
            </label>
            <textarea
              className={INPUT_CLASS}
              rows={2}
              placeholder="Any additional context…"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>

          {/* Footer buttons */}
          <div className="flex gap-3 pt-1">
            <Button variant="secondary" className="flex-1" onClick={handleClose}>
              Cancel
            </Button>
            <Button
              className="flex-1 text-white border-transparent"
              style={{ backgroundColor: '#374151' }}
              onClick={handleConfirm}
              disabled={!reason}
            >
              Dismiss
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
}
