import { useState } from 'react';
import { Incident } from '../../types/index';
import { MOCK_STAFF } from '../../data/mock';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';
import { IncidentPreview } from './IncidentPreview';

interface EscalateModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
  onConfirm: (remark: string) => void;
}

export default function EscalateModal({ isOpen, onClose, incident, onConfirm }: EscalateModalProps) {
  const [note, setNote] = useState('');
  const [assignTo, setAssignTo] = useState('');

  const reset = () => { setNote(''); setAssignTo(''); };
  const handleClose = () => { reset(); onClose(); };
  const handleConfirm = () => {
    if (!note.trim()) return;
    onConfirm(note.trim());
    reset();
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Escalate clip"
      subtitle="Your admin is notified straight away."
      width="460px"
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>Cancel</Button>
          <Button variant="danger-solid" onClick={handleConfirm} disabled={!note.trim()}>Escalate</Button>
        </>
      }
    >
      {incident && (
        <div className="flex flex-col gap-[16px]">
          <IncidentPreview incident={incident} />

          <div>
            <label htmlFor="escalate-note" className={LABEL_CLASS}>
              What needs attention? <span className="text-danger">*</span>
            </label>
            <textarea
              id="escalate-note"
              className={INPUT_CLASS}
              rows={3}
              placeholder="Describe what you saw and why it needs follow-up…"
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
          </div>

          <div>
            <label htmlFor="escalate-assign" className={LABEL_CLASS}>Assign to</label>
            <select id="escalate-assign" className={INPUT_CLASS} value={assignTo} onChange={(e) => setAssignTo(e.target.value)}>
              <option value="">Admin on duty</option>
              {MOCK_STAFF.map((staff) => (
                <option key={staff.id} value={staff.id}>{staff.name}</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </Modal>
  );
}
