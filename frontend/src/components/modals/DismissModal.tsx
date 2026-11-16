import { useState } from 'react';
import { Incident } from '../../types/index';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';
import { IncidentPreview } from './IncidentPreview';

interface DismissModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
  onConfirm: (reason: string) => void;
}

const DISMISS_REASONS = [
  'False alarm — no actual violation',
  'Person left before action needed',
  'Known authorised activity',
  'Other',
];

export default function DismissModal({ isOpen, onClose, incident, onConfirm }: DismissModalProps) {
  const [reason, setReason] = useState('');
  const [notes, setNotes] = useState('');

  const reset = () => { setReason(''); setNotes(''); };
  const handleClose = () => { reset(); onClose(); };
  const handleConfirm = () => {
    if (!reason) return;
    onConfirm(reason);
    reset();
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Dismiss clip"
      subtitle="It will move to Review history as dismissed."
      width="460px"
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>Cancel</Button>
          <Button variant="primary" onClick={handleConfirm} disabled={!reason}>Dismiss clip</Button>
        </>
      }
    >
      {incident && (
        <div className="flex flex-col gap-[16px]">
          <IncidentPreview incident={incident} />

          <fieldset className="m-0 border-none p-0">
            <legend className={LABEL_CLASS}>Reason <span className="text-danger">*</span></legend>
            <div className="flex flex-col gap-[6px]">
              {DISMISS_REASONS.map((r) => (
                <label
                  key={r}
                  className={`flex cursor-pointer items-center gap-[10px] rounded-[6px] border px-[12px] py-[9px] text-[13px] transition-colors ${
                    reason === r ? 'border-brand bg-brand-soft text-ink' : 'border-line text-ink-2 hover:bg-head'
                  }`}
                >
                  <input
                    type="radio"
                    name="dismiss-reason"
                    value={r}
                    checked={reason === r}
                    onChange={() => setReason(r)}
                    className="size-[14px] accent-[var(--color-brand)]"
                  />
                  {r}
                </label>
              ))}
            </div>
          </fieldset>

          <div>
            <label htmlFor="dismiss-notes" className={LABEL_CLASS}>
              Notes <span className="font-normal text-faint">(optional)</span>
            </label>
            <textarea id="dismiss-notes" className={INPUT_CLASS} rows={2} placeholder="Any additional context…" value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
        </div>
      )}
    </Modal>
  );
}
