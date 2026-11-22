import { useState } from 'react';
import type { MonitoringRule, Severity } from '../../types/index';
import { MOCK_CAMERAS } from '../../data/mock';
import { RULE_TYPES, SEVERITIES, type RuleType } from '../../data/rules';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';

interface AddRuleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (rule: Omit<MonitoringRule, 'id'>) => void;
}

export default function AddRuleModal({ isOpen, onClose, onAdd }: AddRuleModalProps) {
  const [camId, setCamId] = useState('');
  const [detectType, setDetectType] = useState<RuleType | ''>('');
  const [activeFrom, setActiveFrom] = useState('00:00');
  const [activeTo, setActiveTo] = useState('23:59');
  const [severity, setSeverity] = useState<Severity | ''>('');
  const [status, setStatus] = useState<'On' | 'Off'>('On');

  const camera = MOCK_CAMERAS.find((c) => c.id === camId);
  const isValid = Boolean(camera && detectType && severity && activeFrom && activeTo);

  const handleClose = () => {
    setCamId('');
    setDetectType('');
    setActiveFrom('00:00');
    setActiveTo('23:59');
    setSeverity('');
    setStatus('On');
    onClose();
  };

  const handleAdd = () => {
    if (!camera || !detectType || !severity) return;
    onAdd({
      location: camera.location,
      cameraIds: [camera.id],
      detectType,
      activeFrom,
      activeTo,
      severity,
      minConfidence: 0.7,
      enabled: status === 'On',
    });
    handleClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Add monitoring rule" width="520px"
      subtitle="The AI will flag matching behaviour on these cameras."
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>Cancel</Button>
          <Button variant="primary" onClick={handleAdd} disabled={!isValid}>Add rule</Button>
        </>
      }
    >
      <div className="flex flex-col gap-[16px]">
        <div className="flex gap-[12px]">
          <div className="flex-1">
            <label htmlFor="rule-camera" className={LABEL_CLASS}>Location</label>
            <select id="rule-camera" className={INPUT_CLASS} value={camId} onChange={(e) => setCamId(e.target.value)}>
              <option value="">Select…</option>
              {MOCK_CAMERAS.map((c) => <option key={c.id} value={c.id}>{c.location} · {c.name}</option>)}
            </select>
          </div>
          <div className="flex-1">
            <label htmlFor="rule-detect" className={LABEL_CLASS}>Detect</label>
            <select id="rule-detect" className={INPUT_CLASS} value={detectType} onChange={(e) => setDetectType(e.target.value as RuleType)}>
              <option value="">Select type…</option>
              {RULE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div className="flex gap-[12px]">
          <div className="flex-1">
            <span className={LABEL_CLASS}>Active hours</span>
            <div className="flex items-center gap-[8px]">
              <input type="time" aria-label="Active from" className={INPUT_CLASS} value={activeFrom} onChange={(e) => setActiveFrom(e.target.value)} />
              <span className="shrink-0 text-[13px] text-faint">–</span>
              <input type="time" aria-label="Active until" className={INPUT_CLASS} value={activeTo} onChange={(e) => setActiveTo(e.target.value)} />
            </div>
          </div>
        </div>

        <div className="flex gap-[12px]">
          <div className="flex-1">
            <label htmlFor="rule-severity" className={LABEL_CLASS}>Severity</label>
            <select id="rule-severity" className={INPUT_CLASS} value={severity} onChange={(e) => setSeverity(e.target.value as Severity)}>
              <option value="">Select…</option>
              {SEVERITIES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="flex-1">
            <label htmlFor="rule-status" className={LABEL_CLASS}>Status</label>
            <select id="rule-status" className={INPUT_CLASS} value={status} onChange={(e) => setStatus(e.target.value as 'On' | 'Off')}>
              <option value="On">On</option>
              <option value="Off">Off</option>
            </select>
          </div>
        </div>
      </div>
    </Modal>
  );
}
