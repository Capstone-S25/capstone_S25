import { useState } from 'react';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';

interface InviteStaffModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInvite: (email: string, name: string) => void;
}



export default function InviteStaffModal({ isOpen, onClose, onInvite }: InviteStaffModalProps) {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('Staff');

  const handleInvite = () => {
    if (!fullName.trim() || !email.trim()) return;
    onInvite(email.trim(), fullName.trim());
    handleClose();
  };

  const handleClose = () => {
    setFullName('');
    setEmail('');
    setRole('Staff');
    onClose();
  };

  const isValid = Boolean(fullName.trim() && email.trim());

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Invite staff" width="480px"
      subtitle="They'll get an email with a temporary password and set their own when they first sign in."
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
          Cancel
          </Button>
          <Button
          variant="primary"
          onClick={handleInvite}
          disabled={!isValid}
          >
          Send invite
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-[16px]">

        {/* Name + Role row */}
        <div className="flex gap-[12px]">
          <div className="flex-1">
            <label className={LABEL_CLASS}>Name</label>
            <input
              type="text"
              className={INPUT_CLASS}
              placeholder="e.g. Jane Tan"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>
          <div className="flex-1">
            <label className={LABEL_CLASS}>Role</label>
            <select
              className={INPUT_CLASS}
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="Staff">Staff</option>
              <option value="Supervisor">Supervisor</option>
            </select>
          </div>
        </div>

        {/* Email */}
        <div>
          <label className={LABEL_CLASS}>Email</label>
          <input
            type="email"
            className={INPUT_CLASS}
            placeholder="e.g. jane.tan@company.sg"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
      </div>
    </Modal>
  );
}
