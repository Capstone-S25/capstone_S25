import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';

interface InviteStaffModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInvite: (email: string, name: string) => void;
}

const INPUT_CLASS =
  'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500';

const LABEL_CLASS = 'block text-sm font-medium text-gray-700 mb-1.5';

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
    <Modal isOpen={isOpen} onClose={handleClose} title="Invite staff" width="480px">
      <div className="space-y-4">
        {/* Subtitle */}
        <p className="text-sm text-gray-500 -mt-2">
          They'll get an email with a temporary password and set their own when they first sign in.
        </p>

        {/* Name + Role row */}
        <div className="flex gap-3">
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

        {/* Footer buttons */}
        <div className="flex gap-3 pt-2">
          <Button variant="secondary" className="flex-1" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            className="flex-1"
            onClick={handleInvite}
            disabled={!isValid}
          >
            Send invite
          </Button>
        </div>
      </div>
    </Modal>
  );
}
