import { useState } from 'react';
import { Info } from 'lucide-react';
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
    <Modal isOpen={isOpen} onClose={handleClose} title="Invite staff member" width="480px">
      <div className="space-y-4">
        {/* Subtitle */}
        <p className="text-sm text-gray-500 -mt-2">
          They'll receive an email with a temporary password to set up their account.
        </p>

        {/* Full name */}
        <div>
          <label className={LABEL_CLASS}>Full name</label>
          <input
            type="text"
            className={INPUT_CLASS}
            placeholder="e.g. Jane Tan"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
          />
        </div>

        {/* Work email */}
        <div>
          <label className={LABEL_CLASS}>Work email</label>
          <input
            type="email"
            className={INPUT_CLASS}
            placeholder="e.g. jane.tan@company.sg"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        {/* Role */}
        <div>
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

        {/* Info box */}
        <div
          className="flex items-start gap-3 rounded-xl p-3.5"
          style={{ backgroundColor: '#eef2fb', border: '1px solid #c7d7f8' }}
        >
          <Info size={16} className="flex-shrink-0 mt-0.5" style={{ color: '#1f4fd1' }} />
          <p className="text-sm" style={{ color: '#1f4fd1' }}>
            An invite email will be sent to their inbox with a temporary sign-in link.
          </p>
        </div>

        {/* Footer buttons */}
        <div className="flex gap-3 pt-1">
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
