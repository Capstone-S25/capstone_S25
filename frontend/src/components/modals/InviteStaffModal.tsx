import { useState } from 'react';
import { Modal, INPUT_CLASS, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';
import { FormError } from '../layout/AuthLayout';
import { AuthError, inviteStaff, type Role } from '../../data/auth';

interface InviteStaffModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** Called after the backend has created the account and sent the email. */
  onInvited?: (email: string, name: string) => void;
}

// What each dropdown choice is saved as in the backend.
// The team's decision: a Supervisor is an admin in the backend.
const ROLE_FOR_API: Record<string, Role> = {
  Staff: 'staff',
  Supervisor: 'admin',
};

export default function InviteStaffModal({ isOpen, onClose, onInvited }: InviteStaffModalProps) {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('Staff');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [sentTo, setSentTo] = useState(''); // set once the invite has gone out

  const handleInvite = async () => {
    if (!fullName.trim() || !email.trim()) return;
    setError('');
    setSending(true);
    try {
      await inviteStaff(fullName.trim(), email.trim(), ROLE_FOR_API[role]);
      onInvited?.(email.trim(), fullName.trim());
      setSentTo(`${fullName.trim()} (${email.trim()})`);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setSending(false);
    }
  };

  const handleClose = () => {
    setFullName('');
    setEmail('');
    setRole('Staff');
    setError('');
    setSentTo('');
    onClose();
  };

  const isValid = Boolean(fullName.trim() && email.trim());

  if (sentTo) {
    return (
      <Modal isOpen={isOpen} onClose={handleClose} title="Invite sent" width="480px"
        footer={<Button variant="primary" onClick={handleClose}>Done</Button>}
      >
        <p className="m-0 text-[14px] leading-[1.5] text-ink-2">
          {sentTo} will get an email with a temporary password. It works for 7 days,
          and they'll choose their own password when they first sign in.
        </p>
      </Modal>
    );
  }

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
          disabled={!isValid || sending}
          >
          {sending ? 'Sending…' : 'Send invite'}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-[16px]">
        {error && <FormError>{error}</FormError>}

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