import { useState } from 'react';
import { Modal, LABEL_CLASS } from '../ui/Modal';
import { Button } from '../ui/Button';
import { FormError, PasswordInput } from '../layout/AuthLayout';
import { AuthError, changePassword } from '../../data/auth';

interface ChangePasswordModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ChangePasswordModal({ isOpen, onClose }: ChangePasswordModalProps) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [done, setDone] = useState(false);

  const mismatch = confirmPassword.length > 0 && newPassword !== confirmPassword;
  const isValid = Boolean(currentPassword && newPassword && confirmPassword) && !mismatch;

  const handleSave = async () => {
    if (!isValid) return;
    setError('');
    setSaving(true);
    try {
      // The backend checks the current password and the password rules,
      // and its message (e.g. "Current password is incorrect") is shown as is.
      await changePassword(newPassword, confirmPassword, currentPassword);
      setDone(true);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setError('');
    setDone(false);
    onClose();
  };

  if (done) {
    return (
      <Modal isOpen={isOpen} onClose={handleClose} title="Password changed" width="440px"
        footer={<Button variant="primary" onClick={handleClose}>Done</Button>}
      >
        <p className="m-0 text-[14px] leading-[1.5] text-ink-2">
          Use your new password the next time you sign in.
        </p>
      </Modal>
    );
  }

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Change password" width="440px"
      subtitle="At least 8 characters, with a lower-case and a capital letter, a number and a symbol."
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>Cancel</Button>
          <Button variant="primary" onClick={handleSave} disabled={!isValid || saving}>
            {saving ? 'Saving…' : 'Change password'}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-[16px]">
        {error && <FormError>{error}</FormError>}

        <div>
          <label htmlFor="current-password" className={LABEL_CLASS}>Current password</label>
          <PasswordInput id="current-password" autoComplete="current-password"
            value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} />
        </div>

        <div>
          <label htmlFor="new-password" className={LABEL_CLASS}>New password</label>
          <PasswordInput id="new-password" autoComplete="new-password"
            value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
        </div>

        <div>
          <label htmlFor="confirm-new-password" className={LABEL_CLASS}>Confirm new password</label>
          <PasswordInput id="confirm-new-password" autoComplete="new-password"
            value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)}
            aria-invalid={mismatch} className={mismatch ? '!border-danger' : ''} />
          {mismatch && <p className="m-0 mt-[6px] text-[12px] text-danger">Passwords don't match.</p>}
        </div>
      </div>
    </Modal>
  );
}