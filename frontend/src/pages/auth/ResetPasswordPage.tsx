import { useState, FormEvent } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Check } from 'lucide-react';
import { AuthLayout, AUTH_LABEL, AUTH_SUBMIT, FormError, PasswordInput } from '../../components/layout/AuthLayout';
import { PASSWORD_REQUIREMENTS, PasswordRequirements } from '../../components/layout/PasswordRequirements';
import { AuthError, resetPassword } from '../../data/auth';

// Opened from the "Forgot password?" email: /reset-password?token=...
export default function ResetPasswordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token') ?? '';
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [linkExpired, setLinkExpired] = useState(!token);
  const [loading, setLoading] = useState(false);

  const allRequirementsMet = PASSWORD_REQUIREMENTS.every((r) => r.test(newPassword));
  const passwordsMatch = newPassword === confirmPassword && confirmPassword.length > 0;
  const mismatch = confirmPassword.length > 0 && !passwordsMatch;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    if (!allRequirementsMet) { setError('Your password does not meet all requirements yet.'); return; }
    if (!passwordsMatch) { setError("Passwords don't match. Please check and try again."); return; }
    setLoading(true);
    try {
      await resetPassword(token, newPassword, confirmPassword);
      navigate('/login?reset=1', { replace: true });
    } catch (err) {
      if (err instanceof AuthError && err.status === 400) {
        setLinkExpired(true);
      } else {
        setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title={linkExpired ? 'This link no longer works' : 'Choose a new password'}
      subtitle={linkExpired ? undefined : 'Pick a password only you know. Every device signed in to your account will be signed out.'}
      panelTitle="Almost there. Pick a new password."
      panelBody="Your new password replaces the old one straight away. Keep it somewhere safe. Your security team can't see it."
      panelFoot="Step 2 of 2 · Choose a new password"
    >
      {linkExpired ? (
        <>
          <FormError>
            Reset links work once, for 1 hour. This one has expired, was already used, or a newer one was sent.
          </FormError>
          <Link to="/forgot-password" className={`${AUTH_SUBMIT} block text-center no-underline`}>Send a new link</Link>
        </>
      ) : (
        <>
          {error && <FormError>{error}</FormError>}
          <form onSubmit={handleSubmit} className="flex flex-col gap-[18px]" noValidate>
            <div>
              <label htmlFor="new-password" className={AUTH_LABEL}>New password</label>
              <PasswordInput id="new-password" autoComplete="new-password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Choose a strong password" />
              <PasswordRequirements password={newPassword} />
            </div>

            <div>
              <label htmlFor="confirm-password" className={AUTH_LABEL}>Confirm new password</label>
              <PasswordInput
                id="confirm-password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Re-enter your new password"
                aria-invalid={mismatch}
                className={mismatch ? '!border-danger' : passwordsMatch ? '!border-success' : ''}
              />
              {mismatch && <p className="m-0 mt-[6px] text-[12px] text-danger">Passwords don't match.</p>}
              {passwordsMatch && <p className="m-0 mt-[6px] flex items-center gap-[4px] text-[12px] text-success"><Check size={12} strokeWidth={3} /> Passwords match.</p>}
            </div>

            <button type="submit" disabled={loading || !allRequirementsMet || !passwordsMatch} className={`${AUTH_SUBMIT} mt-[4px]`}>
              {loading ? 'Saving…' : 'Save new password'}
            </button>
          </form>
        </>
      )}

      <p className="m-0 mt-[20px] text-center text-[13px] text-muted">
        <Link to="/login" className="font-medium text-brand no-underline hover:underline">Back to sign in</Link>
      </p>
    </AuthLayout>
  );
}
