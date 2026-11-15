import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Check } from 'lucide-react';
import { AuthLayout, AUTH_LABEL, AUTH_SUBMIT, FormError, PasswordInput } from '../../components/layout/AuthLayout';

interface Requirement {
  label: string;
  test: (pw: string) => boolean;
}

const REQUIREMENTS: Requirement[] = [
  { label: 'At least 8 characters', test: (pw) => pw.length >= 8 },
  { label: 'Includes a number', test: (pw) => /\d/.test(pw) },
  { label: "Doesn't use personal info", test: (pw) => pw.length > 0 && !/jane|tan|raj|admin|staff|123|password/i.test(pw) },
];

function StrengthBar({ password }: { password: string }) {
  const passed = REQUIREMENTS.filter((r) => r.test(password)).length;
  const pct = password.length === 0 ? 0 : (passed / REQUIREMENTS.length) * 100;
  const color = pct <= 34 ? 'var(--color-danger)' : pct <= 67 ? 'var(--color-warning-bar)' : 'var(--color-success)';
  return (
    <div className="mt-[8px] h-[5px] w-full overflow-hidden rounded-[2.5px] bg-track">
      <div className="h-full rounded-[2.5px] transition-all duration-300" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

export default function SetPasswordPage() {
  const navigate = useNavigate();
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const allRequirementsMet = REQUIREMENTS.every((r) => r.test(newPassword));
  const passwordsMatch = newPassword === confirmPassword && confirmPassword.length > 0;
  const mismatch = confirmPassword.length > 0 && !passwordsMatch;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    if (!allRequirementsMet) { setError('Your password does not meet all requirements yet.'); return; }
    if (!passwordsMatch) { setError("Passwords don't match. Please check and try again."); return; }
    setLoading(true);
    await new Promise((r) => setTimeout(r, 700));
    setLoading(false);
    navigate('/staff/home');
  };

  return (
    <AuthLayout
      title="Set your password"
      subtitle={<>Welcome, <strong className="font-medium text-ink">Jane</strong>. Replace the temporary password with one only you know.</>}
      panelTitle="Almost there. Pick your own password."
      panelBody="Your new password replaces the temporary one. Keep it somewhere safe. Your security team can't see it."
      panelVisual={
        <div className="w-[300px] rounded-[10px] border border-white/20 bg-white/10 p-[20px] text-left">
          <p className="m-0 mb-[12px] text-[14px] font-medium text-white">Tips for a strong password</p>
          <ul className="m-0 flex list-none flex-col gap-[10px] p-0">
            {['Mix letters, numbers and symbols', 'Avoid easily guessed words or dates', "Don't reuse passwords from other services"].map((tip) => (
              <li key={tip} className="flex items-start gap-[10px] text-[12px] leading-[1.5] text-[#dbe5ff]">
                <span className="mt-[6px] size-[5px] shrink-0 rounded-full bg-[#9db5ea]" />
                {tip}
              </li>
            ))}
          </ul>
        </div>
      }
      panelFoot="Step 2 of 2 · Set your password"
    >
      {error && <FormError>{error}</FormError>}

      <form onSubmit={handleSubmit} className="flex flex-col gap-[18px]" noValidate>
        <div>
          <label htmlFor="new-password" className={AUTH_LABEL}>New password</label>
          <PasswordInput id="new-password" autoComplete="new-password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Choose a strong password" />
          <StrengthBar password={newPassword} />
          <ul className="m-0 mt-[12px] flex list-none flex-col gap-[6px] p-0">
            {REQUIREMENTS.map((req) => {
              const met = req.test(newPassword);
              return (
                <li key={req.label} className="flex items-center gap-[8px]">
                  <span className={`flex size-[16px] shrink-0 items-center justify-center rounded-full transition-colors ${met ? 'bg-success' : 'bg-track'}`}>
                    {met && <Check size={10} className="text-white" strokeWidth={3} />}
                  </span>
                  <span className={`text-[12px] transition-colors ${met ? 'font-medium text-success' : 'text-muted'}`}>{req.label}</span>
                </li>
              );
            })}
          </ul>
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
          {loading ? 'Saving…' : 'Save and continue'}
        </button>
      </form>
    </AuthLayout>
  );
}
