import { useState, FormEvent } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { CircleX, TriangleAlert, CircleCheck } from 'lucide-react';
import { AuthLayout, AUTH_INPUT, AUTH_LABEL, AUTH_SUBMIT, FormError, PasswordInput } from '../../components/layout/AuthLayout';
import { StatusPill } from '../../components/ui/Badge';
import { AuthError, homePath, login } from '../../data/auth';

// Preview of a review-queue item, shown on the brand panel
function IncidentPreviewCard() {
  return (
    <div className="flex w-[320px] flex-col gap-[10px] rounded-[10px] bg-white p-[16px] text-left shadow-[0_18px_40px_rgba(0,20,70,0.35)]">
      <div className="flex items-center gap-[8px]">
        <span className="text-[14px] font-medium text-ink">Smoking · Carpark B2</span>
        <StatusPill tone="red">Critical</StatusPill>
      </div>
      <p className="m-0 text-[12px] leading-[1.5] text-muted">Person smoking next to the staircase door, ash dropped on the floor.</p>
      <div className="flex gap-[6px]">
        <span className="flex items-center gap-[4px] rounded-[6px] border border-line px-[8px] py-[5px] text-[11px] font-medium text-ink"><CircleX size={12} />Dismiss</span>
        <span className="flex items-center gap-[4px] rounded-[6px] border border-danger px-[8px] py-[5px] text-[11px] font-medium text-danger"><TriangleAlert size={12} />Escalate</span>
        <span className="flex items-center gap-[4px] rounded-[6px] bg-brand px-[8px] py-[5px] text-[11px] font-medium text-white"><CircleCheck size={12} />Accept</span>
      </div>
    </div>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  // Arrived from the reset-password page after choosing a new password.
  const [searchParams] = useSearchParams();
  const passwordReset = searchParams.get('reset') === '1';
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [keepSignedIn, setKeepSignedIn] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim() || !password.trim()) {
      setError('Please enter your email and password.');
      return;
    }

    setLoading(true);
    try {
      const user = await login(email.trim(), password);
      navigate(homePath(user));
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Sign in"
      subtitle="Review flagged clips from every camera in one place."
      panelTitle={<>Every flagged clip,<br />reviewed in one place</>}
      panelBody="The AI watches every camera and writes up what it saw. Your team only reviews what actually needs a decision."
      panelVisual={<IncidentPreviewCard />}
      panelFoot={
        <div className="flex gap-[8px]">
          {['AI-powered triage', 'Full audit trail', 'Real-time alerts'].map((t) => (
            <span key={t} className="rounded-full bg-white/15 px-[12px] py-[4px]">{t}</span>
          ))}
        </div>
      }
    >
      {passwordReset && !error && (
        <div role="status" className="mb-[16px] flex items-start gap-[8px] rounded-[6px] bg-brand-soft p-[12px] text-[13px] text-ink-2">
          <CircleCheck size={16} className="mt-[1px] shrink-0 text-success" />
          <span>Your password has been changed. Sign in with your new password.</span>
        </div>
      )}
      {error && <FormError>{error}</FormError>}

      <form onSubmit={handleSubmit} className="flex flex-col gap-[16px]" noValidate>
        <div>
          <label htmlFor="email" className={AUTH_LABEL}>Email</label>
          <input id="email" type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@company.sg" className={AUTH_INPUT} />
        </div>

        <div>
          <label htmlFor="password" className={AUTH_LABEL}>Password</label>
          <PasswordInput id="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
        </div>

        <div className="flex items-center justify-between">
          <label htmlFor="keep-signed-in" className="flex cursor-pointer items-center gap-[8px] text-[13px] text-ink-2">
            <input id="keep-signed-in" type="checkbox" checked={keepSignedIn} onChange={(e) => setKeepSignedIn(e.target.checked)} className="size-[15px] cursor-pointer accent-[var(--color-brand)]" />
            Keep me signed in
          </label>
          <Link to="/forgot-password" className="text-[13px] font-medium text-brand no-underline hover:underline">Forgot password?</Link>
        </div>

        <button type="submit" disabled={loading} className={`${AUTH_SUBMIT} mt-[4px]`}>
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
      </form>

      <p className="m-0 mt-[20px] text-center text-[13px] text-muted">
        First time here?{' '}
        <Link to="/first-login" className="font-medium text-brand no-underline hover:underline">Use your invite</Link>
      </p>
    </AuthLayout>
  );
}
