import { useState, FormEvent } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Info, Check } from 'lucide-react';
import { AuthLayout, AUTH_INPUT, AUTH_LABEL, AUTH_SUBMIT, FormError, PasswordInput } from '../../components/layout/AuthLayout';
import { AuthError, homePath, login } from '../../data/auth';

const STEPS = [
  { label: 'Verify your invite', state: 'current' },
  { label: 'Set your password', state: 'todo' },
  { label: 'Start reviewing clips', state: 'todo' },
] as const;

function StepsCard() {
  return (
    <div className="w-[300px] rounded-[10px] border border-white/20 bg-white/10 p-[20px] text-left">
      <div className="mb-[16px] flex items-center gap-[12px]">
        <div className="flex size-[40px] items-center justify-center rounded-full bg-white/20 text-[13px] font-semibold text-white">JT</div>
        <div>
          <div className="text-[14px] font-medium text-white">Jane Tan</div>
          <div className="text-[12px] text-[#c9d8f7]">Security operations · Staff</div>
        </div>
      </div>
      <ol className="m-0 flex list-none flex-col gap-[10px] p-0">
        {STEPS.map((s, i) => (
          <li key={s.label} className="flex items-center gap-[10px]">
            <span className={`flex size-[20px] shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
              s.state === 'current' ? 'bg-white text-brand' : 'border border-white/40 text-white/60'
            }`}>
              {i + 1}
            </span>
            <span className={`text-[13px] ${s.state === 'current' ? 'font-medium text-white' : 'text-white/70'}`}>{s.label}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function FirstTimeLoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [tempPassword, setTempPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    if (!email.trim() || !tempPassword.trim()) {
      setError('Please enter your email and temporary password.');
      return;
    }
    setLoading(true);
    try {
      // The temporary password signs them in; the backend then only allows "set your password".
      const user = await login(email.trim(), tempPassword.trim());
      navigate(homePath(user));
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome to InSight"
      subtitle="Enter the temporary password from your invite email to get started."
      panelTitle="Welcome to the team. Let's get you set up."
      panelBody="Two quick steps and you'll have full access to the InSight monitoring platform."
      panelVisual={<StepsCard />}
      panelFoot="Step 1 of 2 · Verify your invite"
    >
      <div className="mb-[20px] flex items-start gap-[8px] rounded-[6px] bg-brand-soft p-[12px] text-[13px] leading-[1.5] text-ink-2">
        <Info size={16} className="mt-[1px] shrink-0 text-brand" />
        <span>Look for an email from <strong className="font-medium text-ink">no-reply@towers.sg</strong> with your temporary password.</span>
      </div>

      {error && <FormError>{error}</FormError>}

      <form onSubmit={handleSubmit} className="flex flex-col gap-[16px]" noValidate>
        <div>
          <label htmlFor="email" className={AUTH_LABEL}>Email</label>
          <input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@towers.sg" className={AUTH_INPUT} />
        </div>
        <div>
          <label htmlFor="temp-password" className={AUTH_LABEL}>Temporary password</label>
          <PasswordInput id="temp-password" autoComplete="off" value={tempPassword} onChange={(e) => setTempPassword(e.target.value)} placeholder="Paste from your invite email" />
        </div>
        <button type="submit" disabled={loading} className={`${AUTH_SUBMIT} mt-[4px]`}>
          {loading ? 'Verifying…' : 'Continue'}
        </button>
      </form>

      <p className="m-0 mt-[20px] flex items-center justify-center gap-[4px] text-[13px] text-muted">
        <Check size={13} /> Already set up?{' '}
        <Link to="/login" className="font-medium text-brand no-underline hover:underline">Sign in</Link>
      </p>
    </AuthLayout>
  );
}
