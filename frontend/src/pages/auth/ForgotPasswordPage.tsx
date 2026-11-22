import { useState, FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { MailCheck } from 'lucide-react';
import { AuthLayout, AUTH_INPUT, AUTH_LABEL, AUTH_SUBMIT, FormError } from '../../components/layout/AuthLayout';
import { AuthError, requestPasswordReset } from '../../data/auth';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [sentTo, setSentTo] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    if (!email.trim()) {
      setError('Please enter your email.');
      return;
    }
    setLoading(true);
    try {
      await requestPasswordReset(email.trim());
      setSentTo(email.trim());
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title={sentTo ? 'Check your email' : 'Forgot your password?'}
      subtitle={sentTo ? undefined : "Enter your work email and we'll send you a link to choose a new password."}
      panelTitle="Locked out? It happens."
      panelBody="We'll email you a one-time link. Your current password keeps working until you use it."
      panelFoot="Step 1 of 2 · Get a reset link"
    >
      {sentTo ? (
        <div className="flex items-start gap-[10px] rounded-[6px] bg-brand-soft p-[14px] text-[13px] leading-[1.6] text-ink-2">
          <MailCheck size={18} className="mt-[1px] shrink-0 text-brand" />
          <span>
            If <strong className="font-medium text-ink">{sentTo}</strong> has an InSight account, a link to choose a new
            password is on its way. It works once, for 1 hour. Not there? Check your spam folder, or ask again in a minute.
          </span>
        </div>
      ) : (
        <>
          {error && <FormError>{error}</FormError>}
          <form onSubmit={handleSubmit} className="flex flex-col gap-[16px]" noValidate>
            <div>
              <label htmlFor="email" className={AUTH_LABEL}>Email</label>
              <input id="email" type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@company.sg" className={AUTH_INPUT} />
            </div>
            <button type="submit" disabled={loading} className={`${AUTH_SUBMIT} mt-[4px]`}>
              {loading ? 'Sending…' : 'Send reset link'}
            </button>
          </form>
        </>
      )}

      <p className="m-0 mt-[20px] text-center text-[13px] text-muted">
        Remembered it?{' '}
        <Link to="/login" className="font-medium text-brand no-underline hover:underline">Back to sign in</Link>
      </p>
    </AuthLayout>
  );
}
