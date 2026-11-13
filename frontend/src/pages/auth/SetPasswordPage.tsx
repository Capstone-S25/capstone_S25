import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, AlertCircle, Check } from 'lucide-react';

function InsightLogo() {
  return (
    <div className="flex items-center gap-3 mb-8">
      <div className="flex items-center justify-center w-[50px] h-[50px] rounded-[25px]" style={{ backgroundColor: '#1f4fd1' }}>
        <Eye size={22} className="text-white" strokeWidth={2.5} />
      </div>
      <span style={{ fontSize: '32px', letterSpacing: '-0.64px', display: 'flex', lineHeight: 1 }}>
        <span style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 500, color: '#111827' }}>In</span>
        <span style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 600, color: '#1f4fd1' }}>Sight</span>
      </span>
    </div>
  );
}

interface Requirement {
  label: string;
  test: (pw: string) => boolean;
}

const REQUIREMENTS: Requirement[] = [
  { label: 'At least 8 characters',         test: (pw) => pw.length >= 8 },
  { label: 'Includes a number',              test: (pw) => /\d/.test(pw) },
  { label: "Doesn't use personal info",      test: (pw) => !/jane|tan|raj|admin|staff|123|password/i.test(pw) },
];

function PasswordStrengthBar({ password }: { password: string }) {
  const passed = REQUIREMENTS.filter((r) => r.test(password)).length;
  const pct = password.length === 0 ? 0 : Math.round((passed / REQUIREMENTS.length) * 100);

  const color =
    pct === 0 ? 'bg-gray-200' :
    pct <= 33  ? 'bg-red-400' :
    pct <= 66  ? 'bg-orange-400' :
    'bg-green-500';

  return (
    <div className="mt-2 h-1.5 w-full rounded-full bg-gray-100 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-300 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function SetPasswordPage() {
  const navigate = useNavigate();
  const [newPassword, setNewPassword]     = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showNew, setShowNew]             = useState(false);
  const [showConfirm, setShowConfirm]     = useState(false);
  const [error, setError]                 = useState('');
  const [loading, setLoading]             = useState(false);

  const allRequirementsMet = REQUIREMENTS.every((r) => r.test(newPassword));
  const passwordsMatch = newPassword === confirmPassword && confirmPassword.length > 0;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!allRequirementsMet) {
      setError('Your password does not meet all requirements yet.');
      return;
    }
    if (!passwordsMatch) {
      setError("Passwords don't match. Please check and try again.");
      return;
    }

    setLoading(true);
    await new Promise((r) => setTimeout(r, 700));
    setLoading(false);

    navigate('/staff/home');
  };

  return (
    <div className="flex min-h-screen">
      {/* Left — form */}
      <div className="flex flex-col justify-center w-full lg:w-1/2 px-8 sm:px-16 py-12 bg-white">
        <div className="max-w-sm w-full mx-auto">
          <InsightLogo />

          <h1 className="text-2xl font-bold text-gray-900 mb-1">Set your password</h1>
          <p className="text-sm text-gray-500 mb-7">
            Welcome, <strong className="text-gray-700">Jane</strong>. Replace the temporary
            password with one only you know.
          </p>

          {error && (
            <div className="flex items-start gap-2 mb-5 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {/* New password */}
            <div>
              <label htmlFor="new-password" className="block text-sm font-medium text-gray-700 mb-1.5">
                New password
              </label>
              <div className="relative">
                <input
                  id="new-password"
                  type={showNew ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Choose a strong password"
                  className="w-full px-3.5 py-2.5 pr-11 border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                />
                <button
                  type="button"
                  onClick={() => setShowNew((v) => !v)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={showNew ? 'Hide password' : 'Show password'}
                >
                  {showNew ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              <PasswordStrengthBar password={newPassword} />

              {/* Requirements checklist */}
              <ul className="mt-3 space-y-1.5">
                {REQUIREMENTS.map((req) => {
                  const met = req.test(newPassword);
                  return (
                    <li key={req.label} className="flex items-center gap-2">
                      <span
                        className={`flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center transition-colors ${
                          met ? 'bg-green-500' : 'bg-gray-200'
                        }`}
                      >
                        {met && <Check size={10} className="text-white" strokeWidth={3} />}
                      </span>
                      <span
                        className={`text-xs transition-colors ${
                          met ? 'text-green-700 font-medium' : 'text-gray-500'
                        }`}
                      >
                        {req.label}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>

            {/* Confirm password */}
            <div>
              <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-700 mb-1.5">
                Confirm new password
              </label>
              <div className="relative">
                <input
                  id="confirm-password"
                  type={showConfirm ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter your new password"
                  className={`w-full px-3.5 py-2.5 pr-11 border rounded-lg text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:ring-2 focus:border-transparent transition ${
                    confirmPassword.length > 0
                      ? passwordsMatch
                        ? 'border-green-400 focus:ring-green-400'
                        : 'border-red-300 focus:ring-red-400'
                      : 'border-gray-300 focus:ring-blue-500'
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm((v) => !v)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={showConfirm ? 'Hide password' : 'Show password'}
                >
                  {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              {confirmPassword.length > 0 && !passwordsMatch && (
                <p className="mt-1.5 text-xs text-red-600">Passwords don't match.</p>
              )}
              {passwordsMatch && (
                <p className="mt-1.5 text-xs text-green-600 flex items-center gap-1">
                  <Check size={12} strokeWidth={3} /> Passwords match.
                </p>
              )}
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading || !allRequirementsMet || !passwordsMatch}
              style={{ backgroundColor: '#1f4fd1', color: '#ffffff', border: 'none', borderRadius: '10px', padding: '12px 0', fontSize: '15px', fontWeight: 600, cursor: (loading || !allRequirementsMet || !passwordsMatch) ? 'not-allowed' : 'pointer', opacity: (loading || !allRequirementsMet || !passwordsMatch) ? 0.5 : 1, width: '100%' }}
              className=""
            >
              {loading ? 'Saving…' : 'Save and continue'}
            </button>
          </form>
        </div>
      </div>

      {/* Right — marketing */}
      <div
        className="hidden lg:flex flex-col items-center justify-center w-1/2 px-16 py-12 relative overflow-hidden"
        style={{ backgroundColor: '#1f4fd1' }}
      >
        <div className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage: 'radial-gradient(circle at 20% 80%, #fff 0%, transparent 50%), radial-gradient(circle at 80% 20%, #fff 0%, transparent 50%)',
          }}
        />

        <div className="relative z-10 max-w-sm flex flex-col items-center text-center gap-10">
          <div>
            <div className="inline-flex items-center justify-center w-14 h-14 bg-white/15 rounded-2xl mb-5">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="w-7 h-7 text-white"
                aria-hidden="true"
              >
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-white leading-snug mb-4">
              Almost there. Pick your own password.
            </h2>
            <p className="text-blue-200 text-sm leading-relaxed">
              Your new password replaces the temporary one. Keep it somewhere safe — your security team won't be able to see it.
            </p>
          </div>

          {/* Security tips card */}
          <div className="w-72 rounded-2xl bg-white/10 border border-white/20 p-5 backdrop-blur-sm text-left">
            <p className="text-white font-semibold text-sm mb-3">Tips for a strong password</p>
            <ul className="space-y-2.5">
              {[
                'Use a mix of letters, numbers, and symbols',
                'Avoid easily guessable words or dates',
                "Don't reuse passwords from other services",
              ].map((tip) => (
                <li key={tip} className="flex items-start gap-2.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-300 flex-shrink-0 mt-1.5" />
                  <span className="text-blue-100 text-xs leading-relaxed">{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="text-blue-200 text-xs">
            Step 2 of 2 — Set your password
          </div>
        </div>
      </div>
    </div>
  );
}
