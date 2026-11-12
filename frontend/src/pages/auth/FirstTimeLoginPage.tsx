import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, AlertCircle, Info } from 'lucide-react';

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

function WelcomeIllustration() {
  return (
    <div className="w-72 rounded-2xl bg-white/10 border border-white/20 p-5 backdrop-blur-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white font-semibold text-sm">
          JT
        </div>
        <div>
          <div className="text-white font-semibold text-sm">Jane Tan</div>
          <div className="text-blue-200 text-xs">Security Operations · Staff</div>
        </div>
      </div>

      <div className="space-y-2.5">
        {[
          { label: 'Review incidents', done: true },
          { label: 'Set your password', done: false },
          { label: 'Complete profile', done: false },
        ].map((step) => (
          <div key={step.label} className="flex items-center gap-3">
            <div className={`w-5 h-5 rounded-full flex-shrink-0 flex items-center justify-center text-xs ${
              step.done
                ? 'bg-green-400 text-white'
                : 'border-2 border-white/30 text-white/40'
            }`}>
              {step.done ? '✓' : ''}
            </div>
            <span className={`text-sm ${step.done ? 'text-white line-through opacity-60' : 'text-white'}`}>
              {step.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function FirstTimeLoginPage() {
  const navigate = useNavigate();
  const [email, setEmail]       = useState('');
  const [tempPassword, setTempPassword] = useState('');
  const [showPw, setShowPw]     = useState(false);
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim() || !tempPassword.trim()) {
      setError('Please enter your email and temporary password.');
      return;
    }

    setLoading(true);
    await new Promise((r) => setTimeout(r, 600));
    setLoading(false);

    // Navigate to set-password page on success
    navigate('/set-password');
  };

  return (
    <div className="flex min-h-screen">
      {/* Left — form */}
      <div className="flex flex-col justify-center w-full lg:w-1/2 px-8 sm:px-16 py-12 bg-white">
        <div className="max-w-sm w-full mx-auto">
          <InsightLogo />

          <h1 className="text-2xl font-bold text-gray-900 mb-1">Sign in</h1>
          <p className="text-sm text-gray-500 mb-6">
            First time here? Enter the temporary password from your invite email.
          </p>

          {/* Info banner */}
          <div className="flex items-start gap-2 mb-6 p-3.5 rounded-lg text-sm" style={{ backgroundColor: '#eef2fb', border: '1px solid #c7d7f8', color: '#1f4fd1' }}>
            <Info size={16} className="flex-shrink-0 mt-0.5" />
            <span>
              Check your inbox for an email from <strong>no-reply@towers.sg</strong> with your temporary password.
            </span>
          </div>

          {error && (
            <div className="flex items-start gap-2 mb-5 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1.5">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@towers.sg"
                className="w-full px-3.5 py-2.5 border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
              />
            </div>

            {/* Temporary password */}
            <div>
              <label htmlFor="temp-password" className="block text-sm font-medium text-gray-700 mb-1.5">
                Temporary password
              </label>
              <div className="relative">
                <input
                  id="temp-password"
                  type={showPw ? 'text' : 'password'}
                  autoComplete="off"
                  value={tempPassword}
                  onChange={(e) => setTempPassword(e.target.value)}
                  placeholder="Paste from your invite email"
                  className="w-full px-3.5 py-2.5 pr-11 border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={showPw ? 'Hide password' : 'Show password'}
                >
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              style={{ backgroundColor: '#1f4fd1', color: '#ffffff', border: 'none', borderRadius: '10px', padding: '12px 0', fontSize: '15px', fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.6 : 1, width: '100%' }}
              className=""
            >
              {loading ? 'Verifying…' : 'Continue'}
            </button>
          </form>

          <p className="mt-5 text-center text-sm text-gray-500">
            Already set up your account?{' '}
            <a href="/login" style={{ color: '#1f4fd1', fontWeight: 500, textDecoration: 'none' }}>
              Sign in
            </a>
          </p>
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
              <Eye size={28} className="text-white" />
            </div>
            <h2 className="text-3xl font-bold text-white leading-snug mb-4">
              Welcome to the team. Let's get you set up.
            </h2>
            <p className="text-blue-200 text-sm leading-relaxed">
              You're just two steps away from having full access to the InSight monitoring platform.
            </p>
          </div>

          <WelcomeIllustration />

          <div className="text-blue-200 text-xs">
            Step 1 of 2 — Verify your identity
          </div>
        </div>
      </div>
    </div>
  );
}
