import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, AlertCircle } from 'lucide-react';

// Logo component used above the form
function InsightLogo() {
  return (
    <div className="flex items-center gap-3" style={{ marginBottom: '28px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '51px',
          height: '51px',
          backgroundColor: '#1f4fd1',
          borderRadius: '25.5px',
          flexShrink: 0,
        }}
      >
        <Eye size={22} color="#ffffff" strokeWidth={2.5} />
      </div>
      <span
        style={{
          fontSize: '32px',
          letterSpacing: '-0.64px',
          lineHeight: 1,
          display: 'flex',
        }}
      >
        <span style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 500, color: '#111827' }}>In</span>
        <span style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 600, color: '#1f4fd1' }}>Sight</span>
      </span>
    </div>
  );
}

// Incident preview card shown on the right panel
function IncidentPreviewCard() {
  return (
    <div
      style={{
        backgroundColor: '#ffffff',
        borderRadius: '14px',
        padding: '16px',
        gap: '8px',
        boxShadow: '0px 18px 40px 0px rgba(5,13,51,0.35)',
        display: 'flex',
        flexDirection: 'column',
        width: '280px',
      }}
    >
      {/* Top row: severity pill + title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span
          style={{
            backgroundColor: '#f2c0ba',
            color: '#8c2a22',
            borderRadius: '6px',
            padding: '4px 10px',
            fontSize: '11px',
            fontWeight: 600,
          }}
        >
          Critical
        </span>
        <span style={{ fontSize: '14px', fontWeight: 600, color: '#111827' }}>
          Smoking · Carpark B2
        </span>
      </div>

      {/* Description */}
      <p style={{ fontSize: '12px', color: '#6b7380', margin: 0, lineHeight: 1.5 }}>
        Individual observed smoking near pillar P-04. Smoke clearly visible.
      </p>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '6px' }}>
        <button
          type="button"
          style={{
            backgroundColor: '#17994a',
            color: '#ffffff',
            border: 'none',
            borderRadius: '7px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Accept
        </button>
        <button
          type="button"
          style={{
            backgroundColor: '#ffffff',
            color: '#374151',
            border: '1px solid #d9dee5',
            borderRadius: '7px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Dismiss
        </button>
        <button
          type="button"
          style={{
            backgroundColor: '#ffffff',
            color: '#bf2921',
            border: '1px solid #e59994',
            borderRadius: '7px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Escalate
        </button>
      </div>
    </div>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw]     = useState(false);
  const [keepSignedIn, setKeepSignedIn] = useState(false);
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [emailFocused, setEmailFocused] = useState(false);
  const [pwFocused, setPwFocused] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim() || !password.trim()) {
      setError('Please enter your email and password.');
      return;
    }

    setLoading(true);
    await new Promise((r) => setTimeout(r, 600));
    setLoading(false);

    const e_ = email.trim().toLowerCase();
    const p_ = password.trim();

    if (e_ === 'admin@insight.sg' && p_ === 'Admin123!') {
      navigate('/admin/home');
    } else if (e_ === 'staff@insight.sg' && p_ === 'Staff123!') {
      navigate('/staff/home');
    } else {
      setError('Incorrect email or password. Please try again.');
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f6f7f9',
        display: 'flex',
        alignItems: 'stretch',
        position: 'relative',
      }}
    >
      {/* Left half — white background with form */}
      <div
        style={{
          flex: 1,
          backgroundColor: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-start',
          padding: '48px 0 48px 128px',
        }}
      >
        <div style={{ width: '460px' }}>
          {/* Logo above form */}
          <InsightLogo />

          {/* Form card */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              boxShadow: '0px 8px 28px 0px rgba(15,23,51,0.06)',
              width: '460px',
              paddingBottom: '32px',
              paddingTop: '36px',
              paddingLeft: '36px',
              paddingRight: '36px',
              display: 'flex',
              flexDirection: 'column',
              gap: '18px',
            }}
          >
            <h1
              style={{
                fontSize: '28px',
                fontWeight: 600,
                color: '#121726',
                margin: 0,
              }}
            >
              Sign in
            </h1>

            {error && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  padding: '12px',
                  backgroundColor: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: '8px',
                  fontSize: '14px',
                  color: '#b91c1c',
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} noValidate>
              {/* Email */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label
                  htmlFor="email"
                  style={{ fontSize: '13px', fontWeight: 500, color: '#121726' }}
                >
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  autoComplete="off"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setEmailFocused(true)}
                  onBlur={() => setEmailFocused(false)}
                  placeholder="Email address"
                  className="placeholder-gray-400"
                  style={{
                    border: emailFocused ? '1.5px solid #1f4fd1' : '1px solid #d9dee5',
                    borderRadius: '10px',
                    padding: '12px 14px',
                    fontSize: '15px',
                    color: '#111827',
                    backgroundColor: '#ffffff',
                    outline: 'none',
                    width: '100%',
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              {/* Password */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label
                  htmlFor="password"
                  style={{ fontSize: '13px', fontWeight: 500, color: '#121726' }}
                >
                  Password
                </label>
                <div style={{ position: 'relative' }}>
                  <input
                    id="password"
                    type={showPw ? 'text' : 'password'}
                    autoComplete="off"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onFocus={() => setPwFocused(true)}
                    onBlur={() => setPwFocused(false)}
                    placeholder="Password"
                    className="placeholder-gray-400"
                    style={{
                      border: pwFocused ? '1.5px solid #1f4fd1' : '1px solid #d9dee5',
                      borderRadius: '10px',
                      padding: '12px 44px 12px 14px',
                      fontSize: '15px',
                      color: '#111827',
                      backgroundColor: '#ffffff',
                      outline: 'none',
                      width: '100%',
                      boxSizing: 'border-box',
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw((v) => !v)}
                    style={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      paddingRight: '14px',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      color: '#9ca3af',
                    }}
                    aria-label={showPw ? 'Hide password' : 'Show password'}
                  >
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Keep me signed in */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input
                  id="keep-signed-in"
                  type="checkbox"
                  checked={keepSignedIn}
                  onChange={(e) => setKeepSignedIn(e.target.checked)}
                  style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                />
                <label
                  htmlFor="keep-signed-in"
                  style={{ fontSize: '14px', color: '#374151', cursor: 'pointer' }}
                >
                  Keep me signed in
                </label>
              </div>

              {/* Sign in button */}
              <button
                type="submit"
                disabled={loading}
                style={{
                  backgroundColor: loading ? '#a5b4fc' : '#1f4fd1',
                  color: '#ffffff',
                  fontWeight: 600,
                  fontSize: '15px',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '12px 0',
                  width: '100%',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'background-color 0.15s',
                }}
              >
                {loading ? 'Signing in…' : 'Sign in'}
              </button>
            </form>

            <p style={{ textAlign: 'center', fontSize: '14px', color: '#6b7280', margin: 0 }}>
              Forgot your password?{' '}
              <a href="#" style={{ color: '#1f4fd1', fontWeight: 500, textDecoration: 'none' }}>
                Reset here
              </a>
            </p>
          </div>
        </div>
      </div>

      {/* Right panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          backgroundColor: '#f6f7f9',
        }}
      >
        <div
          style={{
            backgroundColor: '#1f4fd1',
            borderRadius: '20px',
            width: '100%',
            height: '100%',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '48px 40px',
            gap: '32px',
          }}
        >
          {/* Top-left decorative ellipse */}
          <div
            style={{
              position: 'absolute',
              top: '-80px',
              left: '-80px',
              width: '320px',
              height: '320px',
              borderRadius: '50%',
              border: '60px solid rgba(255,255,255,0.08)',
              pointerEvents: 'none',
            }}
          />
          {/* Bottom-right decorative ellipse */}
          <div
            style={{
              position: 'absolute',
              bottom: '-100px',
              right: '-100px',
              width: '380px',
              height: '380px',
              borderRadius: '50%',
              border: '70px solid rgba(255,255,255,0.07)',
              pointerEvents: 'none',
            }}
          />

          {/* Hero text */}
          <div style={{ textAlign: 'center', position: 'relative', zIndex: 1 }}>
            <h2
              style={{
                fontSize: '40px',
                fontWeight: 700,
                color: '#ffffff',
                margin: '0 0 16px 0',
                lineHeight: 1.2,
              }}
            >
              Every flagged clip,<br />reviewed in one place
            </h2>
            <p
              style={{
                fontSize: '16px',
                color: '#dbe5ff',
                margin: 0,
                lineHeight: 1.6,
              }}
            >
              The AI watches every camera and writes up what it saw.<br />
              Your team only reviews what actually needs a decision.
            </p>
          </div>

          {/* Preview card */}
          <div style={{ position: 'relative', zIndex: 1 }}>
            <IncidentPreviewCard />
          </div>

          {/* Tag dots */}
          <div style={{ display: 'flex', gap: '8px', position: 'relative', zIndex: 1 }}>
            {['AI-powered triage', 'Full audit trail', 'Real-time alerts'].map((t) => (
              <span
                key={t}
                style={{
                  fontSize: '12px',
                  color: '#dbe5ff',
                  backgroundColor: 'rgba(255,255,255,0.15)',
                  borderRadius: '20px',
                  padding: '4px 12px',
                }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
