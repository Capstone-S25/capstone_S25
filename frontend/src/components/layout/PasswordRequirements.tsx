import { Check } from 'lucide-react';

// Same rules as the backend (cayla-backend/app/password_policy.py).
export const PASSWORD_REQUIREMENTS: { label: string; test: (pw: string) => boolean }[] = [
  { label: 'At least 8 characters', test: (pw) => pw.length >= 8 },
  { label: 'Includes a lower-case letter', test: (pw) => /[a-z]/.test(pw) },
  { label: 'Includes a capital letter', test: (pw) => /[A-Z]/.test(pw) },
  { label: 'Includes a number', test: (pw) => /\d/.test(pw) },
  { label: 'Includes a symbol, such as ! @ # or -', test: (pw) => /[^A-Za-z0-9]/.test(pw) },
];

/** Strength bar plus a tick per rule, under a "new password" box. */
export function PasswordRequirements({ password }: { password: string }) {
  const passed = PASSWORD_REQUIREMENTS.filter((r) => r.test(password)).length;
  const pct = password.length === 0 ? 0 : (passed / PASSWORD_REQUIREMENTS.length) * 100;
  const color = pct <= 34 ? 'var(--color-danger)' : pct <= 67 ? 'var(--color-warning-bar)' : 'var(--color-success)';
  return (
    <>
      <div className="mt-[8px] h-[5px] w-full overflow-hidden rounded-[2.5px] bg-track">
        <div className="h-full rounded-[2.5px] transition-all duration-300" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <ul className="m-0 mt-[12px] flex list-none flex-col gap-[6px] p-0">
        {PASSWORD_REQUIREMENTS.map((req) => {
          const met = req.test(password);
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
    </>
  );
}
