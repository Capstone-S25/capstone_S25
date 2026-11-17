import { ReactNode, useState, InputHTMLAttributes } from 'react';
import { Eye, EyeOff, AlertCircle } from 'lucide-react';

// Split auth screen: form on the left, Panasonic-blue brand panel on the right.

interface AuthLayoutProps {
  title: string;
  subtitle?: ReactNode;
  children: ReactNode;
  panelTitle: ReactNode;
  panelBody: ReactNode;
  panelVisual?: ReactNode;
  panelFoot?: ReactNode;
}

export function AuthLayout({ title, subtitle, children, panelTitle, panelBody, panelVisual, panelFoot }: AuthLayoutProps) {
  return (
    <div className="flex min-h-screen bg-white">
      <div className="flex w-full flex-col justify-center px-[24px] py-[48px] sm:px-[64px] lg:w-1/2">
        <div className="mx-auto w-full max-w-[400px]">
          <BrandLockup />
          <h1 className="m-0 mt-[40px] text-[26px] font-medium text-ink">{title}</h1>
          {subtitle && <p className="m-0 mt-[6px] text-[14px] leading-[1.5] text-muted">{subtitle}</p>}
          <div className="mt-[24px]">{children}</div>
        </div>
      </div>

      <div className="hidden p-[16px] lg:flex lg:w-1/2">
        <div className="relative flex w-full flex-col items-center justify-center gap-[32px] overflow-hidden rounded-[12px] bg-brand px-[48px] py-[48px] text-center">
          {/* subtle grid texture */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 opacity-[0.07]"
            style={{
              backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)',
              backgroundSize: '40px 40px',
            }}
          />
          <div className="relative max-w-[440px]">
            <h2 className="m-0 text-[34px] leading-[1.2] font-medium text-white">{panelTitle}</h2>
            <p className="m-0 mt-[14px] text-[15px] leading-[1.6] text-[#c9d8f7]">{panelBody}</p>
          </div>
          {panelVisual && <div className="relative">{panelVisual}</div>}
          {panelFoot && <div className="relative text-[12px] text-[#c9d8f7]">{panelFoot}</div>}
        </div>
      </div>
    </div>
  );
}

export function BrandLockup() {
  return (
    <div className="flex items-center gap-[12px]">
      <div className="flex size-[40px] items-center justify-center rounded-[10px] bg-brand">
        <Eye size={23} color="#ffffff" strokeWidth={2.25} />
      </div>
      <div className="flex flex-col">
        <span className="text-[20px] leading-tight font-semibold text-ink">InSight</span>
        <span className="text-[12px] text-muted">Panasonic</span>
      </div>
    </div>
  );
}

export const AUTH_INPUT =
  'w-full rounded-[6px] border border-line bg-white px-[14px] py-[11px] text-[14px] text-ink placeholder:text-faint outline-none transition-colors hover:border-faint focus:border-brand focus:ring-2 focus:ring-brand-soft';

export const AUTH_LABEL = 'mb-[6px] block text-[13px] font-medium text-ink-2';

export const AUTH_SUBMIT =
  'w-full cursor-pointer rounded-[6px] border-none bg-brand py-[12px] text-[14px] font-medium text-white transition-colors hover:bg-brand-hover disabled:cursor-not-allowed disabled:opacity-50';

export function PasswordInput({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <input {...props} type={show ? 'text' : 'password'} className={`${AUTH_INPUT} pr-[44px] ${className}`} />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        aria-label={show ? 'Hide password' : 'Show password'}
        className="absolute inset-y-0 right-0 flex cursor-pointer items-center border-none bg-transparent pr-[14px] text-faint transition-colors hover:text-ink-2"
      >
        {show ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  );
}

export function FormError({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="mb-[16px] flex items-start gap-[8px] rounded-[6px] border border-[#f3c4c2] bg-danger-soft p-[12px] text-[13px] text-danger">
      <AlertCircle size={16} className="mt-[1px] shrink-0" />
      <span>{children}</span>
    </div>
  );
}
