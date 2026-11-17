import { ButtonHTMLAttributes, ReactNode } from 'react';

// Figma: Button/Accept (primary), Button/Dismiss (secondary), Button/Escalate (danger)
// 6px radius, 13px medium label, optional 16px leading icon.

type Variant = 'primary' | 'secondary' | 'danger' | 'danger-solid' | 'ghost';
type Size = 'sm' | 'md';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  children?: ReactNode;
  className?: string;
}

const VARIANT: Record<Variant, string> = {
  primary:        'bg-brand text-white border border-brand hover:bg-brand-hover hover:border-brand-hover',
  secondary:      'bg-white text-ink border border-line hover:bg-head',
  danger:         'bg-white text-danger border border-danger hover:bg-danger-soft',
  'danger-solid': 'bg-danger text-white border border-danger hover:bg-[#a92724] hover:border-[#a92724]',
  ghost:          'bg-transparent text-ink-2 border border-transparent hover:bg-head',
};

const SIZE: Record<Size, { base: string; icon: string }> = {
  md: { base: 'py-[10px] px-[16px] text-[13px] gap-[6px]', icon: 'pl-[12px]' },
  sm: { base: 'py-[7px] px-[12px] text-[12px] gap-[6px]',  icon: 'pl-[10px]' },
};

export function Button({
  variant = 'primary',
  size = 'md',
  icon,
  children,
  className = '',
  type = 'button',
  ...rest
}: ButtonProps) {
  const s = SIZE[size];
  return (
    <button
      type={type}
      className={`inline-flex shrink-0 cursor-pointer items-center justify-center rounded-[6px] font-medium leading-none whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT[variant]} ${s.base} ${icon && children ? s.icon : ''} ${className}`}
      {...rest}
    >
      {icon && <span className="flex shrink-0 items-center">{icon}</span>}
      {children}
    </button>
  );
}
