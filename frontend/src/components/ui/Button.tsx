import { ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'danger';
type Size = 'sm' | 'md';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
  className?: string;
}

export function Button({
  variant = 'primary',
  size = 'md',
  children,
  className = '',
  disabled,
  type = 'button',
  style,
  ...rest
}: ButtonProps) {
  const baseStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: 'none',
    transition: 'background-color 0.15s',
    fontFamily: 'inherit',
    opacity: disabled ? 0.6 : 1,
  };

  const variantStyle: React.CSSProperties =
    variant === 'primary'
      ? {
          backgroundColor: '#1f4fd1',
          color: '#ffffff',
          border: 'none',
          fontWeight: 600,
        }
      : variant === 'secondary'
      ? {
          backgroundColor: '#ffffff',
          color: '#111827',
          border: '1px solid #d1d5db',
          fontWeight: 500,
        }
      : {
          // danger
          backgroundColor: '#dc2626',
          color: '#ffffff',
          border: 'none',
          fontWeight: 600,
        };

  const sizeStyle: React.CSSProperties =
    size === 'sm'
      ? {
          padding: '10px 16px',
          fontSize: '14px',
          borderRadius: '8px',
        }
      : {
          padding: '10px 16px',
          fontSize: '14px',
          borderRadius: '8px',
        };

  return (
    <button
      type={type}
      disabled={disabled}
      className={className}
      style={{ ...baseStyle, ...variantStyle, ...sizeStyle, ...style }}
      {...rest}
    >
      {children}
    </button>
  );
}
