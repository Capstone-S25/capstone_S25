import { ReactNode, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { CloseButton } from './Modal';

// Right-hand side panel, used for clip and camera details.
// Matches the Home side panel: flat white, 1px line border, 20–28px padding.

interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title: ReactNode;
  titleAdornment?: ReactNode;
  subtitle?: ReactNode;
  width?: number;
  children: ReactNode;
  footer?: ReactNode;
}

export function Drawer({ isOpen, onClose, title, titleAdornment, subtitle, width = 520, children, footer }: DrawerProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <>
      <div className="v2-fade-in fixed inset-0 z-[199] bg-[rgba(18,26,43,0.35)]" onClick={onClose} aria-hidden="true" />
      <aside
        role="dialog"
        aria-modal="true"
        className="v2-slide-in fixed top-0 right-0 bottom-0 z-[200] flex max-w-full flex-col border-l border-line bg-white shadow-[-8px_0_32px_rgba(18,26,43,0.12)]"
        style={{ width }}
      >
        <div className="flex items-start gap-[10px] border-b border-line-soft px-[28px] pt-[20px] pb-[16px]">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-[10px]">
              <h2 className="m-0 truncate text-[20px] font-medium text-ink">{title}</h2>
              {titleAdornment}
            </div>
            {subtitle && <div className="mt-[4px] text-[12px] text-muted">{subtitle}</div>}
          </div>
          <CloseButton onClick={onClose} />
        </div>

        <div className="thin-scroll flex-1 overflow-y-auto px-[28px] py-[20px]">{children}</div>

        {footer && <div className="flex shrink-0 gap-[8px] border-t border-line-soft px-[28px] py-[14px]">{footer}</div>}
      </aside>
    </>,
    document.body,
  );
}
