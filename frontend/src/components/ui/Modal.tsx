import { ReactNode, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  subtitle?: string;
  width?: string;
  footer?: ReactNode;
}

export function Modal({ isOpen, onClose, children, title, subtitle, width = '480px', footer }: ModalProps) {
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  // Lock body scroll while open
  useEffect(() => {
    if (!isOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, [isOpen]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[300] flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label={title}>
      <div className="v2-fade-in absolute inset-0 bg-[rgba(18,26,43,0.45)]" onClick={onClose} aria-hidden="true" />

      <div
        className="v2-pop-in relative flex max-h-[90vh] flex-col rounded-[10px] border border-line bg-white shadow-[0_16px_40px_rgba(18,26,43,0.18)]"
        style={{ width, maxWidth: '100%' }}
        onClick={(e) => e.stopPropagation()}
      >
        {title !== undefined ? (
          <div className="flex items-start gap-[12px] border-b border-line-soft px-[24px] pt-[20px] pb-[16px]">
            <div className="min-w-0 flex-1">
              <h2 className="m-0 text-[17px] font-medium text-ink">{title}</h2>
              {subtitle && <p className="m-0 mt-[2px] text-[13px] text-muted">{subtitle}</p>}
            </div>
            <CloseButton onClick={onClose} />
          </div>
        ) : (
          <div className="absolute top-[14px] right-[14px] z-10">
            <CloseButton onClick={onClose} />
          </div>
        )}

        <div className="thin-scroll flex-1 overflow-y-auto px-[24px] py-[20px]">{children}</div>

        {footer && <div className="flex justify-end gap-[8px] border-t border-line-soft px-[24px] py-[14px]">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}

export function CloseButton({ onClick, label = 'Close' }: { onClick: () => void; label?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="flex size-[32px] shrink-0 cursor-pointer items-center justify-center rounded-[6px] border border-line bg-white text-ink-2 transition-colors hover:bg-head"
    >
      <X size={16} />
    </button>
  );
}

// Shared form-control classes for modals and forms
export const INPUT_CLASS =
  'w-full rounded-[6px] border border-line bg-white px-[12px] py-[9px] text-[13px] text-ink placeholder:text-faint outline-none transition-colors hover:border-faint focus:border-brand focus:ring-2 focus:ring-brand-soft';

export const LABEL_CLASS = 'mb-[6px] block text-[12px] font-medium text-ink-2';
