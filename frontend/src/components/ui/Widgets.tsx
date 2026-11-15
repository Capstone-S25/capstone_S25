import { useState, type ReactNode } from 'react';
import { ArrowRight, EllipsisVertical, LayoutGrid, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { Button } from './Button';
import { Toggle } from './Toggle';
import { SectionTitle } from './Primitives';

// Shared building blocks for the editable right-hand side panel on Home pages.

import type { WidgetConfig, WidgetMeta } from './useWidgetLayout';

export type { WidgetConfig, WidgetMeta };

interface SidePanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Number shown on the collapsed rail (e.g. items waiting). */
  badge?: number;
  children: ReactNode;
}

/** Collapsible right-hand panel: 340px open, 48px rail when collapsed. */
export function SidePanel({ open, onOpenChange, badge = 0, children }: SidePanelProps) {
  if (!open) {
    return (
      <aside className="flex w-[48px] shrink-0 flex-col items-center border-l border-line bg-head pt-[20px]">
        <button type="button" onClick={() => onOpenChange(true)} aria-label="Show side panel" className="relative flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink">
          <PanelRightOpen size={16} />
          {badge > 0 && (
            <span className="absolute -top-[8px] -right-[10px] flex size-[16px] items-center justify-center rounded-full bg-brand text-[10px] font-semibold text-white">{badge}</span>
          )}
        </button>
      </aside>
    );
  }
  return (
    <aside className="thin-scroll flex w-[340px] shrink-0 flex-col gap-[10px] overflow-y-auto border-l border-line bg-head p-[20px]">
      {children}
    </aside>
  );
}

export function PanelCloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} aria-label="Hide side panel" className="flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink">
      <PanelRightClose size={16} />
    </button>
  );
}

export function EditWidgetsButton({ onClick }: { onClick: () => void }) {
  return (
    <>
      <div className="flex-1" />
      <button
        type="button"
        onClick={onClick}
        className="flex w-full cursor-pointer items-center justify-center gap-[6px] rounded-[6px] border-none bg-transparent py-[8px] text-[12px] font-medium text-ink-2 hover:bg-white"
      >
        <LayoutGrid size={14} /> Edit widgets
      </button>
    </>
  );
}

export function ViewAll({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className="flex cursor-pointer items-center gap-[4px] border-none bg-transparent p-0 text-[12px] font-medium text-brand hover:underline">
      View all <ArrowRight size={14} />
    </button>
  );
}

/** Wraps each visible widget: divider between widgets after the first. */
export function WidgetSection({ index, children }: { index: number; children: ReactNode }) {
  return <div className={`flex flex-col gap-[10px] ${index > 0 ? 'mt-[4px] border-t border-line pt-[18px]' : ''}`}>{children}</div>;
}

interface WidgetEditorProps<T extends string> {
  layout: WidgetConfig<T>[];
  meta: WidgetMeta<T>;
  onChange: (layout: WidgetConfig<T>[]) => void;
  onReset: () => void;
  onDone: () => void;
}

export function WidgetEditor<T extends string>({ layout, meta, onChange, onReset, onDone }: WidgetEditorProps<T>) {
  // Drag is only armed from the handle, so the toggle and text stay clickable.
  const [armed, setArmed] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  const moveTo = (from: number, to: number) => {
    if (from === to || to < 0 || to >= layout.length) return;
    const next = [...layout];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    onChange(next);
  };
  const toggle = (i: number, visible: boolean) => onChange(layout.map((w, k) => (k === i ? { ...w, visible } : w)));

  return (
    <div className="flex flex-col gap-[10px]">
      <div className="pb-[4px]">
        <SectionTitle meta="Drag to reorder, switch to show">Edit widgets</SectionTitle>
      </div>
      <ul className="m-0 flex list-none flex-col gap-[8px] p-0">
        {layout.map((w, i) => (
          <li
            key={w.id}
            draggable={armed === i}
            onDragStart={(e) => {
              setDragging(i);
              e.dataTransfer.effectAllowed = 'move';
              e.dataTransfer.setData('text/plain', w.id); // required by Firefox
            }}
            onDragOver={(e) => {
              if (dragging === null) return;
              e.preventDefault();
              if (dragging !== i) { moveTo(dragging, i); setDragging(i); setArmed(i); }
            }}
            onDragEnd={() => { setDragging(null); setArmed(null); }}
            className={`flex items-center gap-[10px] rounded-[8px] border bg-white py-[10px] pr-[10px] pl-[4px] transition-shadow ${
              dragging === i ? 'border-brand opacity-70 shadow-[0_6px_16px_rgba(18,26,43,0.12)]' : 'border-line'
            }`}
          >
            <button
              type="button"
              aria-label={`Reorder ${meta[w.id].label}. Use arrow keys to move.`}
              title="Drag to reorder"
              onPointerDown={() => setArmed(i)}
              onPointerUp={() => { if (dragging === null) setArmed(null); }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowUp') { e.preventDefault(); moveTo(i, i - 1); }
                if (e.key === 'ArrowDown') { e.preventDefault(); moveTo(i, i + 1); }
              }}
              className="flex h-[28px] w-[20px] shrink-0 cursor-grab items-center justify-center rounded-[4px] border-none bg-transparent p-0 text-faint hover:bg-head hover:text-ink-2 active:cursor-grabbing"
            >
              <EllipsisVertical size={16} />
            </button>
            <span className="min-w-0 flex-1">
              <span className={`block text-[13px] font-medium ${w.visible ? 'text-ink' : 'text-faint'}`}>{meta[w.id].label}</span>
              <span className="block text-[11px] text-muted">{meta[w.id].hint}</span>
            </span>
            <Toggle checked={w.visible} onChange={(v) => toggle(i, v)} label={`Show ${meta[w.id].label}`} />
          </li>
        ))}
      </ul>
      <div className="mt-[6px] flex gap-[8px]">
        <Button variant="secondary" size="sm" className="flex-1" onClick={onReset}>Reset</Button>
        <Button variant="primary" size="sm" className="flex-1" onClick={onDone}>Done</Button>
      </div>
    </div>
  );
}
