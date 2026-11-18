import type { ReactNode } from 'react';
import { ArrowDown, ArrowUp, Video, VideoOff } from 'lucide-react';
import { clipLabel } from '../../data/clipId';

// ── PageHeader (Figma: "Page header") ───────────────────────────────────────
// 22px medium title, 13px muted subtitle, actions on the right.

export function PageHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex w-full items-center gap-[12px]">
      <div className="flex min-w-0 flex-1 flex-col gap-[2px]">
        <h1 className="m-0 truncate text-[22px] font-medium text-ink">{title}</h1>
        {subtitle && <p className="m-0 truncate text-[13px] text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-[8px]">{actions}</div>}
    </div>
  );
}

// ── SectionTitle ────────────────────────────────────────────────────────────

export function SectionTitle({ children, meta, action }: { children: ReactNode; meta?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex w-full items-center gap-[8px]">
      <p className="m-0 text-[15px] font-medium text-ink">{children}</p>
      {meta && <p className="m-0 text-[12px] text-muted">{meta}</p>}
      <div className="flex-1" />
      {action}
    </div>
  );
}

// ── FieldLabel — 11px muted caption above a value ──────────────────────────

export function FieldLabel({ children }: { children: ReactNode }) {
  return <p className="m-0 text-[11px] font-medium text-muted">{children}</p>;
}

// ── BarMeter (Figma: "bar-track" / "value") ────────────────────────────────

export function BarMeter({ pct, className = '', tone = 'brand' }: { pct: number; className?: string; tone?: 'brand' | 'soft' }) {
  return (
    <div className={`h-[5px] overflow-hidden rounded-[2.5px] ${tone === 'soft' ? 'bg-brand-mid' : 'bg-track'} ${className}`}>
      <div className="h-full rounded-[2.5px] bg-brand transition-[width] duration-300" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

// ── ClipThumb ──────────────────────────────────────────────────────────────
// Stand-in for CCTV footage until real thumbnails are wired up.

const THUMB_TINTS = [
  'linear-gradient(135deg, #2a3550 0%, #1b2334 100%)',
  'linear-gradient(135deg, #34405c 0%, #222b40 100%)',
  'linear-gradient(135deg, #2c3a4f 0%, #1d2638 100%)',
  'linear-gradient(135deg, #313a52 0%, #1b2334 100%)',
];

interface ClipThumbProps {
  seed?: number;
  className?: string;
  offline?: boolean;
  offlineLabel?: string;
  iconSize?: number;
  children?: ReactNode;
  src?: string;
}

export function ClipThumb({ seed = 0, className = '', offline = false, offlineLabel, iconSize = 18, children, src }: ClipThumbProps) {
  return (
    <div
      className={`relative flex items-center justify-center overflow-hidden ${className}`}
      style={{ background: offline ? 'var(--color-dark)' : THUMB_TINTS[Math.abs(seed) % THUMB_TINTS.length] }}
    >
      {src && !offline ? (
        <img src={src} alt="" className="absolute inset-0 size-full object-cover" onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
      ) : null}
      {offline ? (
        offlineLabel ? (
          <span className="text-[12px] text-faint">{offlineLabel}</span>
        ) : (
          <VideoOff size={iconSize} className="text-faint" aria-hidden />
        )
      ) : (
        <Video size={iconSize} className="text-white/25" aria-hidden />
      )}
      {children}
    </div>
  );
}

// ── ClipId — the clip's ID number, e.g. "#012" ──────────────────────────────
// "text" sits inline in meta lines; "chip" is a small bordered tag for titles;
// "overlay" sits on a thumbnail or video.

export function ClipId({ id, variant = 'text', className = '' }: { id: string; variant?: 'text' | 'chip' | 'overlay'; className?: string }) {
  const label = clipLabel(id);
  const styles = {
    text: 'text-muted',
    chip: 'rounded-[4px] border border-line bg-head px-[6px] py-[1px] text-ink-2',
    overlay: 'rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[6px] py-[2px] text-white',
  }[variant];
  return (
    <span title={`Clip ID ${label}`} className={`shrink-0 text-[11px] font-medium whitespace-nowrap tabular-nums ${styles} ${className}`}>
      {label}
    </span>
  );
}

// ── EmptyState ─────────────────────────────────────────────────────────────

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="w-full py-[48px] text-center text-[13px] text-muted">{children}</div>;
}

// ── Kpi (Figma: "KPI · …") ────────────────────────────────────────────────────

export type Trend = 'down-bad' | 'down-good' | 'up-bad' | 'up-good';

export function Kpi({ value, extra, label, trend, note, first = false }: { value: string; extra?: string; label: string; trend: Trend; note: string; first?: boolean }) {
  const up = trend.startsWith('up');
  const good = trend.endsWith('good');
  const Arrow = up ? ArrowUp : ArrowDown;
  return (
    <div className={`flex min-w-0 flex-1 items-center gap-[12px] pr-[16px] ${first ? '' : 'border-l border-line-soft pl-[20px]'}`}>
      <div className="flex shrink-0 items-baseline gap-[3px]">
        <span className="text-[24px] font-semibold text-ink">{value}</span>
        {extra && <span className="text-[13px] text-muted">{extra}</span>}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-px">
        <span className="truncate text-[12px] font-medium text-ink-2">{label}</span>
        <span className="flex min-w-0 items-center gap-[3px]">
          <Arrow size={10} strokeWidth={3} className="shrink-0" style={{ color: good ? 'var(--color-success)' : 'var(--color-danger)' }} />
          <span className="truncate text-[11px] text-muted">{note}</span>
        </span>
      </div>
    </div>
  );
}
