import { useNavigate } from 'react-router-dom';
import { Play, Download, Search as SearchIcon, Clock, Video, Timer } from 'lucide-react';
import { SeverityBadge } from '../ui/Badge';
import { OUTCOME_DOT } from '../ui/tones';
import { Button } from '../ui/Button';
import { Drawer } from '../ui/Drawer';
import { BarMeter, ClipThumb, FieldLabel } from '../ui/Primitives';
import type { Incident } from '../../types';
import { MOCK_STAFF } from '../../data/mock';

interface ClipDetailDrawerProps {
  inc: Incident | null;
  onClose: () => void;
}

const STAFF_MAP: Record<string, string> = Object.fromEntries(MOCK_STAFF.map((s) => [s.id, s.name]));

const camLabel = (camId: string) => 'Cam ' + camId.replace('cam-', '').padStart(2, '0');

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}, ${d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`;
}

/** Approximate "reviewed at" = incident time + 3 min */
function reviewedAt(iso: string) {
  return new Date(new Date(iso).getTime() + 3 * 60 * 1000).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

export default function ClipDetailDrawer({ inc, onClose }: ClipDetailDrawerProps) {
  const navigate = useNavigate();
  if (!inc) return null;

  const assignee = inc.assignedTo ? STAFF_MAP[inc.assignedTo] ?? '—' : '—';
  const confidence = Math.round(inc.confidence * 100);

  let remarkText: string | null = null;
  if (inc.remark) remarkText = inc.outcome === 'Accepted' ? inc.remark : `“${inc.remark}”`;
  else if (inc.outcome === 'Accepted') remarkText = 'Logged as an incident. No remark needed.';

  return (
    <Drawer
      isOpen
      onClose={onClose}
      title={`${inc.type} · ${inc.location}`}
      titleAdornment={<SeverityBadge severity={inc.severity} />}
      footer={
        <>
          <Button variant="secondary" className="flex-1" icon={<Download size={16} strokeWidth={1.75} />}>Download clip</Button>
          <Button
            variant="secondary"
            className="flex-1"
            icon={<SearchIcon size={16} strokeWidth={1.75} />}
            onClick={() => { onClose(); navigate(`/staff/search?q=${encodeURIComponent(inc.location)}`); }}
          >
            Find similar clips
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-[20px]">
        {/* Video: the incident's own clip when the API has one */}
        {inc.clipUrl ? (
          <video
            key={inc.clipUrl}
            src={inc.clipUrl}
            poster={inc.thumbnailUrl}
            controls
            loop
            muted
            playsInline
            className="aspect-[16/9] w-full rounded-[6px] bg-dark object-contain"
          />
        ) : (
        <ClipThumb seed={inc.id.length} src={inc.thumbnailUrl} className="aspect-[16/9] w-full rounded-[6px]" iconSize={0}>
          <span className="absolute top-[10px] left-[10px] rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[8px] py-[4px] text-[11px] font-medium text-white">
            {camLabel(inc.camId).toUpperCase()}
          </span>
          <button
            type="button"
            aria-label="Play clip"
            className="flex size-[48px] cursor-pointer items-center justify-center rounded-full border-none bg-[rgba(27,35,52,0.8)] text-white transition-transform hover:scale-105"
          >
            <Play size={20} fill="currentColor" className="ml-[2px]" />
          </button>
        </ClipThumb>
        )}

        {/* Facts */}
        <div className="flex flex-wrap gap-x-[20px] gap-y-[6px] text-[12px] text-ink-2">
          <span className="flex items-center gap-[6px]"><Video size={14} />{camLabel(inc.camId)}</span>
          <span className="flex items-center gap-[6px]"><Clock size={14} />{formatDateTime(inc.dateTime)}</span>
          <span className="flex items-center gap-[6px]"><Timer size={14} />{inc.duration ?? '—'}</span>
        </div>

        <div className="flex flex-col gap-[6px] border-t border-line-soft pt-[16px]">
          <FieldLabel>Description</FieldLabel>
          <p className="m-0 text-[13px] leading-[1.5] text-ink">{inc.description}</p>
        </div>

        <div className="flex items-center gap-[10px]">
          <FieldLabel>AI confidence</FieldLabel>
          <BarMeter pct={confidence} className="w-[80px]" />
          <span className="text-[12px] font-semibold text-brand">{confidence}%</span>
        </div>

        {/* Outcome */}
        <div className="flex flex-col gap-[8px] rounded-[8px] border border-line bg-head px-[16px] py-[14px]">
          <div className="flex items-center gap-[8px]">
            <span className="size-[7px] shrink-0 rounded-full" style={{ backgroundColor: OUTCOME_DOT[inc.outcome] }} />
            <span className="text-[13px] font-medium text-ink">{inc.outcome}</span>
            <span className="h-px flex-1 bg-line" />
            {assignee !== '—' && <span className="text-[12px] text-muted">{assignee} at {reviewedAt(inc.dateTime)}</span>}
          </div>
          {remarkText && <p className="m-0 text-[13px] leading-[1.5] text-ink-2">{remarkText}</p>}
          {inc.outcome === 'Escalated' && <p className="m-0 text-[12px] font-medium text-brand">Open · waiting on admin</p>}
        </div>
      </div>
    </Drawer>
  );
}
