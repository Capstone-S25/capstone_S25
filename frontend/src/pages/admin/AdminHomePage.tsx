import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CircleCheck, MapPin, Clock, UserRound, Play, ArrowRight, Sparkles, MessageSquareQuote, Inbox } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { StatusPill } from '../../components/ui/Badge';
import { SEVERITY_TONE, SEVERITY_DOT } from '../../components/ui/tones';
import { Button } from '../../components/ui/Button';
import { BarMeter, ClipThumb, FieldLabel, Kpi, SectionTitle } from '../../components/ui/Primitives';
import { INPUT_CLASS } from '../../components/ui/Modal';
import { MOCK_INCIDENTS, MOCK_STAFF } from '../../data/mock';

const STAFF_MAP: Record<string, string> = Object.fromEntries(MOCK_STAFF.map((s) => [s.id, s.name]));
const camLabel = (camId: string) => 'Cam ' + camId.replace('cam-', '').padStart(2, '0');
const fmtDate = (iso: string) => new Date(iso).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
const fmtTime = (iso: string) => new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });

const REPORTS = [
  { label: '5–11 Oct', dates: '5 Oct – 11 Oct 2026' },
  { label: '1–4 Oct', dates: '1 Oct – 4 Oct 2026' },
];

export default function AdminHomePage() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(() => MOCK_INCIDENTS.filter((i) => i.outcome === 'Escalated'));
  const [resolved, setResolved] = useState(() => MOCK_INCIDENTS.filter((i) => i.outcome === 'Accepted').slice(0, 2));
  const [selectedId, setSelectedId] = useState(open[0]?.id ?? '');
  const [assignments, setAssignments] = useState<Record<string, string>>({});

  const current = open.find((i) => i.id === selectedId) ?? open[0] ?? null;

  function resolveCurrent() {
    if (!current) return;
    const idx = open.findIndex((i) => i.id === current.id);
    const next = open.filter((i) => i.id !== current.id);
    setOpen(next);
    setResolved((r) => [current, ...r].slice(0, 4));
    setSelectedId(next[Math.min(idx, next.length - 1)]?.id ?? '');
  }

  return (
    <AppLayout role="admin" activePath="/admin/home" variant="bare">
      <section className="thin-scroll flex min-w-0 flex-1 flex-col gap-[16px] overflow-y-auto px-[28px] pt-[20px] pb-[24px]">
        <div className="flex w-full shrink-0 items-start border-b border-line-soft pb-[16px]">
          <Kpi first value={String(open.length)} label="Escalated to you" trend="up-bad" note="1 more than yesterday" />
          <Kpi value="9" label="Dismissed this week" trend="up-bad" note="2 more than last week" />
          <Kpi value="6" extra="/ 8" label="Cameras online" trend="down-bad" note="2 offline since 6:12 PM" />
        </div>

        {current ? (
          <>
            <div className="flex w-full shrink-0 items-center gap-[10px]">
              <h1 className="m-0 truncate text-[22px] font-medium text-ink">{current.type} · {current.location}</h1>
              <StatusPill tone={SEVERITY_TONE[current.severity]}>{current.severity}</StatusPill>
              <StatusPill tone={assignments[current.id] ? 'blue' : 'neutral'}>
                {assignments[current.id] ? `Assigned · ${STAFF_MAP[assignments[current.id]]}` : 'Unassigned'}
              </StatusPill>
              <div className="flex-1" />
              <Button variant="primary" icon={<CircleCheck size={16} strokeWidth={1.75} />} onClick={resolveCurrent}>Mark resolved</Button>
            </div>

            <div className="flex w-full shrink-0 flex-wrap items-center gap-x-[20px] gap-y-[6px] text-[12px] text-brand-faded">
              <span className="flex items-center gap-[6px]"><MapPin size={14} />{current.location} · {camLabel(current.camId)}</span>
              <span className="flex items-center gap-[6px]"><Clock size={14} />{fmtDate(current.dateTime)}, {fmtTime(current.dateTime)} · {current.duration ?? '—'}</span>
              <span className="flex items-center gap-[6px]"><UserRound size={14} />Escalated by {current.assignedTo ? STAFF_MAP[current.assignedTo] : 'staff'}</span>
            </div>

            <ClipThumb seed={current.id.length} className="aspect-[16/9] w-full shrink-0 rounded-[6px]" iconSize={0}>
              <span className="absolute top-[12px] left-[12px] rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[8px] py-[4px] text-[11px] font-medium text-white">
                {camLabel(current.camId).toUpperCase()} · {fmtTime(current.dateTime)}
              </span>
              <button type="button" aria-label="Play clip" className="flex size-[56px] cursor-pointer items-center justify-center rounded-full border-none bg-[rgba(27,35,52,0.8)] text-white transition-transform hover:scale-105">
                <Play size={22} fill="currentColor" className="ml-[3px]" />
              </button>
            </ClipThumb>

            <div className="flex w-full shrink-0 items-start gap-[32px] border-t border-line-soft pt-[16px]">
              <div className="flex min-w-0 flex-1 flex-col gap-[6px]">
                <div className="flex items-center gap-[6px]">
                  <Sparkles size={12} className="shrink-0 text-brand" aria-hidden />
                  <FieldLabel>Description</FieldLabel>
                  <span className="ml-[4px] inline-flex items-center gap-[8px] rounded-full bg-brand-soft px-[10px] py-[4px]">
                    <BarMeter pct={current.confidence * 100} tone="soft" className="w-[44px]" />
                    <span className="text-[11px] font-semibold text-brand">{Math.round(current.confidence * 100)}% confidence</span>
                  </span>
                </div>
                <p className="m-0 text-[13px] text-ink">{current.description}</p>
                {current.remark && (
                  <p className="m-0 mt-[4px] flex items-start gap-[6px] text-[12px] text-ink-2">
                    <MessageSquareQuote size={14} className="mt-[1px] shrink-0 text-muted" />“{current.remark}”
                  </p>
                )}
              </div>
              <div className="flex w-[260px] shrink-0 flex-col gap-[6px]">
                <label htmlFor="assign-staff"><FieldLabel>Assign to staff</FieldLabel></label>
                <select
                  id="assign-staff"
                  className={INPUT_CLASS}
                  value={assignments[current.id] ?? ''}
                  onChange={(e) => setAssignments((a) => ({ ...a, [current.id]: e.target.value }))}
                >
                  <option value="">Select staff…</option>
                  {MOCK_STAFF.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-[8px] py-[80px] text-center">
            <Inbox size={28} className="text-faint" />
            <p className="m-0 text-[15px] font-medium text-ink">No open escalations</p>
            <p className="m-0 text-[13px] text-muted">Cases your staff escalate will show here.</p>
          </div>
        )}
      </section>

      <aside className="thin-scroll flex w-[340px] shrink-0 flex-col gap-[10px] overflow-y-auto border-l border-line bg-head p-[20px]">
        <div className="pb-[4px]"><SectionTitle meta={`${open.length} open`}>Escalated cases</SectionTitle></div>
        {open.length === 0 && <p className="m-0 py-[4px] text-[12px] text-muted">Nothing open.</p>}
        {open.map((inc, i) => {
          const selected = inc.id === current?.id;
          return (
            <button
              key={inc.id}
              type="button"
              onClick={() => setSelectedId(inc.id)}
              aria-current={selected ? 'true' : undefined}
              className={`flex w-full cursor-pointer items-center gap-[10px] rounded-[8px] border p-[8px] text-left transition-colors ${selected ? 'border-brand bg-white' : 'border-transparent bg-transparent hover:bg-white'}`}
            >
              <ClipThumb seed={i} className="h-[42px] w-[60px] shrink-0 rounded-[4px]" iconSize={14} />
              <span className="flex min-w-0 flex-1 flex-col gap-[3px]">
                <span className="truncate text-[12px] font-medium text-ink">{inc.type} · {inc.location}</span>
                <span className="flex items-center gap-[6px] text-[11px] text-muted">
                  <span className="size-[7px] shrink-0 rounded-full" style={{ backgroundColor: SEVERITY_DOT[inc.severity] }} />
                  {camLabel(inc.camId)} · {fmtDate(inc.dateTime)} · {Math.round(inc.confidence * 100)}%
                </span>
              </span>
            </button>
          );
        })}

        <div className="mt-[4px] border-t border-line pt-[18px] pb-[4px]"><SectionTitle>Recently resolved</SectionTitle></div>
        {resolved.map((inc) => (
          <div key={inc.id} className="flex items-center gap-[10px] px-[8px] py-[2px]">
            <CircleCheck size={15} className="shrink-0 text-success" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[12px] font-medium text-ink">{inc.type} · {inc.location}</span>
              <span className="block text-[11px] text-muted">{fmtDate(inc.dateTime)}</span>
            </span>
          </div>
        ))}

        <div className="mt-[4px] border-t border-line pt-[18px] pb-[4px]">
          <SectionTitle
            action={
              <button type="button" onClick={() => navigate('/admin/workload')} className="flex cursor-pointer items-center gap-[4px] border-none bg-transparent p-0 text-[12px] font-medium text-brand hover:underline">
                View all <ArrowRight size={14} />
              </button>
            }
          >
            Workload reports
          </SectionTitle>
        </div>
        {REPORTS.map((r) => (
          <button
            key={r.label}
            type="button"
            onClick={() => navigate('/admin/workload')}
            className="flex w-full cursor-pointer items-center gap-[10px] rounded-[6px] border-none bg-transparent px-[8px] py-[6px] text-left transition-colors hover:bg-white"
          >
            <span className="min-w-0 flex-1">
              <span className="block text-[12px] font-medium text-ink">{r.label}</span>
              <span className="block text-[11px] text-muted">{r.dates}</span>
            </span>
            <ArrowRight size={14} className="text-brand" />
          </button>
        ))}
      </aside>
    </AppLayout>
  );
}
