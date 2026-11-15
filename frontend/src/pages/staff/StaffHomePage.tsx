import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CircleX,
  TriangleAlert,
  CircleCheck,
  MapPin,
  Clock,
  UserRound,
  Rewind,
  FastForward,
  Pause,
  Play,
  Sparkles,
  PanelRightClose,
  PanelRightOpen,
  ArrowRight,
  LayoutGrid,
  Inbox,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { StatusPill } from '../../components/ui/Badge';
import { SEVERITY_TONE, SEVERITY_DOT } from '../../components/ui/tones';
import { Button } from '../../components/ui/Button';
import { BarMeter, ClipThumb, FieldLabel, Kpi, SectionTitle } from '../../components/ui/Primitives';
import { Toggle } from '../../components/ui/Toggle';
import { MOCK_CAMERAS } from '../../data/mock';
import DismissModal from '../../components/modals/DismissModal';
import EscalateModal from '../../components/modals/EscalateModal';
import type { Incident, IncidentType, Severity } from '../../types';

// ── Data ──────────────────────────────────────────────────────────────────────

interface QueueItem {
  id: string;
  type: IncidentType;
  location: string;
  severity: Severity;
  cam: string;
  time: string;
  confidence: number;
  place: string;
  window: string;
  duration: string;
  durationSec: number;
  people: string;
  description: string;
  reason: string;
  rule: string;
  stamp: string;
  flagged: [number, number]; // flagged segment as fractions of the clip
  box: { left: string; top: string; width: string; height: string };
}

const QUEUE: QueueItem[] = [
  {
    id: 'q-001', type: 'Smoking', location: 'Carpark B2', severity: 'Critical', cam: 'Cam 04', time: '11:18 PM', confidence: 92,
    place: 'Carpark B2, by the stair door', window: 'Mon 11 Oct, 11:18 – 11:21 PM', duration: '3 min 32 s', durationSec: 212,
    people: '1 adult · Grey hoodie, dark cap',
    description: 'Person smoking a cigarette next to the staircase door, ash dropped on the floor.',
    reason: 'Repeated hand-to-mouth motion with visible smoke in a no-smoking zone.',
    rule: 'Carpark B2 · No Smoking · 10 PM – 6 AM', stamp: '23:18:48', flagged: [0.4, 0.85],
    box: { left: '65%', top: '54%', width: '10.6%', height: '33.7%' },
  },
  {
    id: 'q-002', type: 'Unattended item', location: 'Meeting Room 1', severity: 'Moderate', cam: 'Cam 01', time: '10:15 AM', confidence: 81,
    place: 'Meeting Room 1, beside the door', window: 'Mon 11 Oct, 10:15 – 10:31 AM', duration: '16 min 04 s', durationSec: 964,
    people: 'No one in frame',
    description: 'Laptop bag left on a chair after the 9:30 AM meeting ended.',
    reason: 'Object stationary for more than 15 minutes with no owner in view.',
    rule: 'Meeting rooms · Unattended item · All day', stamp: '10:15:22', flagged: [0.3, 1],
    box: { left: '38%', top: '58%', width: '12%', height: '18%' },
  },
  {
    id: 'q-003', type: 'Loitering', location: 'Corridor 2', severity: 'Minor', cam: 'Cam 03', time: '11:02 AM', confidence: 74,
    place: 'Corridor 2, by the fire exit', window: 'Mon 11 Oct, 11:02 – 11:14 AM', duration: '12 min 10 s', durationSec: 730,
    people: '1 adult · Dark jacket, backpack',
    description: 'Person waiting by the fire exit, pacing between the exit and the lift.',
    reason: 'Same person in a restricted corridor for more than 10 minutes.',
    rule: 'Corridor 2 · Loitering · 10 min', stamp: '11:02:05', flagged: [0.2, 0.9],
    box: { left: '52%', top: '30%', width: '9%', height: '45%' },
  },
  {
    id: 'q-004', type: 'Littering', location: 'Lift Lobby 1', severity: 'Minor', cam: 'Cam 05', time: '8:37 PM', confidence: 68,
    place: 'Lift Lobby 1, outside lift B', window: 'Mon 11 Oct, 8:37 – 8:38 PM', duration: '0 min 48 s', durationSec: 48,
    people: '1 adult · White shirt',
    description: 'Tissue paper dropped outside the lift doors and left behind.',
    reason: 'Object dropped and person left the frame without picking it up.',
    rule: 'Lobbies · Littering · All day', stamp: '20:37:14', flagged: [0.35, 0.6],
    box: { left: '44%', top: '40%', width: '10%', height: '40%' },
  },
];

const HOTSPOTS = [
  { name: 'Carpark B2', count: 22 },
  { name: 'Lift Lobby 1', count: 16 },
  { name: 'Corridor 2', count: 11 },
];

const SPEEDS = [1, 1.5, 2];
const TICKS = 64;

function toIncident(q: QueueItem): Incident {
  return {
    id: q.id,
    camId: 'cam-' + q.cam.replace('Cam ', ''),
    location: q.location,
    type: q.type,
    severity: q.severity,
    dateTime: '2026-10-11T23:18:00',
    outcome: 'Pending',
    confidence: q.confidence / 100,
    description: q.description,
  };
}

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// ── Side-panel widgets ───────────────────────────────────────────────────────

type WidgetId = 'queue' | 'hotspots' | 'cameras';
interface WidgetConfig { id: WidgetId; visible: boolean }

const WIDGET_META: Record<WidgetId, { label: string; hint: string }> = {
  queue:    { label: 'Review queue',  hint: 'Clips waiting for your decision' },
  hotspots: { label: 'Hotspots',      hint: 'Places with the most incidents' },
  cameras:  { label: 'Camera status', hint: 'Cameras that are offline' },
};

const DEFAULT_LAYOUT: WidgetConfig[] = [
  { id: 'queue', visible: true },
  { id: 'hotspots', visible: true },
  { id: 'cameras', visible: false },
];

const LAYOUT_KEY = 'insight.staffHomeWidgets';

function readLayout(): WidgetConfig[] {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    if (!raw) return DEFAULT_LAYOUT;
    const saved = JSON.parse(raw) as WidgetConfig[];
    const known = saved.filter((w) => w.id in WIDGET_META);
    // append any widgets added since the layout was saved
    const missing = DEFAULT_LAYOUT.filter((d) => !known.some((w) => w.id === d.id));
    return [...known, ...missing];
  } catch {
    return DEFAULT_LAYOUT;
  }
}

function WidgetEditor({ layout, onChange, onReset, onDone }: { layout: WidgetConfig[]; onChange: (l: WidgetConfig[]) => void; onReset: () => void; onDone: () => void }) {
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= layout.length) return;
    const next = [...layout];
    [next[i], next[j]] = [next[j], next[i]];
    onChange(next);
  };
  const toggle = (i: number, visible: boolean) => onChange(layout.map((w, k) => (k === i ? { ...w, visible } : w)));

  return (
    <div className="flex flex-col gap-[10px]">
      <div className="pb-[4px]">
        <SectionTitle meta="Show, hide and reorder">Edit widgets</SectionTitle>
      </div>
      <ul className="m-0 flex list-none flex-col gap-[8px] p-0">
        {layout.map((w, i) => (
          <li key={w.id} className="flex items-center gap-[10px] rounded-[8px] border border-line bg-white p-[10px]">
            <span className="flex flex-col">
              <button type="button" onClick={() => move(i, -1)} disabled={i === 0} aria-label={`Move ${WIDGET_META[w.id].label} up`} className="flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink disabled:cursor-default disabled:text-line">
                <ChevronUp size={16} />
              </button>
              <button type="button" onClick={() => move(i, 1)} disabled={i === layout.length - 1} aria-label={`Move ${WIDGET_META[w.id].label} down`} className="flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink disabled:cursor-default disabled:text-line">
                <ChevronDown size={16} />
              </button>
            </span>
            <span className="min-w-0 flex-1">
              <span className={`block text-[13px] font-medium ${w.visible ? 'text-ink' : 'text-faint'}`}>{WIDGET_META[w.id].label}</span>
              <span className="block text-[11px] text-muted">{WIDGET_META[w.id].hint}</span>
            </span>
            <Toggle checked={w.visible} onChange={(v) => toggle(i, v)} label={`Show ${WIDGET_META[w.id].label}`} />
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

function ViewAll({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className="flex cursor-pointer items-center gap-[4px] border-none bg-transparent p-0 text-[12px] font-medium text-brand hover:underline">
      View all <ArrowRight size={14} />
    </button>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

type Toast = { text: string; item: QueueItem; index: number } | null;

export default function StaffHomePage() {
  const navigate = useNavigate();
  const [queue, setQueue] = useState<QueueItem[]>(QUEUE);
  const [selectedId, setSelectedId] = useState<string>(QUEUE[0].id);
  const [panelOpen, setPanelOpen] = useState(true);
  const [modal, setModal] = useState<'dismiss' | 'escalate' | null>(null);
  const [toast, setToast] = useState<Toast>(null);
  const [escalatedToday, setEscalatedToday] = useState(0);
  const [editing, setEditing] = useState(false);
  const [layout, setLayout] = useState<WidgetConfig[]>(readLayout);

  useEffect(() => {
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch { /* storage unavailable */ }
  }, [layout]);

  const visibleWidgets = layout.filter((w) => w.visible);
  const offlineCams = MOCK_CAMERAS.filter((c) => c.status === 'offline');
  const onlineCams = MOCK_CAMERAS.length - offlineCams.length;

  const featured = queue.find((q) => q.id === selectedId) ?? queue[0] ?? null;
  const featuredIncident = useMemo(() => (featured ? toIncident(featured) : null), [featured]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(t);
  }, [toast]);

  function resolve(label: string) {
    if (!featured) return;
    const index = queue.findIndex((q) => q.id === featured.id);
    const next = queue.filter((q) => q.id !== featured.id);
    setQueue(next);
    setSelectedId(next[Math.min(index, next.length - 1)]?.id ?? '');
    setToast({ text: `${label} · ${featured.type} · ${featured.location}`, item: featured, index });
    if (label === 'Escalated') setEscalatedToday((n) => n + 1);
  }

  function undo() {
    if (!toast) return;
    const restored = [...queue];
    restored.splice(toast.index, 0, toast.item);
    setQueue(restored);
    setSelectedId(toast.item.id);
    if (toast.text.startsWith('Escalated')) setEscalatedToday((n) => Math.max(0, n - 1));
    setToast(null);
  }

  const maxHotspot = Math.max(...HOTSPOTS.map((h) => h.count));

  const closeBtn = (
    <button type="button" onClick={() => setPanelOpen(false)} aria-label="Hide side panel" className="flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink">
      <PanelRightClose size={16} />
    </button>
  );

  return (
    <AppLayout role="staff" activePath="/staff/home" variant="bare">
      {/* ── Main column ─────────────────────────────────────────────── */}
      <section className="thin-scroll flex min-w-0 flex-1 flex-col gap-[16px] overflow-y-auto px-[28px] pt-[20px] pb-[24px]">
        {/* KPIs */}
        <div className="flex w-full shrink-0 items-start border-b border-line-soft pb-[16px]">
          <Kpi value="6" extra="/ 8" label="Cameras online" trend="down-bad" note="2 offline since 6:12 PM" first />
          <Kpi value={String(escalatedToday)} label="Escalated today" trend={escalatedToday === 0 ? 'down-good' : 'up-bad'} note={escalatedToday === 0 ? '2 fewer than yesterday' : `${escalatedToday} since you signed in`} />
          <Kpi value={String(queue.length)} label="To review" trend="up-bad" note="1 more than this time yesterday" />
        </div>

        {featured ? (
          <>
            {/* Incident header */}
            <div className="flex w-full shrink-0 items-center gap-[10px]">
              <h1 className="m-0 truncate text-[22px] font-medium text-ink">{featured.type} · {featured.location}</h1>
              <StatusPill tone={SEVERITY_TONE[featured.severity]}>{featured.severity}</StatusPill>
              <div className="flex-1" />
              <Button variant="secondary" icon={<CircleX size={16} strokeWidth={1.75} />} onClick={() => setModal('dismiss')}>Dismiss</Button>
              <Button variant="danger" icon={<TriangleAlert size={16} strokeWidth={1.75} />} onClick={() => setModal('escalate')}>Escalate</Button>
              <Button variant="primary" icon={<CircleCheck size={16} strokeWidth={1.75} />} onClick={() => resolve('Accepted')}>Accept</Button>
            </div>

            {/* Facts */}
            <div className="flex w-full shrink-0 flex-wrap items-center gap-x-[20px] gap-y-[6px] text-[12px] text-brand-faded">
              <Fact icon={<MapPin size={14} />}>{featured.place} · {featured.cam}</Fact>
              <Fact icon={<Clock size={14} />}>{featured.window} · {featured.duration}</Fact>
              <Fact icon={<UserRound size={14} />}>{featured.people}</Fact>
            </div>

            <ClipPlayer key={featured.id} item={featured} />

            {/* Details */}
            <div className="flex w-full shrink-0 items-start gap-[32px] border-t border-line-soft pt-[16px]">
              <div className="flex min-w-0 flex-1 flex-col gap-[6px]">
                <FieldLabel>Description</FieldLabel>
                <p className="m-0 text-[13px] text-ink">{featured.description}</p>
              </div>
              <div className="flex min-w-0 flex-1 flex-col gap-[6px]">
                <div className="flex w-full items-center gap-[6px]">
                  <Sparkles size={12} className="shrink-0 text-brand" aria-hidden />
                  <FieldLabel>Reason for flagging</FieldLabel>
                  <span className="ml-[4px] inline-flex items-center gap-[8px] rounded-full bg-brand-soft px-[10px] py-[4px]">
                    <BarMeter pct={featured.confidence} tone="soft" className="w-[44px]" />
                    <span className="text-[11px] font-semibold text-brand">{featured.confidence}% confidence</span>
                  </span>
                </div>
                <p className="m-0 text-[13px] text-ink">{featured.reason}</p>
                <div className="flex items-center gap-[6px]">
                  <span className="text-[11px] text-muted">Matched rule</span>
                  <span className="text-[12px] font-medium text-brand">{featured.rule}</span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-[8px] py-[80px] text-center">
            <Inbox size={28} className="text-faint" />
            <p className="m-0 text-[15px] font-medium text-ink">You're all caught up</p>
            <p className="m-0 text-[13px] text-muted">New clips will show here as soon as they're flagged.</p>
          </div>
        )}
      </section>

      {/* ── Side panel ──────────────────────────────────────────────── */}
      {panelOpen ? (
        <aside className="thin-scroll flex w-[340px] shrink-0 flex-col gap-[10px] overflow-y-auto border-l border-line bg-head p-[20px]">
          {editing ? (
            <WidgetEditor
              layout={layout}
              onChange={setLayout}
              onReset={() => setLayout(DEFAULT_LAYOUT)}
              onDone={() => setEditing(false)}
            />
          ) : (
            <>
              {visibleWidgets.length === 0 && (
                <div className="flex items-center justify-between">
                  <p className="m-0 text-[12px] text-muted">No widgets shown.</p>
                  {closeBtn}
                </div>
              )}
              {visibleWidgets.map((w, i) => (
                <div key={w.id} className={`flex flex-col gap-[10px] ${i > 0 ? 'mt-[4px] border-t border-line pt-[18px]' : ''}`}>
                  {w.id === 'queue' && (
                    <>
                      <div className="pb-[4px]">
                        <SectionTitle meta={`${queue.length} pending`} action={i === 0 ? closeBtn : undefined}>Review queue</SectionTitle>
                      </div>
                      {queue.length === 0 && <p className="m-0 py-[8px] text-[12px] text-muted">Nothing waiting for review.</p>}
                      {queue.map((item, qi) => {
                        const selected = item.id === featured?.id;
                        return (
                          <button
                            key={item.id}
                            type="button"
                            onClick={() => setSelectedId(item.id)}
                            aria-current={selected ? 'true' : undefined}
                            className={`flex w-full cursor-pointer items-center gap-[10px] rounded-[8px] border p-[8px] text-left transition-colors ${
                              selected ? 'border-brand bg-white' : 'border-transparent bg-transparent hover:bg-white'
                            }`}
                          >
                            <ClipThumb seed={qi} className="h-[42px] w-[60px] shrink-0 rounded-[4px]" iconSize={14} />
                            <span className="flex min-w-0 flex-1 flex-col gap-[3px]">
                              <span className="truncate text-[12px] font-medium text-ink">{item.type} · {item.location}</span>
                              <span className="flex items-center gap-[6px] text-[11px] text-muted">
                                <span className="size-[7px] shrink-0 rounded-full" style={{ backgroundColor: SEVERITY_DOT[item.severity] }} />
                                {item.cam} · {item.time} · {item.confidence}%
                              </span>
                            </span>
                          </button>
                        );
                      })}
                    </>
                  )}

                  {w.id === 'hotspots' && (
                    <>
                      <div className="pb-[4px]">
                        <SectionTitle
                          action={
                            <span className="flex items-center gap-[12px]">
                              <ViewAll onClick={() => navigate('/staff/hotspots')} />
                              {i === 0 && closeBtn}
                            </span>
                          }
                        >
                          Hotspots
                        </SectionTitle>
                      </div>
                      {HOTSPOTS.map((h) => (
                        <button
                          key={h.name}
                          type="button"
                          onClick={() => navigate('/staff/hotspots')}
                          className="flex w-full cursor-pointer items-center gap-[10px] border-none bg-transparent p-0 text-left"
                        >
                          <span className="w-[86px] shrink-0 truncate text-[12px] text-ink-2">{h.name}</span>
                          <BarMeter pct={(h.count / maxHotspot) * 100} className="flex-1" />
                          <span className="w-[24px] shrink-0 text-right text-[12px] font-medium text-ink">{h.count}</span>
                        </button>
                      ))}
                    </>
                  )}

                  {w.id === 'cameras' && (
                    <>
                      <div className="pb-[4px]">
                        <SectionTitle
                          meta={`${onlineCams} of ${MOCK_CAMERAS.length} online`}
                          action={
                            <span className="flex items-center gap-[12px]">
                              <ViewAll onClick={() => navigate('/staff/cameras')} />
                              {i === 0 && closeBtn}
                            </span>
                          }
                        >
                          Camera status
                        </SectionTitle>
                      </div>
                      {offlineCams.length === 0 ? (
                        <p className="m-0 text-[12px] text-muted">All cameras are online.</p>
                      ) : (
                        offlineCams.map((c) => (
                          <button
                            key={c.id}
                            type="button"
                            onClick={() => navigate('/staff/cameras')}
                            className="flex w-full cursor-pointer items-center gap-[8px] border-none bg-transparent p-0 text-left"
                          >
                            <span className="size-[7px] shrink-0 rounded-full bg-danger" />
                            <span className="min-w-0 flex-1 truncate text-[12px] text-ink">{c.name} · {c.location}</span>
                            <span className="shrink-0 text-[11px] text-muted">Offline</span>
                          </button>
                        ))
                      )}
                    </>
                  )}
                </div>
              ))}

              <div className="flex-1" />
              <button
                type="button"
                onClick={() => setEditing(true)}
                className="flex w-full cursor-pointer items-center justify-center gap-[6px] rounded-[6px] border-none bg-transparent py-[8px] text-[12px] font-medium text-ink-2 hover:bg-white"
              >
                <LayoutGrid size={14} /> Edit widgets
              </button>
            </>
          )}
        </aside>
      ) : (
        <aside className="flex w-[48px] shrink-0 flex-col items-center border-l border-line bg-head pt-[20px]">
          <button type="button" onClick={() => setPanelOpen(true)} aria-label="Show side panel" className="relative flex cursor-pointer border-none bg-transparent p-0 text-ink-2 hover:text-ink">
            <PanelRightOpen size={16} />
            {queue.length > 0 && (
              <span className="absolute -top-[8px] -right-[10px] flex size-[16px] items-center justify-center rounded-full bg-brand text-[10px] font-semibold text-white">{queue.length}</span>
            )}
          </button>
        </aside>
      )}

      {/* Toast */}
      {toast && (
        <div role="status" className="v2-pop-in fixed bottom-[24px] left-1/2 z-[400] flex -translate-x-1/2 items-center gap-[16px] rounded-[8px] bg-dark py-[10px] pr-[10px] pl-[16px] text-[13px] text-white shadow-[0_12px_32px_rgba(18,26,43,0.3)]">
          <span>{toast.text}</span>
          <button type="button" onClick={undo} className="cursor-pointer rounded-[4px] border-none bg-white/15 px-[10px] py-[4px] text-[12px] font-medium text-white hover:bg-white/25">
            Undo
          </button>
        </div>
      )}

      <DismissModal isOpen={modal === 'dismiss'} onClose={() => setModal(null)} incident={featuredIncident} onConfirm={() => resolve('Dismissed')} />
      <EscalateModal isOpen={modal === 'escalate'} onClose={() => setModal(null)} incident={featuredIncident} onConfirm={() => resolve('Escalated')} />
    </AppLayout>
  );
}

// ── ClipPlayer ────────────────────────────────────────────────────────────────

function ClipPlayer({ item }: { item: QueueItem }) {
  const [playing, setPlaying] = useState(true);
  const [pos, setPos] = useState(item.durationSec * 0.2);
  const [speedIdx, setSpeedIdx] = useState(0);
  const speed = SPEEDS[speedIdx];

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setPos((p) => {
        const next = p + 0.25 * speed;
        if (next >= item.durationSec) { setPlaying(false); return item.durationSec; }
        return next;
      });
    }, 250);
    return () => clearInterval(id);
  }, [playing, speed, item.durationSec]);

  const pct = (pos / item.durationSec) * 100;
  const seek = (frac: number) => setPos(Math.max(0, Math.min(item.durationSec, frac * item.durationSec)));
  const [fs, fe] = item.flagged;

  return (
    <>
      {/* Video */}
      <ClipThumb seed={0} className="aspect-[16/9] w-full shrink-0 rounded-[6px]" iconSize={40}>
        <span className="absolute top-[12px] left-[12px] rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[8px] py-[4px] text-[11px] font-medium text-white">
          {item.cam.toUpperCase()} · {item.stamp}
        </span>
        {/* Detection */}
        <div className="absolute rounded-[2px] border-2 border-[#7fb2ff]" style={item.box} />
        <span
          className="absolute -translate-y-full rounded-[3px] bg-brand px-[6px] py-[2px] text-[10px] font-medium text-white"
          style={{ left: item.box.left, top: `calc(${item.box.top} - 4px)` }}
        >
          {item.type}
        </span>
        {/* Controls */}
        <div className="absolute right-[12px] bottom-[12px] left-[12px] flex h-[36px] items-center gap-[12px] rounded-[6px] bg-[rgba(27,35,52,0.8)] px-[12px] text-white">
          <PlayerBtn label="Back 10 seconds" onClick={() => setPos((p) => Math.max(0, p - 10))}><Rewind size={16} fill="currentColor" /></PlayerBtn>
          <PlayerBtn label={playing ? 'Pause' : 'Play'} onClick={() => { if (pos >= item.durationSec) setPos(0); setPlaying((v) => !v); }}>
            {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
          </PlayerBtn>
          <PlayerBtn label="Forward 10 seconds" onClick={() => setPos((p) => Math.min(item.durationSec, p + 10))}><FastForward size={16} fill="currentColor" /></PlayerBtn>
          <div
            role="slider"
            tabIndex={0}
            aria-label="Seek"
            aria-valuemin={0}
            aria-valuemax={item.durationSec}
            aria-valuenow={Math.round(pos)}
            onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek((e.clientX - r.left) / r.width); }}
            onKeyDown={(e) => { if (e.key === 'ArrowRight') setPos((p) => Math.min(item.durationSec, p + 5)); if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 5)); }}
            className="flex h-[12px] min-w-0 flex-1 cursor-pointer items-center"
          >
            <div className="h-[4px] w-full overflow-hidden rounded-[2px] bg-white/30">
              <div className="h-full bg-white" style={{ width: `${pct}%` }} />
            </div>
          </div>
          <span className="shrink-0 text-[11px] font-medium tabular-nums">{fmt(pos)} / {fmt(item.durationSec)}</span>
          <button
            type="button"
            onClick={() => setSpeedIdx((i) => (i + 1) % SPEEDS.length)}
            aria-label="Playback speed"
            className="shrink-0 cursor-pointer rounded-[4px] border-none bg-white/20 px-[6px] py-[2px] text-[11px] font-medium text-white hover:bg-white/30"
          >
            {speed}x
          </button>
        </div>
      </ClipThumb>

      {/* Clip timeline */}
      <div className="relative h-[58px] w-full shrink-0">
        <button
          type="button"
          aria-label="Clip timeline"
          onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek((e.clientX - r.left) / r.width); }}
          className="absolute inset-x-0 top-0 flex h-[34px] cursor-pointer items-center justify-between border-none bg-transparent p-0"
        >
          <span className="absolute top-[4px] bottom-[4px] rounded-[2px] bg-brand-soft" style={{ left: `${fs * 100}%`, width: `${(fe - fs) * 100}%` }} />
          {Array.from({ length: TICKS }).map((_, i) => {
            const f = i / (TICKS - 1);
            const inFlag = f >= fs && f <= fe;
            return (
              <span
                key={i}
                className="relative w-px"
                style={{ height: i % 8 === 0 ? 20 : 12, backgroundColor: inFlag ? 'var(--color-brand)' : 'var(--color-line)', opacity: inFlag ? 0.7 : 1 }}
              />
            );
          })}
          <span className="pointer-events-none absolute top-0 bottom-0 w-[2px] -translate-x-1/2 bg-brand" style={{ left: `${pct}%` }}>
            <span className="absolute -top-[3px] left-1/2 size-[7px] -translate-x-1/2 rounded-full bg-brand" />
          </span>
        </button>
        <span className="absolute top-[40px] left-[6px] text-[10px] text-muted">Start</span>
        <span className="absolute top-[40px] text-[10px] font-medium text-brand" style={{ left: `${fs * 100}%` }}>Flagged segment</span>
        <span className="absolute top-[40px] right-[6px] text-[10px] text-muted">{fmt(item.durationSec)}</span>
      </div>
    </>
  );
}

function PlayerBtn({ children, label, onClick }: { children: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button type="button" aria-label={label} onClick={onClick} className="flex shrink-0 cursor-pointer border-none bg-transparent p-0 text-white opacity-90 hover:opacity-100">
      {children}
    </button>
  );
}

function Fact({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <span className="flex items-center gap-[6px] whitespace-nowrap">
      <span className="flex">{icon}</span>
      {children}
    </span>
  );
}
