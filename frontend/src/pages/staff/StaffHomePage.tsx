import { useState } from 'react';
import { Play } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { SeverityBadge } from '../../components/ui/Badge';
import { MOCK_INCIDENTS } from '../../data/mock';
import { useIncidents } from '../../data/useIncidents';

const NOTIFICATIONS = [
  { id: 1, bold: 'New alert: ',       text: 'Smoking · Carpark B2',                               time: '2 min ago', unread: true  },
  { id: 2, bold: 'Sam Lee ',          text: 'resolved your escalation · Smoking · Staircase B',   time: '1 h ago',   unread: true  },
  { id: 3, bold: 'Cam 08 · Pantry L3 ', text: 'went offline',                                     time: '6:12 PM',   unread: false },
];

const QUEUE_ITEMS = [
  { id: 'inc-001', severity: 'Critical' as const, title: 'Smoking · Carpark B2',          subtitle: 'Cam 04 · 11:18 PM', confidence: 92 },
  { id: 'inc-002', severity: 'Moderate' as const, title: 'Unattended item · Meeting Room 1', subtitle: 'Cam 01 · 10:15 AM', confidence: 81 },
  { id: 'inc-003', severity: 'Minor'    as const, title: 'Loitering · Corridor 2',         subtitle: 'Cam 03 · 11:02 AM', confidence: 74 },
  { id: 'inc-004', severity: 'Minor'    as const, title: 'Littering · Lift Lobby 1',       subtitle: 'Cam 05 · 8:37 PM',  confidence: 68 },
];

const HOTSPOTS = [
  { name: 'Carpark B2',   count: 22, pct: 100  },
  { name: 'Lift Lobby 1', count: 16, pct: 66.7 },
  { name: 'Corridor 2',  count: 11, pct: 44.4 },
];

const SEVERITY_DOT: Record<string, string> = {
  Critical: '#D9655C',
  Moderate: '#E08A3C',
  Minor:    '#E6C65C',
};

export default function StaffHomePage() {
  const [outcome, setOutcome] = useState<'none' | 'accepted' | 'dismissed' | 'escalated'>('none');
  const [selectedQueueId, setSelectedQueueId] = useState<string>('');
  // Live incidents from the API replace the demo queue once it answers.
  const { incidents, source } = useIncidents();
  const live = source === 'live';
  const pending = incidents.filter((i) => i.outcome === 'Pending');
  const queueItems = live
    ? pending.map((i) => ({
        id: i.id,
        severity: i.severity,
        title: `${i.type} · ${i.location}`,
        subtitle: `${camLabel(i.camId)} · ${timeLabel(i.dateTime)}`,
        confidence: Math.round(i.confidence * 100),
      }))
    : QUEUE_ITEMS;
  const featured = live
    ? pending.find((i) => i.id === selectedQueueId) ?? pending[0] ?? incidents[0]
    : MOCK_INCIDENTS[0];

  if (!featured) {
    return (
      <AppLayout role="staff" activePath="/staff/home">
        <div style={{ padding: '40px', color: '#4b5563', fontSize: '15px' }}>
          No incidents yet. New detections appear here as cameras are processed.
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout role="staff" activePath="/staff/home">
      {/*
        Outer wrapper fills the 100vh main container exactly.
        Column layout: KPI bar (fixed) → two-column section (flex:1).
      */}
      <div
        style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '20px 28px 20px 20px',
          gap: '16px',
          overflow: 'hidden',
          boxSizing: 'border-box',
        }}
      >
        {/* ── KPI bar ───────────────────────────────────────────── */}
        <div
          style={{
            backgroundColor: '#ffffff',
            borderRadius: '16px',
            paddingTop: '14px',
            paddingBottom: '14px',
            display: 'flex',
            alignItems: 'stretch',
            flexShrink: 0,
          }}
        >
          <KpiCard label="Cameras online"  value="6" valueExtra="/ 8" deltaColor="#b82e24" deltaChar="▼" deltaLabel="2 offline since 6:12 PM" />
          <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
          <KpiCard label="Escalated today" value="0"                  deltaColor="#14803d" deltaChar="▼" deltaLabel="2 fewer than yesterday" />
          <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
          <KpiCard label="To review"       value={String(queueItems.length)}                  deltaColor="#b82e24" deltaChar="▲" deltaLabel="1 more than this time yesterday" />
        </div>

        {/* ── Two-column section (fills remaining height) ───────── */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            gap: '20px',
          }}
        >
          {/* Left: incident card */}
          <div
            style={{
              flex: 1,
              minWidth: 0,
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
              overflowY: 'auto',
            }}
          >
            {/* Incident header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0, overflow: 'hidden' }}>
              <SeverityBadge severity={featured.severity} />
              <p
                style={{
                  flex: '1 0 0',
                  minWidth: 0,
                  fontSize: '20px',
                  fontWeight: 600,
                  color: '#111827',
                  margin: 0,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {featured.type} · {featured.location}
              </p>
              <button type="button" onClick={() => setOutcome('accepted')}
                style={{ ...btnStyle, backgroundColor: outcome === 'accepted' ? '#14863a' : '#16a34a', color: '#fff', border: 'none' }}>
                Accept
              </button>
              <button type="button" onClick={() => setOutcome('dismissed')}
                style={{ ...btnStyle, backgroundColor: '#fff', color: '#111827', border: '1px solid #1f2937' }}>
                Dismiss
              </button>
              <button type="button" onClick={() => setOutcome('escalated')}
                style={{ ...btnStyle, backgroundColor: '#fff', color: '#dc2626', border: '1px solid #dc2626' }}>
                Escalate
              </button>
            </div>

            {/* Video — 16:9 proportional */}
            <div
              style={{
                position: 'relative',
                width: '100%',
                aspectRatio: '16 / 9',
                flexShrink: 0,
                backgroundColor: '#d1d5db',
                borderRadius: '4px',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {featured.clipUrl ? (
                <video
                  key={featured.clipUrl}
                  src={featured.clipUrl}
                  poster={featured.thumbnailUrl}
                  controls
                  loop
                  muted
                  playsInline
                  style={{ width: '100%', height: '100%', objectFit: 'contain', backgroundColor: '#111827' }}
                />
              ) : (
                <>
                <div style={{
                  width: '64px', height: '64px', borderRadius: '32px',
                  backgroundColor: 'rgba(31,41,55,0.85)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                }}>
                  <Play size={26} color="#ffffff" style={{ marginLeft: '3px' }} />
                </div>
                <p style={{ position: 'absolute', top: '16px', left: '18px', color: '#fff', fontSize: '13px', margin: 0 }}>
                  {live ? `${camLabel(featured.camId).toUpperCase()} · ${timeLabel(featured.dateTime)}` : 'CAM 04 · 23:18:06'}
                </p>
                {/* Scrubber */}
                <div style={{ position: 'absolute', bottom: '14px', left: '18px', right: '38px', height: '6px', backgroundColor: '#1f2937', borderRadius: '3px' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: '19%', backgroundColor: '#1f4fd1', borderRadius: '3px' }} />
                </div>
                </>
              )}
            </div>

            {/* Description + reason — compact */}
            <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '10px', padding: '0 4px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>Description</p>
                <p style={{ margin: 0, fontSize: '14px', color: '#111827', lineHeight: 1.5 }}>
                  {featured.description ?? 'Person smoking a cigarette next to the staircase door, ash dropped on the floor.'}
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', flexShrink: 0 }}>{live ? 'Model verdict' : 'Matched rule'}</p>
                <p style={{ margin: 0, fontSize: '13px', color: '#1f4fd1', fontWeight: 600 }}>
                  {live ? featured.verdict : 'Carpark B2 · No Smoking · 10 PM – 6 AM'}
                </p>
                <span style={{ marginLeft: 'auto', fontSize: '13px', color: '#6b7380', fontWeight: 600, flexShrink: 0 }}>
                  {Math.round(featured.confidence * 100)}% confidence
                </span>
              </div>
            </div>

            {/* Fact cards */}
            <div style={{ flexShrink: 0, display: 'flex', gap: '12px' }}>
              {live ? (
                <>
                  <FactCard label="Location" value={featured.location} sub={camLabel(featured.camId)} />
                  <FactCard label="Duration" value={featured.duration ?? '—'} sub={`From ${timeLabel(featured.dateTime)}`} />
                  <FactCard label="Detected" value={featured.type} sub={new Date(featured.dateTime).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })} />
                </>
              ) : (
                <>
                  <FactCard label="Location"        value="Carpark B2, by the stair door" sub="Cam 04" />
                  <FactCard label="Duration"        value="3 min 32 s"                   sub="11:18 PM – 11:21 PM" />
                  <FactCard label="People involved" value="1 adult"                       sub="Grey hoodie, dark cap" />
                </>
              )}
            </div>
          </div>

          {/* Right column: notifications + queue + hotspots — independently scrollable */}
          <div
            style={{
              width: '400px',
              flexShrink: 0,
              height: '100%',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}
          >
            {/* Notifications */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px 10px', flexShrink: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#111827' }}>Notifications</p>
                <div style={{ backgroundColor: '#e8f0ff', color: '#1f4fd1', fontSize: '11px', fontWeight: 600, padding: '2px 7px', borderRadius: '10px' }}>2 new</div>
                <div style={{ flex: 1 }} />
                <button type="button" style={{ color: '#1f4fd1', fontSize: '12px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer' }}>
                  Mark all read
                </button>
              </div>
              {NOTIFICATIONS.map((n) => (
                <div key={n.id} style={{ display: 'flex', gap: '10px', padding: '6px 0' }}>
                  <div style={{ width: '7px', height: '18px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <div style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: n.unread ? '#1f4fd1' : 'transparent' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ margin: 0, fontSize: '13px', color: '#121726', lineHeight: 'normal' }}>
                      <span style={{ fontWeight: 500 }}>{n.bold}</span>
                      <span style={{ fontWeight: 400 }}>{n.text}</span>
                    </p>
                    <p style={{ margin: '1px 0 0', fontSize: '12px', color: '#6b7380' }}>{n.time}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Review queue */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', overflow: 'hidden', flexShrink: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px 12px' }}>
                <p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#111827' }}>Review queue</p>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <p style={{ margin: 0, fontSize: '13px', color: '#4b5563' }}>{queueItems.length}</p>
                  <p style={{ margin: 0, fontSize: '13px', color: '#4b5563' }}>pending</p>
                </div>
              </div>
              <div>
              {queueItems.map((item) => {
                const isSelected = item.id === (live ? featured.id : selectedQueueId || QUEUE_ITEMS[0].id);
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedQueueId(item.id)}
                    style={{
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '14px',
                      paddingLeft: '20px',
                      paddingRight: '20px',
                      paddingTop: '11px',
                      paddingBottom: '11px',
                      borderTop: '1px solid #eef0f3',
                      width: '100%',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    {isSelected && (
                      <div style={{ position: 'absolute', inset: '-1px 0 0 0', backgroundColor: '#eef2fb', pointerEvents: 'none' }} />
                    )}
                    <div style={{ position: 'relative', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: SEVERITY_DOT[item.severity], flexShrink: 0 }} />
                    <div style={{ position: 'relative', flex: '1 0 0', minWidth: 0 }}>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {item.title}
                      </p>
                      <p style={{ margin: '2px 0 0', fontSize: '12px', color: '#4b5563' }}>{item.subtitle}</p>
                    </div>
                    <p style={{ position: 'relative', margin: 0, fontSize: '13px', fontWeight: 500, color: '#6b7380', flexShrink: 0 }}>
                      {item.confidence}%
                    </p>
                  </button>
                );
              })}
              </div>
            </div>

            {/* Hotspots */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px 18px', flexShrink: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#111827' }}>Hotspots</p>
                <button type="button" style={{ color: '#1f4fd1', fontSize: '13px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer' }}>
                  View all
                </button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {HOTSPOTS.map((h) => (
                  <div key={h.name} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <p style={{ margin: 0, fontSize: '14px', color: '#111827', width: '100px', flexShrink: 0 }}>{h.name}</p>
                    <div style={{ flex: 1, height: '6px', backgroundColor: '#eef0f3', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${h.pct}%`, backgroundColor: '#1f4fd1', borderRadius: '3px' }} />
                    </div>
                    <p style={{ margin: 0, fontSize: '14px', color: '#111827', width: '24px', textAlign: 'right', flexShrink: 0 }}>{h.count}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

function camLabel(camId: string) {
  return 'Cam ' + camId.replace('cam-', '').padStart(2, '0');
}

function timeLabel(iso: string) {
  return new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

// ── Shared button style ────────────────────────────────────────────────────────
const btnStyle: React.CSSProperties = {
  borderRadius: '8px',
  padding: '8px 14px',
  fontSize: '14px',
  fontWeight: 600,
  cursor: 'pointer',
  flexShrink: 0,
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
};

// ── KpiCard ────────────────────────────────────────────────────────────────────
interface KpiCardProps {
  label: string;
  value: string;
  valueExtra?: string;
  deltaChar: string;
  deltaColor: string;
  deltaLabel: string;
}
function KpiCard({ label, value, valueExtra, deltaChar, deltaColor, deltaLabel }: KpiCardProps) {
  return (
    <div style={{ flex: '1 0 0', minWidth: 0, paddingLeft: '24px', paddingRight: '24px', paddingTop: '2px', paddingBottom: '2px', display: 'flex', flexDirection: 'column', gap: '4px', overflow: 'hidden' }}>
      <p style={{ margin: 0, fontSize: '13px', color: '#6b7380', fontWeight: 400 }}>{label}</p>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
        <span style={{ fontSize: '32px', fontWeight: 700, color: '#111827', lineHeight: 1 }}>{value}</span>
        {valueExtra && <span style={{ fontSize: '14px', fontWeight: 400, color: '#374151' }}>{valueExtra}</span>}
      </div>
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7380', fontWeight: 400, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        <span style={{ color: deltaColor }}>{deltaChar}</span>{` ${deltaLabel}`}
      </p>
    </div>
  );
}

// ── FactCard ───────────────────────────────────────────────────────────────────
interface FactCardProps { label: string; value: string; sub: string; }
function FactCard({ label, value, sub }: FactCardProps) {
  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: '12px', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: '6px', flex: 1, minWidth: 0 }}>
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>{label}</p>
      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{value}</p>
      <p style={{ margin: 0, fontSize: '12px', color: '#4b5563', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{sub}</p>
    </div>
  );
}
