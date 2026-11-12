import { useState } from 'react';
import { Play } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { SeverityBadge } from '../../components/ui/Badge';
import { MOCK_INCIDENTS } from '../../data/mock';

const NOTIFICATIONS = [
  {
    id: 1,
    bold: 'New alert: ',
    text: 'Smoking · Carpark B2',
    time: '2 min ago',
    unread: true,
  },
  {
    id: 2,
    bold: 'Sam Lee ',
    text: 'resolved your escalation · Smoking · Staircase B',
    time: '1 h ago',
    unread: true,
  },
  {
    id: 3,
    bold: 'Cam 08 · Pantry L3 ',
    text: 'went offline',
    time: '6:12 PM',
    unread: false,
  },
];

const QUEUE_ITEMS = [
  { id: 'inc-001', severity: 'Critical' as const, title: 'Smoking · Carpark B2',       subtitle: 'Cam 04 · 11:18 PM', confidence: 92 },
  { id: 'inc-002', severity: 'Moderate' as const, title: 'Unattended item · Meeting Room 1', subtitle: 'Cam 01 · 10:15 AM', confidence: 81 },
  { id: 'inc-003', severity: 'Minor'    as const, title: 'Loitering · Corridor 2',      subtitle: 'Cam 03 · 11:02 AM', confidence: 74 },
  { id: 'inc-004', severity: 'Minor'    as const, title: 'Littering · Lift Lobby 1',    subtitle: 'Cam 05 · 8:37 PM',  confidence: 68 },
];

const HOTSPOTS = [
  { name: 'Carpark B2',  count: 22, pct: 100 },
  { name: 'Lift Lobby 1', count: 16, pct: 66.7 },
  { name: 'Corridor 2',  count: 11, pct: 44.4 },
];

// Severity dot colors for review queue circles
const SEVERITY_DOT: Record<string, string> = {
  Critical: '#D9655C',
  Moderate: '#E08A3C',
  Minor:    '#E6C65C',
};

export default function StaffHomePage() {
  const [outcome, setOutcome] = useState<'none' | 'accepted' | 'dismissed' | 'escalated'>('none');
  const [selectedQueueId, setSelectedQueueId] = useState<string>(QUEUE_ITEMS[0].id);

  const featured = MOCK_INCIDENTS[0];

  return (
    <AppLayout role="staff" activePath="/staff/home">
      <div
        style={{
          display: 'flex',
          gap: '28px',
          padding: '28px 28px 28px 20px',
          minHeight: '100%',
          alignItems: 'flex-start',
        }}
      >
        {/* Left column */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* KPI bar */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              paddingTop: '18px',
              paddingBottom: '18px',
              display: 'flex',
              alignItems: 'stretch',
            }}
          >
            <KpiCard
              label="Cameras online"
              value="6"
              valueExtra="/ 8"
              deltaColor="#b82e24"
              deltaChar="▼"
              deltaLabel="2 offline since 6:12 PM"
            />
            <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
            <KpiCard
              label="Escalated today"
              value="0"
              deltaColor="#14803d"
              deltaChar="▼"
              deltaLabel="2 fewer than yesterday"
            />
            <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
            <KpiCard
              label="To review"
              value="4"
              deltaColor="#b82e24"
              deltaChar="▲"
              deltaLabel="1 more than this time yesterday"
            />
          </div>

          {/* Incident card */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', overflow: 'hidden' }}>
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
              {/* Buttons */}
              <button
                type="button"
                onClick={() => setOutcome('accepted')}
                style={{
                  backgroundColor: outcome === 'accepted' ? '#14863a' : '#16a34a',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '9px 14px',
                  fontSize: '15px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  flexShrink: 0,
                }}
              >
                Accept
              </button>
              <button
                type="button"
                onClick={() => setOutcome('dismissed')}
                style={{
                  backgroundColor: '#ffffff',
                  color: '#111827',
                  border: '1px solid #1f2937',
                  borderRadius: '8px',
                  padding: '9px 14px',
                  fontSize: '15px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  flexShrink: 0,
                }}
              >
                Dismiss
              </button>
              <button
                type="button"
                onClick={() => setOutcome('escalated')}
                style={{
                  backgroundColor: '#ffffff',
                  color: '#dc2626',
                  border: '1px solid #dc2626',
                  borderRadius: '8px',
                  padding: '9px 14px',
                  fontSize: '15px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  flexShrink: 0,
                }}
              >
                Escalate
              </button>
            </div>

            {/* Video placeholder */}
            <div
              style={{
                position: 'relative',
                width: '100%',
                height: '352px',
                backgroundColor: '#d1d5db',
                borderRadius: '4px',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {/* Play button */}
              <div
                style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '32px',
                  backgroundColor: 'rgba(31,41,55,0.85)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                }}
              >
                <Play size={26} color="#ffffff" style={{ marginLeft: '3px' }} />
              </div>
              {/* Cam label */}
              <p
                style={{
                  position: 'absolute',
                  top: '16px',
                  left: '18px',
                  color: '#ffffff',
                  fontSize: '13px',
                  fontWeight: 400,
                  margin: 0,
                }}
              >
                CAM 04 · 23:18:06
              </p>
              {/* Scrubber track */}
              <div
                style={{
                  position: 'absolute',
                  bottom: '14px',
                  left: '18px',
                  right: '38px',
                  height: '6px',
                  backgroundColor: '#1f2937',
                  borderRadius: '3px',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: '19%',
                    backgroundColor: '#1f4fd1',
                    borderRadius: '3px',
                  }}
                />
              </div>
            </div>

            {/* Details */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '18px',
                padding: '8px 16px 4px',
              }}
            >
              {/* Description */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <p style={{ margin: 0, fontSize: '13px', color: '#6b7280', fontWeight: 400 }}>Description</p>
                <p style={{ margin: 0, fontSize: '16px', color: '#111827', fontWeight: 400 }}>
                  {featured.description ?? 'Person smoking a cigarette next to the staircase door, ash dropped on the floor.'}
                </p>
              </div>
              {/* Reason for flagging */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <p style={{ margin: 0, fontSize: '13px', color: '#6b7280', fontWeight: 400, flexShrink: 0 }}>
                    Reason for flagging
                  </p>
                  <p style={{ margin: 0, fontSize: '14px', color: '#6b7380', fontWeight: 600 }}>
                    {Math.round(featured.confidence * 100)}% Confidence
                  </p>
                </div>
                <p style={{ margin: 0, fontSize: '16px', color: '#111827', fontWeight: 400 }}>
                  Repeated hand-to-mouth motion with visible smoke in a no-smoking zone.
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <p style={{ margin: 0, fontSize: '13px', color: '#6b7280', fontWeight: 400 }}>Matched rule</p>
                  <p style={{ margin: 0, fontSize: '14px', color: '#1f4fd1', fontWeight: 600 }}>
                    Carpark B2 · No Smoking · 10 PM – 6 AM
                  </p>
                </div>
              </div>
            </div>

            {/* Fact cards */}
            <div style={{ display: 'flex', gap: '16px' }}>
              <FactCard
                icon="📍"
                label="Location"
                value="Carpark B2, by the stair door"
                sub="Cam 04"
              />
              <FactCard
                icon="🕐"
                label="Duration"
                value="3 min 32 s"
                sub="11:18:06 PM – 11:21:38 PM"
              />
              <FactCard
                icon="👤"
                label="People involved"
                value="1 adult"
                sub="Grey hoodie, dark cap"
              />
            </div>
          </div>
        </div>

        {/* Right column — 3 separate cards */}
        <div
          style={{
            width: '440px',
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '20px',
          }}
        >
          {/* Notifications card */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              paddingTop: '18px',
              paddingBottom: '10px',
              paddingLeft: '20px',
              paddingRight: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            {/* Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                paddingBottom: '4px',
              }}
            >
              <p style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#111827', flexShrink: 0 }}>
                Notifications
              </p>
              <div
                style={{
                  backgroundColor: '#e8f0ff',
                  color: '#1f4fd1',
                  fontSize: '11px',
                  fontWeight: 600,
                  padding: '2px 7px',
                  borderRadius: '10px',
                  flexShrink: 0,
                }}
              >
                2 new
              </div>
              <div style={{ flex: '1 0 0' }} />
              <button
                type="button"
                style={{
                  color: '#1f4fd1',
                  fontSize: '12px',
                  fontWeight: 500,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                Mark all read
              </button>
            </div>
            {/* Notification items */}
            {NOTIFICATIONS.map((n) => (
              <div
                key={n.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  padding: '7px 0',
                }}
              >
                {/* Dot — unread: blue filled, read: transparent */}
                <div style={{ width: '7px', height: '18px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <div
                    style={{
                      width: '7px',
                      height: '7px',
                      borderRadius: '50%',
                      backgroundColor: n.unread ? '#1f4fd1' : 'transparent',
                    }}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ margin: 0, fontSize: '13px', color: '#121726', lineHeight: 'normal' }}>
                    <span style={{ fontWeight: 500 }}>{n.bold}</span>
                    <span style={{ fontWeight: 400 }}>{n.text}</span>
                  </p>
                  <p style={{ margin: '1px 0 0 0', fontSize: '12px', color: '#6b7380', fontWeight: 400 }}>
                    {n.time}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Review queue card */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                paddingLeft: '20px',
                paddingRight: '28px',
                paddingTop: '20px',
                paddingBottom: '14px',
              }}
            >
              <p style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#111827' }}>
                Review queue
              </p>
              <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                <p style={{ margin: 0, fontSize: '13px', color: '#4b5563', fontWeight: 400 }}>4</p>
                <p style={{ margin: 0, fontSize: '13px', color: '#4b5563', fontWeight: 400 }}>pending</p>
              </div>
            </div>

            {/* Queue items */}
            {QUEUE_ITEMS.map((item) => {
              const isSelected = item.id === selectedQueueId;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedQueueId(item.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '14px',
                    paddingLeft: '20px',
                    paddingRight: '28px',
                    paddingTop: '13px',
                    paddingBottom: '13px',
                    borderTop: '1px solid #eef0f3',
                    width: '100%',
                    background: 'none',
                    border: 'none',
                    borderTop: '1px solid #eef0f3',
                    cursor: 'pointer',
                    textAlign: 'left',
                    position: 'relative',
                  }}
                >
                  {/* Selected highlight */}
                  {isSelected && (
                    <div
                      style={{
                        position: 'absolute',
                        inset: '-1px 0 0 0',
                        backgroundColor: '#eef2fb',
                        pointerEvents: 'none',
                      }}
                    />
                  )}
                  {/* Severity dot */}
                  <div
                    style={{
                      position: 'relative',
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      backgroundColor: SEVERITY_DOT[item.severity],
                      flexShrink: 0,
                    }}
                  />
                  {/* Text */}
                  <div
                    style={{
                      position: 'relative',
                      flex: '1 0 0',
                      minWidth: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '3px',
                    }}
                  >
                    <p
                      style={{
                        margin: 0,
                        fontSize: '15px',
                        fontWeight: 600,
                        color: '#111827',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {item.title}
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', color: '#4b5563', fontWeight: 400 }}>
                      {item.subtitle}
                    </p>
                  </div>
                  {/* Confidence */}
                  <p
                    style={{
                      position: 'relative',
                      margin: 0,
                      fontSize: '13px',
                      fontWeight: 500,
                      color: '#6b7380',
                      flexShrink: 0,
                    }}
                  >
                    {item.confidence}%
                  </p>
                </button>
              );
            })}
          </div>

          {/* Hotspots card */}
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              paddingLeft: '20px',
              paddingRight: '28px',
              paddingTop: '20px',
              paddingBottom: '22px',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <p style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#111827' }}>Hotspots</p>
              <button
                type="button"
                style={{
                  color: '#1f4fd1',
                  fontSize: '14px',
                  fontWeight: 500,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                View all
              </button>
            </div>
            {/* Hotspot rows */}
            {HOTSPOTS.map((h) => (
              <div
                key={h.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '14px',
                }}
              >
                <p
                  style={{
                    margin: 0,
                    fontSize: '15px',
                    fontWeight: 400,
                    color: '#111827',
                    width: '110px',
                    flexShrink: 0,
                  }}
                >
                  {h.name}
                </p>
                <div
                  style={{
                    flex: 1,
                    height: '6px',
                    backgroundColor: '#eef0f3',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${h.pct}%`,
                      backgroundColor: '#1f4fd1',
                      borderRadius: '3px',
                    }}
                  />
                </div>
                <p
                  style={{
                    margin: 0,
                    fontSize: '15px',
                    fontWeight: 400,
                    color: '#111827',
                    width: '20px',
                    textAlign: 'right',
                    flexShrink: 0,
                  }}
                >
                  {h.count}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

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
    <div
      style={{
        flex: '1 0 0',
        minWidth: 0,
        paddingLeft: '24px',
        paddingRight: '24px',
        paddingTop: '4px',
        paddingBottom: '4px',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        overflow: 'hidden',
      }}
    >
      <p style={{ margin: 0, fontSize: '13px', color: '#6b7380', fontWeight: 400 }}>{label}</p>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
        <span style={{ fontSize: '34px', fontWeight: 700, color: '#111827', lineHeight: 1 }}>{value}</span>
        {valueExtra && (
          <span style={{ fontSize: '15px', fontWeight: 400, color: '#374151' }}>{valueExtra}</span>
        )}
      </div>
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7380', fontWeight: 400 }}>
        <span style={{ color: deltaColor }}>{deltaChar}</span>
        {` ${deltaLabel}`}
      </p>
    </div>
  );
}

interface FactCardProps {
  icon: string;
  label: string;
  value: string;
  sub: string;
}

function FactCard({ icon: _icon, label, value, sub }: FactCardProps) {
  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '14px',
        paddingLeft: '18px',
        paddingRight: '18px',
        paddingTop: '16px',
        paddingBottom: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        flex: 1,
        minWidth: 0,
      }}
    >
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', fontWeight: 400 }}>{label}</p>
      <p style={{ margin: 0, fontSize: '15px', fontWeight: 600, color: '#111827' }}>{value}</p>
      <p style={{ margin: 0, fontSize: '14px', color: '#4b5563', fontWeight: 400 }}>{sub}</p>
    </div>
  );
}
