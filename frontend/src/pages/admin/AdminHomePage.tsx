import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, ChevronRight, AlertTriangle, Play } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { MOCK_INCIDENTS, MOCK_STAFF } from '../../data/mock';

const INITIAL_NOTIFICATIONS = [
  { id: 1, bold: 'Cam 08: ', text: 'Pantry L3 went offline since 9:21 AM', time: '9:21 AM', unread: true },
  { id: 2, bold: 'Ben Lee ', text: 'received an escalation from Jane Tan', time: '10:03 AM', unread: true },
  { id: 3, bold: '',         text: 'Weekly workload report is ready',         time: 'Yesterday', unread: false },
];

const SEVERITY_COLOR: Record<string, string> = {
  Critical: '#ef4444',
  Moderate: '#f97316',
  Minor:    '#eab308',
};

export default function AdminHomePage() {
  const navigate = useNavigate();
  const [assignedTo, setAssignedTo] = useState('');
  const [notifications, setNotifications] = useState(INITIAL_NOTIFICATIONS);

  const escalated = MOCK_INCIDENTS.filter((i) => i.outcome === 'Escalated');
  const resolved  = MOCK_INCIDENTS.filter((i) => i.outcome === 'Accepted').slice(0, 2);
  const unreadCount = notifications.filter((n) => n.unread).length;

  return (
    <AppLayout role="admin" activePath="/admin/home">
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
          <KpiCard label="Escalated to you"    value="3" deltaColor="#b82e24" deltaChar="▲" deltaLabel="1 more than yesterday" />
          <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
          <KpiCard label="Dismissed this week" value="9" deltaColor="#b82e24" deltaChar="▲" deltaLabel="+2 more than last week" />
          <div style={{ width: '1px', backgroundColor: '#e0e3e8', flexShrink: 0 }} />
          <KpiCard label="Cameras online"      value="6" valueExtra="/ 8" deltaColor="#14803d" deltaChar="▼" deltaLabel="1 offline since 9 AM" />
        </div>

        {/* ── Two-column section ────────────────────────────────── */}
        <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: '20px' }}>

          {/* Left: incident detail */}
          <div
            style={{
              flex: 1,
              minWidth: 0,
              overflowY: 'auto',
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', overflow: 'hidden' }}>
              <span style={{ backgroundColor: '#f2c0ba', color: '#8c2a22', borderRadius: '6px', padding: '4px 10px', fontSize: '13px', fontWeight: 600, flexShrink: 0 }}>
                Critical
              </span>
              <p style={{ flex: '1 0 0', minWidth: 0, margin: 0, fontSize: '20px', fontWeight: 600, color: '#111827', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Smoking · Carpark B2
              </p>
              <span style={{ fontSize: '13px', color: '#6b7280', backgroundColor: '#f3f4f6', borderRadius: '20px', padding: '4px 12px', flexShrink: 0, fontWeight: 500 }}>
                Unassigned
              </span>
            </div>

            {/* Video placeholder */}
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
              <div style={{ width: '64px', height: '64px', borderRadius: '32px', backgroundColor: 'rgba(31,41,55,0.85)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                <Play size={26} color="#ffffff" style={{ marginLeft: '3px' }} />
              </div>
              <p style={{ position: 'absolute', top: '16px', left: '18px', color: '#fff', fontSize: '13px', margin: 0 }}>
                CAM 04 · 22:47:00
              </p>
              <div style={{ position: 'absolute', bottom: '14px', left: '18px', right: '38px', height: '6px', backgroundColor: '#1f2937', borderRadius: '3px' }}>
                <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: '30%', backgroundColor: '#1f4fd1', borderRadius: '3px' }} />
              </div>
            </div>

            {/* Metadata */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#374151' }}>Cam 04 · 10:47 PM</span>
              <span style={{ fontSize: '13px', color: '#6b7280' }}>Mon 11 Oct</span>
              <span style={{ fontSize: '13px', color: '#6b7280' }}>3 min 32 s</span>
              <span style={{ fontSize: '13px', color: '#6b7280' }}>1 adult</span>
              <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '13px', color: '#6b7280' }}>
                <AlertTriangle size={13} color="#f97316" />
                AI confidence: <strong style={{ color: '#374151' }}>94%</strong>
              </span>
            </div>

            {/* Description */}
            <div style={{ padding: '0 4px' }}>
              <p style={{ margin: '0 0 4px', fontSize: '12px', color: '#6b7280' }}>Description</p>
              <p style={{ margin: 0, fontSize: '14px', color: '#111827', lineHeight: 1.5 }}>
                Individual observed smoking near pillar P-04. Smoke clearly visible. Approx. 3 minutes duration before moving off.
              </p>
            </div>

            {/* Assign */}
            <div style={{ borderTop: '1px solid #f3f4f6', paddingTop: '14px', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '14px', fontWeight: 500, color: '#374151', flexShrink: 0 }}>Assign to staff:</span>
              <select
                value={assignedTo}
                onChange={(e) => setAssignedTo(e.target.value)}
                style={{
                  appearance: 'none',
                  border: '1px solid #d0d5dd',
                  borderRadius: '8px',
                  padding: '8px 32px 8px 12px',
                  fontSize: '14px',
                  color: '#101828',
                  backgroundColor: '#ffffff',
                  cursor: 'pointer',
                  outline: 'none',
                  boxShadow: '0px 1px 2px rgba(16,24,40,0.05)',
                }}
              >
                <option value="">— Select staff —</option>
                {MOCK_STAFF.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Right column */}
          <div
            style={{
              width: '400px',
              flexShrink: 0,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}
          >
            {/* Notifications */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px 10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#111827' }}>Notifications</p>
                {unreadCount > 0 && (
                  <div style={{ backgroundColor: '#e8f0ff', color: '#1f4fd1', fontSize: '11px', fontWeight: 600, padding: '2px 7px', borderRadius: '10px' }}>{unreadCount} new</div>
                )}
                <div style={{ flex: 1 }} />
                <button
                  type="button"
                  style={{ color: '#1f4fd1', fontSize: '12px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer' }}
                  onClick={() => setNotifications((prev) => prev.map((n) => ({ ...n, unread: false })))}
                >
                  Mark all read
                </button>
              </div>
              {notifications.map((n) => (
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

            {/* Escalated cases */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#111827' }}>Escalated cases</p>
                  <div style={{ backgroundColor: '#fee2e2', color: '#dc2626', fontSize: '11px', fontWeight: 600, padding: '2px 7px', borderRadius: '10px' }}>{escalated.length} open</div>
                </div>
                <button type="button" onClick={() => navigate('/admin/workload')} style={{ color: '#1f4fd1', fontSize: '13px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '2px' }}>
                  View all <ChevronRight size={13} />
                </button>
              </div>
              {escalated.map((inc) => (
                <div key={inc.id} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 0', borderTop: '1px solid #f3f4f6' }}>
                  <div style={{ width: '4px', height: '40px', borderRadius: '2px', backgroundColor: SEVERITY_COLOR[inc.severity], flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 500, color: '#111827', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{inc.type} · {inc.location}</p>
                    <p style={{ margin: '2px 0 0', fontSize: '12px', color: '#6b7280' }}>{new Date(inc.dateTime).toLocaleDateString('en-SG', { day: 'numeric', month: 'short' })}</p>
                  </div>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: '#1f4fd1', backgroundColor: '#dbe4fb', padding: '2px 8px', borderRadius: '20px', flexShrink: 0 }}>
                    {Math.round(inc.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Recently resolved */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px' }}>
              <p style={{ margin: '0 0 10px', fontSize: '16px', fontWeight: 600, color: '#111827' }}>Recently resolved</p>
              {resolved.map((inc) => (
                <div key={inc.id} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 0', borderTop: '1px solid #f3f4f6' }}>
                  <CheckCircle size={15} color="#16a34a" style={{ flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 500, color: '#111827', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{inc.type} · {inc.location}</p>
                    <p style={{ margin: '2px 0 0', fontSize: '12px', color: '#6b7280' }}>{new Date(inc.dateTime).toLocaleDateString('en-SG', { day: 'numeric', month: 'short' })}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Recent workload reports */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '16px', padding: '16px 20px' }}>
              <p style={{ margin: '0 0 10px', fontSize: '16px', fontWeight: 600, color: '#111827' }}>Workload reports</p>
              {[
                { label: '5–11 Oct', dates: '5 Oct – 11 Oct 2026' },
                { label: '1–4 Oct',  dates: '1 Oct – 4 Oct 2026'  },
              ].map((r) => (
                <div key={r.label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderTop: '1px solid #f3f4f6' }}>
                  <div>
                    <p style={{ margin: 0, fontSize: '13px', fontWeight: 500, color: '#111827' }}>{r.label}</p>
                    <p style={{ margin: '2px 0 0', fontSize: '12px', color: '#6b7280' }}>{r.dates}</p>
                  </div>
                  <button type="button" onClick={() => navigate('/admin/workload')} style={{ color: '#1f4fd1', fontSize: '13px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '2px' }}>
                    View <ChevronRight size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

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
