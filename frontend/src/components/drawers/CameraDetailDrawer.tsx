import { createPortal } from 'react-dom';
import { X, Wifi, WifiOff, AlertTriangle, ChevronRight } from 'lucide-react';
import type { Camera } from '../../types';
import { MOCK_INCIDENTS, MOCK_RULES } from '../../data/mock';
import { SeverityBadge, OutcomeDot } from '../ui/Badge';

interface CameraDetailDrawerProps {
  cam: Camera | null;
  onClose: () => void;
}

function formatLastSeen(iso?: string): string {
  if (!iso) return 'Unknown';
  const d = new Date(iso);
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) +
    ', ' + d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
}

export default function CameraDetailDrawer({ cam, onClose }: CameraDetailDrawerProps) {
  if (!cam) return null;

  const online = cam.status === 'online';

  // Incidents for this camera (most recent first, cap at 5)
  const camIncidents = MOCK_INCIDENTS
    .filter((i) => i.camId === cam.id)
    .sort((a, b) => new Date(b.dateTime).getTime() - new Date(a.dateTime).getTime())
    .slice(0, 5);

  // Rules applied to this camera
  const camRules = MOCK_RULES.filter((r) => cam.rules.includes(r.id));

  return createPortal(
    <>
      {/* ── Backdrop ──────────────────────────────────────────────── */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.35)',
          zIndex: 199,
        }}
        onClick={onClose}
      />

      {/* ── Drawer panel ──────────────────────────────────────────── */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          bottom: 0,
          width: '520px',
          backgroundColor: '#ffffff',
          zIndex: 200,
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '-4px 0 32px rgba(0,0,0,0.14)',
        }}
      >
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>

          {/* ── Header ───────────────────────────────────────────── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            {/* Status dot */}
            <span
              style={{
                width: 8, height: 8, borderRadius: '50%',
                backgroundColor: online ? '#22c55e' : '#ef4444',
                display: 'inline-block', flexShrink: 0,
              }}
            />
            <p style={{
              flex: 1, margin: 0,
              fontSize: '20px', fontWeight: 600, color: '#111827',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {cam.name} · {cam.location}
            </p>
            <button
              type="button"
              onClick={onClose}
              style={{
                width: '32px', height: '32px', borderRadius: '8px',
                border: 'none', backgroundColor: '#f3f4f6',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer', flexShrink: 0,
              }}
            >
              <X size={16} color="#374151" />
            </button>
          </div>

          {/* ── Video feed placeholder ────────────────────────────── */}
          <div
            style={{
              position: 'relative',
              width: '100%',
              aspectRatio: '16 / 9',
              borderRadius: '10px',
              overflow: 'hidden',
              backgroundColor: online ? '#1f2937' : '#374151',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px',
            }}
          >
            {online ? (
              <>
                <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.4)' }}>Live feed</span>
                {/* Overlay label */}
                <div
                  style={{
                    position: 'absolute', top: '10px', left: '12px',
                    backgroundColor: 'rgba(0,0,0,0.55)',
                    borderRadius: '4px', padding: '2px 8px',
                  }}
                >
                  <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: 500 }}>
                    {cam.name} · LIVE
                  </span>
                </div>
                {/* Green live indicator */}
                <div
                  style={{
                    position: 'absolute', top: '12px', right: '12px',
                    display: 'flex', alignItems: 'center', gap: '5px',
                    backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: '20px', padding: '3px 8px',
                  }}
                >
                  <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#22c55e', display: 'inline-block' }} />
                  <span style={{ fontSize: '11px', color: '#ffffff', fontWeight: 600 }}>Online</span>
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                <WifiOff size={28} color="rgba(255,255,255,0.4)" />
                <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>Feed unavailable</span>
              </div>
            )}
          </div>

          {/* ── Status + last seen facts ──────────────────────────── */}
          <div
            style={{
              display: 'flex',
              border: '1px solid #e5e8ed',
              borderRadius: '10px',
              overflow: 'hidden',
              marginBottom: '20px',
            }}
          >
            <div style={{ flex: 1, padding: '10px 14px', borderRight: '1px solid #e5e8ed' }}>
              <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>Status</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '3px' }}>
                {online
                  ? <Wifi size={13} color="#22c55e" />
                  : <WifiOff size={13} color="#ef4444" />}
                <span style={{ fontSize: '14px', fontWeight: 600, color: online ? '#16a34a' : '#dc2626' }}>
                  {online ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
            <div style={{ flex: 1, padding: '10px 14px', borderRight: '1px solid #e5e8ed' }}>
              <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>Last seen</p>
              <p style={{ margin: '3px 0 0', fontSize: '14px', fontWeight: 600, color: '#111827' }}>
                {formatLastSeen(cam.lastSeen)}
              </p>
            </div>
            <div style={{ flex: 1, padding: '10px 14px' }}>
              <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>Today's incidents</p>
              <p style={{ margin: '3px 0 0', fontSize: '14px', fontWeight: 600, color: '#111827' }}>
                {camIncidents.length}
              </p>
            </div>
          </div>

          {/* ── Active rules ──────────────────────────────────────── */}
          <div style={{ marginBottom: '20px' }}>
            <p style={{ margin: '0 0 10px', fontSize: '13px', fontWeight: 600, color: '#111827' }}>
              Active rules
            </p>
            {camRules.length === 0 ? (
              <p style={{ margin: 0, fontSize: '13px', color: '#9ca3af' }}>No rules assigned</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {camRules.map((rule) => (
                  <div
                    key={rule.id}
                    style={{
                      display: 'flex', alignItems: 'center',
                      padding: '10px 14px',
                      backgroundColor: rule.enabled ? '#f9fafb' : '#f3f4f6',
                      borderRadius: '8px',
                      gap: '10px',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{
                        margin: 0, fontSize: '13px', fontWeight: 500,
                        color: rule.enabled ? '#111827' : '#9ca3af',
                      }}>
                        {rule.detectType}
                      </p>
                      <p style={{ margin: '2px 0 0', fontSize: '12px', color: '#6b7280' }}>
                        {rule.activeFrom} – {rule.activeTo} · {rule.severity}
                      </p>
                    </div>
                    <span
                      style={{
                        fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '20px',
                        backgroundColor: rule.enabled ? '#dcfce7' : '#f3f4f6',
                        color: rule.enabled ? '#16a34a' : '#9ca3af',
                        flexShrink: 0,
                      }}
                    >
                      {rule.enabled ? 'Active' : 'Off'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Recent incidents ──────────────────────────────────── */}
          <div>
            <p style={{ margin: '0 0 10px', fontSize: '13px', fontWeight: 600, color: '#111827' }}>
              Recent incidents
            </p>
            {camIncidents.length === 0 ? (
              <p style={{ margin: 0, fontSize: '13px', color: '#9ca3af' }}>No incidents recorded</p>
            ) : (
              <div>
                {camIncidents.map((inc, i) => (
                  <div
                    key={inc.id}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '10px',
                      padding: '8px 0',
                      borderTop: i > 0 ? '1px solid #f3f4f6' : 'none',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                        <SeverityBadge severity={inc.severity} />
                        <p style={{
                          margin: 0, fontSize: '13px', fontWeight: 500, color: '#111827',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                          {inc.type}
                        </p>
                      </div>
                      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
                        {formatTime(inc.dateTime)}
                      </p>
                    </div>
                    <OutcomeDot outcome={inc.outcome} />
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>

        {/* ── Footer ───────────────────────────────────────────────── */}
        <div
          style={{
            flexShrink: 0,
            padding: '14px 28px',
            borderTop: '1px solid #f3f4f6',
            display: 'flex',
            gap: '10px',
          }}
        >
          <button
            type="button"
            style={{
              flex: 1, height: '39px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              backgroundColor: '#ffffff',
              border: '1px solid #d0d5dd',
              borderRadius: '8px',
              fontSize: '14px', fontWeight: 500, color: '#374151',
              cursor: 'pointer',
            }}
          >
            <AlertTriangle size={14} color="#f97316" />
            View incidents
          </button>
          <button
            type="button"
            style={{
              flex: 1, height: '39px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              backgroundColor: '#1f4fd1',
              border: 'none',
              borderRadius: '8px',
              fontSize: '14px', fontWeight: 500, color: '#ffffff',
              cursor: 'pointer',
            }}
          >
            Edit camera
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </>,
    document.body
  );
}
