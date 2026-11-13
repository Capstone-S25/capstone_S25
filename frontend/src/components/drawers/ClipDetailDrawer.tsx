import { createPortal } from 'react-dom';
import { Play, X, Download, Search as SearchIcon } from 'lucide-react';
import { SeverityBadge } from '../ui/Badge';
import type { Incident } from '../../types';
import { MOCK_STAFF } from '../../data/mock';

interface ClipDetailDrawerProps {
  inc: Incident | null;
  onClose: () => void;
}

const STAFF_MAP: Record<string, string> = Object.fromEntries(
  MOCK_STAFF.map((s) => [s.id, s.name])
);

const OUTCOME_DOT_COLOR: Record<string, string> = {
  Escalated: '#ef4444',
  Accepted:  '#22c55e',
  Dismissed: '#9ca3af',
  Pending:   '#3b82f6',
};

function camLabel(camId: string) {
  return 'Cam ' + camId.replace('cam-', '').padStart(2, '0');
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-GB', {
    weekday: 'short', day: 'numeric', month: 'short',
  });
}

function formatTime(iso: string) {
  return new Date(iso).toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit',
  });
}

/** Approximate "reviewed at" = incident time + 3 min */
function reviewedAt(iso: string) {
  const d = new Date(new Date(iso).getTime() + 3 * 60 * 1000);
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
}

export default function ClipDetailDrawer({ inc, onClose }: ClipDetailDrawerProps) {
  if (!inc) return null;

  const assignee   = inc.assignedTo ? (STAFF_MAP[inc.assignedTo] ?? '—') : '—';
  const dotColor   = OUTCOME_DOT_COLOR[inc.outcome] ?? '#9ca3af';
  const confidence = Math.round(inc.confidence * 100) + '%';

  // Remark text logic
  let remarkText: string | null = null;
  if (inc.remark) {
    // Escalated / Dismissed remarks are shown in quotes
    remarkText = inc.outcome === 'Accepted' ? inc.remark : `"${inc.remark}"`;
  } else if (inc.outcome === 'Accepted') {
    remarkText = 'Logged as an incident. No remark needed.';
  }

  // Bottom status line in the outcome card
  const outcomeFooter =
    inc.outcome === 'Escalated' ? 'Open · waiting on admin' : null;

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
        {/* ── Scrollable content ───────────────────────────────────── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px 24px' }}>

          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <SeverityBadge severity={inc.severity} />
            <p style={{
              flex: 1, margin: 0,
              fontSize: '20px', fontWeight: 600, color: '#111827',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {inc.type} · {inc.location}
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

          {/* Video placeholder */}
          <div
            style={{
              position: 'relative',
              width: '100%',
              aspectRatio: '16 / 9',
              backgroundColor: '#d1d5db',
              borderRadius: '8px',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px',
            }}
          >
            {inc.clipUrl ? (
              <video
                key={inc.clipUrl}
                src={inc.clipUrl}
                poster={inc.thumbnailUrl}
                controls
                loop
                muted
                playsInline
                style={{ width: '100%', height: '100%', objectFit: 'contain', backgroundColor: '#111827' }}
              />
            ) : (
              <div
                style={{
                  width: '52px', height: '52px', borderRadius: '26px',
                  backgroundColor: 'rgba(31,41,55,0.8)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer',
                }}
              >
                <Play size={22} color="#ffffff" style={{ marginLeft: '3px' }} />
              </div>
            )}
          </div>

          {/* Facts row */}
          <div
            style={{
              display: 'flex',
              border: '1px solid #e5e8ed',
              borderRadius: '10px',
              overflow: 'hidden',
              marginBottom: '20px',
            }}
          >
            {[
              { label: 'Camera',      value: camLabel(inc.camId) },
              { label: 'Date & time', value: `${formatDate(inc.dateTime)}, ${formatTime(inc.dateTime)}` },
              { label: 'Duration',    value: inc.duration ?? '—' },
            ].map((fact, i) => (
              <div
                key={fact.label}
                style={{
                  flex: 1,
                  padding: '10px 14px',
                  borderLeft: i > 0 ? '1px solid #e5e8ed' : 'none',
                }}
              >
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>{fact.label}</p>
                <p style={{ margin: '3px 0 0', fontSize: '14px', fontWeight: 600, color: '#111827' }}>
                  {fact.value}
                </p>
              </div>
            ))}
          </div>

          {/* Description */}
          <div style={{ marginBottom: '20px' }}>
            <p style={{ margin: '0 0 4px', fontSize: '12px', color: '#6b7280' }}>Description</p>
            <p style={{ margin: 0, fontSize: '14px', color: '#111827', lineHeight: 1.6 }}>
              {inc.description}
            </p>
          </div>

          {/* AI confidence */}
          <div style={{ marginBottom: '20px' }}>
            <p style={{ margin: '0 0 4px', fontSize: '12px', color: '#6b7280' }}>AI confidence</p>
            <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827' }}>
              {confidence}
            </p>
          </div>

          {/* Outcome card */}
          <div style={{ backgroundColor: '#f9fafb', borderRadius: '10px', padding: '14px 16px' }}>

            {/* Status line: dot + outcome + divider + "Staff at HH:MM" */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span
                style={{
                  width: 8, height: 8, borderRadius: '50%',
                  backgroundColor: dotColor,
                  display: 'inline-block', flexShrink: 0,
                }}
              />
              <span style={{ fontSize: '14px', fontWeight: 600, color: '#111827', flexShrink: 0 }}>
                {inc.outcome}
              </span>
              <div style={{ flex: 1, height: '1px', backgroundColor: '#e5e8ed' }} />
              {assignee !== '—' && (
                <span style={{ fontSize: '13px', color: '#6b7280', flexShrink: 0 }}>
                  {assignee} at {reviewedAt(inc.dateTime)}
                </span>
              )}
            </div>

            {/* Remark */}
            {remarkText && (
              <p style={{ margin: '10px 0 0', fontSize: '14px', color: '#374151', lineHeight: 1.5 }}>
                {remarkText}
              </p>
            )}

            {/* Footer status */}
            {outcomeFooter && (
              <p style={{ margin: '8px 0 0', fontSize: '13px', color: '#1f4fd1', fontWeight: 500 }}>
                {outcomeFooter}
              </p>
            )}
          </div>
        </div>

        {/* ── Footer buttons ────────────────────────────────────────── */}
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
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
              backgroundColor: '#ffffff',
              border: '1px solid #d0d5dd',
              borderRadius: '8px',
              fontSize: '14px', fontWeight: 500, color: '#374151',
              cursor: 'pointer',
            }}
          >
            <Download size={14} />
            Download clip
          </button>
          <button
            type="button"
            style={{
              flex: 1, height: '39px',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
              backgroundColor: '#ffffff',
              border: '1px solid #d0d5dd',
              borderRadius: '8px',
              fontSize: '14px', fontWeight: 500, color: '#374151',
              cursor: 'pointer',
            }}
          >
            <SearchIcon size={14} />
            Open in Search
          </button>
        </div>
      </div>
    </>,
    document.body
  );
}
