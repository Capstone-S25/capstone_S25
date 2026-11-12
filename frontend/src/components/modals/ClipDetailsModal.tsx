import { useEffect } from 'react';
import { X, Download, Search } from 'lucide-react';
import { Incident } from '../../types/index';
import { SeverityBadge } from '../ui/Badge';
import { Button } from '../ui/Button';

interface ClipDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
}

export default function ClipDetailsModal({ isOpen, onClose, incident }: ClipDetailsModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  if (!isOpen || !incident) return null;

  const dateObj = new Date(incident.dateTime);
  const formattedDate = dateObj.toLocaleDateString('en-GB', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
  const formattedTime = dateObj.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  const camLabel = incident.camId
    .replace('cam-0', 'Cam 0')
    .replace('cam-', 'Cam ');

  const isEscalated = incident.outcome === 'Escalated';
  const confidencePct = Math.round(incident.confidence * 100);

  return (
    <>
      {/* Dark overlay */}
      <div
        className="fixed inset-0 z-40"
        style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Right-side panel */}
      <div
        className="fixed inset-y-0 right-0 bg-white shadow-2xl z-50 flex flex-col overflow-y-auto"
        style={{ width: '480px' }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-gray-100 flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <SeverityBadge severity={incident.severity} />
            <span className="text-sm font-semibold text-gray-900 truncate">
              {incident.type} · {incident.location}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 flex-shrink-0 ml-2"
            aria-label="Close panel"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 px-6 py-5 space-y-5">
          {/* 16:9 video placeholder */}
          <div className="relative w-full rounded-xl overflow-hidden bg-gray-800" style={{ aspectRatio: '16/9' }}>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg className="text-gray-600" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <polygon points="10,8 16,12 10,16" fill="currentColor" stroke="none" />
              </svg>
            </div>
            {/* Camera label badge - top left */}
            <div className="absolute top-3 left-3">
              <span
                className="text-xs font-medium text-white px-2 py-1 rounded-md"
                style={{ backgroundColor: 'rgba(0,0,0,0.6)' }}
              >
                {camLabel} · {formattedTime}
              </span>
            </div>
            {/* Duration badge - bottom right */}
            <div className="absolute bottom-3 right-3">
              <span
                className="text-xs font-medium text-white px-2 py-1 rounded-md"
                style={{ backgroundColor: 'rgba(0,0,0,0.6)' }}
              >
                {incident.duration ?? '—'}
              </span>
            </div>
          </div>

          {/* Metadata row */}
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="space-y-0.5">
              <p className="text-gray-400 font-medium uppercase tracking-wide" style={{ fontSize: '10px' }}>Camera</p>
              <p className="text-gray-700 font-medium">{camLabel}</p>
            </div>
            <div className="space-y-0.5">
              <p className="text-gray-400 font-medium uppercase tracking-wide" style={{ fontSize: '10px' }}>Date &amp; time</p>
              <p className="text-gray-700 font-medium">{formattedDate}, {formattedTime}</p>
            </div>
            <div className="space-y-0.5">
              <p className="text-gray-400 font-medium uppercase tracking-wide" style={{ fontSize: '10px' }}>Duration</p>
              <p className="text-gray-700 font-medium">{incident.duration ?? '—'}</p>
            </div>
          </div>

          {/* Description */}
          <div className="space-y-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide" style={{ fontSize: '11px' }}>Description</p>
            <p className="text-sm text-gray-700 leading-relaxed">{incident.description}</p>
          </div>

          {/* AI Confidence */}
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide" style={{ fontSize: '11px' }}>AI Confidence</p>
            <span className="text-sm font-semibold text-gray-900">{confidencePct}%</span>
          </div>

          {/* Escalated section */}
          {isEscalated && (
            <div
              className="rounded-xl p-4 space-y-2"
              style={{ backgroundColor: '#eef2fb', borderLeft: '4px solid #1f4fd1' }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="text-xs font-semibold uppercase tracking-wide"
                  style={{ color: '#1f4fd1' }}
                >
                  Escalated
                </span>
              </div>
              <p className="text-sm font-medium text-gray-800">
                {incident.assignedTo === 's1' ? 'Jane Tan' : 'Raj A.'} at{' '}
                {new Date(new Date(incident.dateTime).getTime() + 2 * 60000).toLocaleTimeString('en-US', {
                  hour: 'numeric',
                  minute: '2-digit',
                  hour12: true,
                })}
              </p>
              {incident.remark && (
                <p className="text-sm text-gray-600 italic">"{incident.remark}"</p>
              )}
              <span
                className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full"
                style={{ backgroundColor: '#dbeafe', color: '#1d4ed8' }}
              >
                Open · waiting on admin
              </span>
            </div>
          )}
        </div>

        {/* Footer buttons */}
        <div className="flex gap-3 px-6 py-4 border-t border-gray-100 flex-shrink-0">
          <Button variant="secondary" className="flex-1 gap-2">
            <Download size={15} />
            Download clip
          </Button>
          <Button variant="primary" className="flex-1 gap-2">
            <Search size={15} />
            Open in Search
          </Button>
        </div>
      </div>
    </>
  );
}
