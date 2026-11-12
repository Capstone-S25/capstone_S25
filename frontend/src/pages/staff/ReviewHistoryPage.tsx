import { useState } from 'react';
import { Search, Download, ChevronDown } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { SeverityBadge, OutcomeDot, OutcomeTabs } from '../../components/ui/Badge';
import { MOCK_INCIDENTS } from '../../data/mock';
import { MOCK_STAFF } from '../../data/mock';
import type { IncidentOutcome } from '../../types';

type Tab = 'All' | IncidentOutcome;

const TABS: Tab[] = ['All', 'Accepted', 'Dismissed', 'Escalated'];

const STAFF_MAP: Record<string, string> = Object.fromEntries(MOCK_STAFF.map((s) => [s.id, s.name]));

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
    time: d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
  };
}

export default function ReviewHistoryPage() {
  const [activeTab, setActiveTab] = useState<Tab>('All');
  const [severity, setSeverity] = useState('all');
  const [location, setLocation] = useState('all');
  const [period, setPeriod] = useState('7d');
  const [searchQuery, setSearchQuery] = useState('');

  const tabCounts: Partial<Record<Tab, number>> = {
    All: MOCK_INCIDENTS.length,
    Accepted: MOCK_INCIDENTS.filter((i) => i.outcome === 'Accepted').length,
    Dismissed: MOCK_INCIDENTS.filter((i) => i.outcome === 'Dismissed').length,
    Escalated: MOCK_INCIDENTS.filter((i) => i.outcome === 'Escalated').length,
  };

  const filtered = MOCK_INCIDENTS.filter((inc) => {
    if (activeTab !== 'All' && inc.outcome !== activeTab) return false;
    if (severity !== 'all' && inc.severity !== severity) return false;
    if (location !== 'all' && inc.location !== location) return false;
    if (searchQuery && !inc.type.toLowerCase().includes(searchQuery.toLowerCase()) && !inc.location.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  return (
    <AppLayout role="staff" activePath="/staff/review-history">
      <div
        style={{
          padding: '28px 28px 28px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          height: '100%',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 600, color: '#111827', margin: 0 }}>Review History</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* Search bar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: '#ffffff',
                paddingLeft: '14px',
                paddingRight: '16px',
                paddingTop: '12px',
                paddingBottom: '12px',
                borderRadius: '10px',
                width: '360px',
                border: '1px solid #e5e7eb',
              }}
            >
              <Search size={16} color="#6b7280" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search incidents…"
                style={{
                  flex: 1,
                  border: 'none',
                  outline: 'none',
                  fontSize: '14px',
                  color: '#111827',
                  backgroundColor: 'transparent',
                }}
              />
            </div>
            {/* Export button */}
            <button
              type="button"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '8px',
                padding: '10px 16px',
                fontSize: '14px',
                fontWeight: 500,
                color: '#111827',
                cursor: 'pointer',
              }}
            >
              <Download size={14} />
              Export
            </button>
          </div>
        </div>

        {/* Tabs + filter row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          {/* OutcomeTabs pill container */}
          <OutcomeTabs<Tab>
            tabs={TABS}
            active={activeTab}
            counts={tabCounts}
            onChange={setActiveTab}
          />

          {/* Filter dropdowns */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <FilterDropdown label="Severity" value={severity} onChange={setSeverity}>
              <option value="all">All</option>
              <option value="Critical">Critical</option>
              <option value="Moderate">Moderate</option>
              <option value="Minor">Minor</option>
            </FilterDropdown>
            <FilterDropdown label="Location" value={location} onChange={setLocation}>
              <option value="all">All locations</option>
              <option value="Carpark B2">Carpark B2</option>
              <option value="Lift Lobby 1">Lift Lobby 1</option>
              <option value="Corridor 2">Corridor 2</option>
              <option value="Main entrance">Main entrance</option>
              <option value="Meeting Room 1">Meeting Room 1</option>
              <option value="Staircase B">Staircase B</option>
              <option value="Pantry L3">Pantry L3</option>
            </FilterDropdown>
            <FilterDropdown label="Date" value={period} onChange={setPeriod}>
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </FilterDropdown>
          </div>
        </div>

        {/* Table — flex:1 keeps page height stable when rows change */}
        <div
          style={{
            backgroundColor: '#ffffff',
            borderRadius: '14px',
            overflow: 'hidden',
            flex: 1,
          }}
        >
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle('112px')}>Clip</th>
                <th style={thStyle()}>Incident</th>
                <th style={thStyle('90px')}>Camera</th>
                <th style={thStyle('180px')}>Date &amp; Time</th>
                <th style={thStyle('110px')}>Severity</th>
                <th style={thStyle('120px')}>Outcome</th>
                <th style={thStyle('120px')}>Reviewed by</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((inc) => {
                const { date, time } = formatDateTime(inc.dateTime);
                const assignee = inc.assignedTo ? STAFF_MAP[inc.assignedTo] ?? '—' : '—';
                return (
                  <tr
                    key={inc.id}
                    style={{ borderTop: '1px solid #eef0f3', cursor: 'pointer' }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {/* Clip thumbnail */}
                    <td style={tdStyle}>
                      <div style={{ position: 'relative', width: '96px', height: '54px', flexShrink: 0 }}>
                        <div
                          style={{
                            width: '96px',
                            height: '54px',
                            backgroundColor: '#d1d5db',
                            borderRadius: '4px',
                          }}
                        />
                        <span
                          style={{
                            position: 'absolute',
                            bottom: '3px',
                            right: '3px',
                            backgroundColor: 'rgba(0,0,0,0.55)',
                            color: '#ffffff',
                            fontSize: '11px',
                            padding: '1px 4px',
                            borderRadius: '3px',
                          }}
                        >
                          {inc.duration ?? '—'}
                        </span>
                      </div>
                    </td>
                    {/* Incident */}
                    <td style={tdStyle}>
                      <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827' }}>{inc.type}</p>
                      <p style={{ margin: '2px 0 0 0', fontSize: '13px', color: '#6b7280' }}>{inc.location}</p>
                    </td>
                    {/* Camera */}
                    <td style={tdStyle}>
                      <span style={{ fontSize: '14px', color: '#374151' }}>
                        {inc.camId.replace('cam-', 'Cam ')}
                      </span>
                    </td>
                    {/* Date & time */}
                    <td style={tdStyle}>
                      <p style={{ margin: 0, fontSize: '14px', color: '#374151' }}>{time}</p>
                      <p style={{ margin: '2px 0 0 0', fontSize: '13px', color: '#6b7280' }}>{date}</p>
                    </td>
                    {/* Severity */}
                    <td style={tdStyle}>
                      <SeverityBadge severity={inc.severity} />
                    </td>
                    {/* Outcome */}
                    <td style={tdStyle}>
                      <OutcomeDot outcome={inc.outcome} />
                    </td>
                    {/* Reviewed by */}
                    <td style={tdStyle}>
                      <p style={{ margin: 0, fontSize: '14px', color: '#374151' }}>{assignee}</p>
                      <p style={{ margin: '2px 0 0 0', fontSize: '13px', color: '#6b7280' }}>
                        {time}
                      </p>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <div
              style={{
                textAlign: 'center',
                padding: '48px 0',
                fontSize: '14px',
                color: '#9ca3af',
              }}
            >
              No incidents match the selected filters.
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}

// ── Style helpers ─────────────────────────────────────────────────────────────

function thStyle(width?: string): React.CSSProperties {
  return {
    textAlign: 'left',
    padding: '14px 16px',
    fontSize: '12px',
    fontWeight: 500,
    color: '#6b7280',
    backgroundColor: '#ffffff',
    ...(width ? { width } : {}),
  };
}

const tdStyle: React.CSSProperties = {
  padding: '8px 16px',
  verticalAlign: 'middle',
};

// ── FilterDropdown ─────────────────────────────────────────────────────────────

interface FilterDropdownProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}

function FilterDropdown({ label, value, onChange, children }: FilterDropdownProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{ fontSize: '14px', color: '#4b5563', flexShrink: 0 }}>{label}</span>
      <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{
            appearance: 'none',
            backgroundColor: '#ffffff',
            border: '1px solid #d0d5dd',
            borderRadius: '8px',
            paddingLeft: '12px',
            paddingRight: '32px',
            paddingTop: '9px',
            paddingBottom: '9px',
            fontSize: '14px',
            color: '#101828',
            boxShadow: '0px 1px 2px 0px rgba(16,24,40,0.05)',
            cursor: 'pointer',
            outline: 'none',
          }}
        >
          {children}
        </select>
        <ChevronDown
          size={14}
          color="#6b7280"
          style={{ position: 'absolute', right: '10px', pointerEvents: 'none' }}
        />
      </div>
    </div>
  );
}
