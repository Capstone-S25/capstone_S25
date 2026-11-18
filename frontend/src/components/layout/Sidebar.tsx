import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Eye,
  House,
  History,
  Radio,
  Camera,
  BarChart3,
  ListChecks,
  SlidersHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  ChevronDown,
} from 'lucide-react';
import { UserRole } from '../../types';
import { Toggle } from '../ui/Toggle';
import InviteStaffModal from '../modals/InviteStaffModal';

// Figma: v2/Sidebar — brand lockup, labelled sections, Settings pinned to the bottom.

interface SidebarProps {
  role: UserRole;
  activePath: string;
}

interface NavItem {
  label: string;
  icon: React.ElementType;
  path: string;
}

const STAFF_NAV: NavItem[] = [
  { label: 'Home',           icon: House,   path: '/staff/home' },
  { label: 'Review history', icon: History, path: '/staff/review-history' },
  { label: 'Cameras',        icon: Camera,  path: '/staff/cameras' },
  { label: 'Hotspots',       icon: Radio,   path: '/staff/hotspots' },
];

const ADMIN_NAV: NavItem[] = [
  { label: 'Home',             icon: House,      path: '/admin/home' },
  { label: 'Workload summary', icon: BarChart3,  path: '/admin/workload' },
  { label: 'Monitoring rules', icon: ListChecks, path: '/admin/rules' },
  { label: 'Cameras',          icon: Camera,     path: '/admin/cameras' },
];

const COLLAPSE_KEY = 'insight.sidebarCollapsed';

function readCollapsed(): boolean {
  try { return localStorage.getItem(COLLAPSE_KEY) === '1'; } catch { return false; }
}

// ── Settings panel helpers ─────────────────────────────────────────────────

function SettingsSection({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-line-soft px-[16px] py-[12px] last:border-b-0">
      {title && <p className="m-0 mb-[4px] text-[11px] font-medium tracking-[0.02em] text-muted uppercase">{title}</p>}
      {children}
    </div>
  );
}

function SettingsRow({ label, sub, children }: { label: string; sub?: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-[16px] py-[8px]">
      <div className="min-w-0 flex-1">
        <p className="m-0 text-[13px] font-medium text-ink">{label}</p>
        {sub && <p className="m-0 mt-[1px] text-[12px] text-muted">{sub}</p>}
      </div>
      {children}
    </div>
  );
}

function LinkButton({ children, onClick, tone = 'brand' }: { children: React.ReactNode; onClick?: () => void; tone?: 'brand' | 'muted' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`shrink-0 cursor-pointer border-none bg-transparent p-0 text-[12px] font-medium hover:underline ${tone === 'brand' ? 'text-brand' : 'text-muted'}`}
    >
      {children}
    </button>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────

export function Sidebar({ role, activePath }: SidebarProps) {
  const navigate = useNavigate();
  const navItems = role === 'staff' ? STAFF_NAV : ADMIN_NAV;
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [notifEscalated, setNotifEscalated] = useState(true);
  const [notifCamOffline, setNotifCamOffline] = useState(true);
  const [notifWeekly, setNotifWeekly] = useState(true);
  const [notifNewAlert, setNotifNewAlert] = useState(true);
  const [teamOpen, setTeamOpen] = useState(false);
  const [showInvite, setShowInvite] = useState(false);

  const settingsBtnRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try { localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0'); } catch { /* storage unavailable */ }
  }, [collapsed]);

  // Close settings on outside click / Escape
  useEffect(() => {
    if (!settingsOpen) return;
    function onDown(e: MouseEvent) {
      const t = e.target as Node;
      if (panelRef.current?.contains(t) || settingsBtnRef.current?.contains(t)) return;
      setSettingsOpen(false);
    }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setSettingsOpen(false); }
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [settingsOpen]);

  const isActive = (path: string) => activePath === path || activePath.startsWith(path);

  const navBtn = (active: boolean) =>
    `flex w-full cursor-pointer items-center gap-[10px] rounded-[6px] border-none py-[9px] text-[13px] font-medium whitespace-nowrap transition-colors ${
      collapsed ? 'justify-center px-0' : 'px-[12px]'
    } ${active ? 'bg-brand-soft text-brand' : 'bg-transparent text-ink-2 hover:bg-head hover:text-ink'}`;

  return (
    <>
      <aside
        className="sticky top-0 z-[100] flex h-screen shrink-0 flex-col gap-[4px] border-r border-line bg-white pt-[16px] pb-[14px] transition-[width] duration-200"
        style={{ width: collapsed ? 64 : 232, paddingLeft: collapsed ? 10 : 12, paddingRight: collapsed ? 10 : 12 }}
      >
        {/* Brand lockup */}
        <div className={`flex items-center gap-[10px] pb-[14px] ${collapsed ? 'flex-col px-0' : 'px-[4px]'}`}>
          <button
            type="button"
            onClick={() => navigate(role === 'staff' ? '/staff/home' : '/admin/home')}
            aria-label="InSight home"
            className="flex size-[32px] shrink-0 cursor-pointer items-center justify-center rounded-[8px] border-none bg-brand"
          >
            <Eye size={19} color="#ffffff" strokeWidth={2.25} />
          </button>
          {!collapsed && (
            <div className="flex min-w-0 flex-1 flex-col whitespace-nowrap">
              <span className="text-[15px] font-semibold text-ink">InSight</span>
              <span className="text-[11px] text-muted">Panasonic · {role === 'staff' ? 'Staff' : 'Admin'}</span>
            </div>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="flex size-[24px] shrink-0 cursor-pointer items-center justify-center rounded-[4px] border-none bg-transparent text-muted transition-colors hover:bg-head hover:text-ink"
          >
            {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-[4px]" aria-label="Main navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.path);
            return (
              <button
                key={item.path}
                type="button"
                onClick={() => navigate(item.path)}
                title={collapsed ? item.label : undefined}
                aria-label={collapsed ? item.label : undefined}
                aria-current={active ? 'page' : undefined}
                className={navBtn(active)}
              >
                <Icon size={18} strokeWidth={1.75} className="shrink-0" />
                {!collapsed && item.label}
              </button>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* Settings */}
        <button
          ref={settingsBtnRef}
          type="button"
          onClick={() => setSettingsOpen((v) => !v)}
          title={collapsed ? 'Settings' : undefined}
          aria-label={collapsed ? 'Settings' : undefined}
          aria-expanded={settingsOpen}
          className={navBtn(settingsOpen)}
        >
          <SlidersHorizontal size={18} strokeWidth={1.75} className="shrink-0" />
          {!collapsed && 'Settings'}
        </button>
      </aside>

      {/* ── Settings panel ── */}
      {settingsOpen && (
        <div
          ref={panelRef}
          className="v2-pop-in thin-scroll fixed bottom-[14px] z-[1000] flex max-h-[calc(100vh-28px)] w-[360px] flex-col overflow-y-auto rounded-[10px] border border-line bg-white shadow-[0_12px_32px_rgba(18,26,43,0.16)]"
          style={{ left: (collapsed ? 64 : 232) + 8 }}
        >
          <div className="border-b border-line-soft px-[16px] pt-[14px] pb-[12px]">
            <p className="m-0 text-[15px] font-medium text-ink">Settings</p>
          </div>

          <SettingsSection title="Notifications">
            {role === 'admin' ? (
              <>
                <SettingsRow label="Escalated cases" sub="Desktop and email, as soon as staff escalate">
                  <Toggle checked={notifEscalated} onChange={setNotifEscalated} label="Escalated cases notifications" />
                </SettingsRow>
                <SettingsRow label="Camera goes offline" sub="Desktop alert after 5 min without signal">
                  <Toggle checked={notifCamOffline} onChange={setNotifCamOffline} label="Camera offline notifications" />
                </SettingsRow>
                <SettingsRow label="Weekly workload report" sub="Emailed every Monday, 9:00 AM">
                  <Toggle checked={notifWeekly} onChange={setNotifWeekly} label="Weekly report notifications" />
                </SettingsRow>
              </>
            ) : (
              <>
                <SettingsRow label="New clips to review" sub="Desktop alert when a clip is flagged">
                  <Toggle checked={notifNewAlert} onChange={setNotifNewAlert} label="New clip notifications" />
                </SettingsRow>
                <SettingsRow label="Camera goes offline" sub="Desktop alert after 5 min without signal">
                  <Toggle checked={notifCamOffline} onChange={setNotifCamOffline} label="Camera offline notifications" />
                </SettingsRow>
              </>
            )}
          </SettingsSection>

          {role === 'admin' && (
            <SettingsSection title="Team">
              <button
                type="button"
                onClick={() => setTeamOpen((v) => !v)}
                aria-expanded={teamOpen}
                className="flex w-full cursor-pointer items-center gap-[10px] border-none bg-transparent py-[8px] text-left"
              >
                <span className="flex-1 text-[13px] font-medium text-ink">2 staff members</span>
                <span className="flex">
                  {[{ i: 'JT', bg: 'var(--color-brand-soft)', fg: 'var(--color-brand)' }, { i: 'RK', bg: 'var(--color-success-soft)', fg: 'var(--color-success)' }].map((m, idx) => (
                    <span
                      key={m.i}
                      className="flex size-[24px] items-center justify-center rounded-full border-2 border-white text-[9px] font-semibold"
                      style={{ backgroundColor: m.bg, color: m.fg, marginLeft: idx === 0 ? 0 : -6 }}
                    >
                      {m.i}
                    </span>
                  ))}
                </span>
                <ChevronDown size={16} className="shrink-0 text-muted transition-transform" style={{ transform: teamOpen ? 'rotate(180deg)' : undefined }} />
              </button>
              {teamOpen && (
                <>
                  <SettingsRow label="Jane Tan" sub="Staff · jane.tan@towera.sg"><LinkButton tone="muted">Remove</LinkButton></SettingsRow>
                  <SettingsRow label="Ravi K." sub="Staff · ravi.k@towera.sg"><LinkButton tone="muted">Remove</LinkButton></SettingsRow>
                </>
              )}
              <SettingsRow label="Invite staff">
                <LinkButton onClick={() => setShowInvite(true)}>+ Invite</LinkButton>
              </SettingsRow>
            </SettingsSection>
          )}

          <SettingsSection title="Account">
            <SettingsRow label="Change password">
              <LinkButton>Change</LinkButton>
            </SettingsRow>
          </SettingsSection>
        </div>
      )}

      <InviteStaffModal isOpen={showInvite} onClose={() => setShowInvite(false)} />    </>
  );
}
