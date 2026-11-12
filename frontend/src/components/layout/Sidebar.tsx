import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Eye,
  House,
  ClipboardList,
  Search,
  Radio,
  Camera,
  BarChart2,
  SlidersHorizontal,
  Settings,
  LogOut,
  User,
  ChevronDown,
} from 'lucide-react';
import { UserRole } from '../../types';
import { Toggle } from '../ui/Toggle';
import InviteStaffModal from '../modals/InviteStaffModal';

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
  { label: 'Home',           icon: House,         path: '/staff/home' },
  { label: 'Review History', icon: ClipboardList, path: '/staff/review-history' },
  { label: 'Search',         icon: Search,        path: '/staff/search' },
  { label: 'Hotspots',       icon: Radio,         path: '/staff/hotspots' },
  { label: 'Cameras',        icon: Camera,        path: '/staff/cameras' },
];

const ADMIN_NAV: NavItem[] = [
  { label: 'Home',     icon: House,             path: '/admin/home' },
  { label: 'Workload', icon: BarChart2,         path: '/admin/workload' },
  { label: 'Rules',    icon: SlidersHorizontal, path: '/admin/rules' },
  { label: 'Cameras',  icon: Camera,            path: '/admin/cameras' },
];

const USER_BY_ROLE: Record<UserRole, { initials: string; name: string; role: string }> = {
  staff: { initials: 'JT', name: 'Jane Tan', role: 'Staff' },
  admin: { initials: 'BL', name: 'Ben Lee',  role: 'Admin' },
};

// ── Admin settings panel helpers ───────────────────────────────────────────

function SettingsCard({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div style={{ backgroundColor: '#ffffff', borderRadius: '12px', paddingTop: title ? '14px' : '0', paddingBottom: '4px', paddingLeft: '16px', paddingRight: '16px', flexShrink: 0 }}>
      {title && <><p style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#121726' }}>{title}</p><div style={{ height: '8px' }} /></>}
      {children}
    </div>
  );
}

function SettingsRow({ label, sub, border = false, children }: { label: string; sub?: string; border?: boolean; children?: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingTop: '11px', paddingBottom: '11px', borderTop: border ? '1px solid #e5e8ed' : 'none' }}>
      <div style={{ flex: '1 0 0', minWidth: 0, display: 'flex', flexDirection: 'column', gap: sub ? '2px' : 0 }}>
        <p style={{ margin: 0, fontSize: '14px', fontWeight: 500, color: '#121726' }}>{label}</p>
        {sub && <p style={{ margin: 0, fontSize: '13px', fontWeight: 400, color: '#6b7380' }}>{sub}</p>}
      </div>
      {children}
    </div>
  );
}

// ── Staff popover helper ───────────────────────────────────────────────────

interface PopoverItemProps {
  icon: React.ReactNode;
  label: string;
  danger?: boolean;
  onClick: () => void;
}

function PopoverItem({ icon, label, danger = false, onClick }: PopoverItemProps) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        width: '100%',
        padding: '9px 10px',
        borderRadius: '8px',
        border: 'none',
        cursor: 'pointer',
        backgroundColor: hovered ? '#f9fafb' : 'transparent',
        color: danger ? '#dc2626' : '#374151',
        fontSize: '14px',
        fontWeight: 500,
        textAlign: 'left',
        transition: 'background-color 0.1s',
      }}
    >
      {icon}
      <span dangerouslySetInnerHTML={{ __html: label }} />
    </button>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────

export function Sidebar({ role, activePath }: SidebarProps) {
  const navigate = useNavigate();
  const navItems = role === 'staff' ? STAFF_NAV : ADMIN_NAV;
  const currentUser = USER_BY_ROLE[role];
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [popoverPos, setPopoverPos] = useState({ bottom: 0, left: 0 });

  // Admin notification toggles
  const [notifEscalated,  setNotifEscalated]  = useState(true);
  const [notifCamOffline, setNotifCamOffline] = useState(true);
  const [notifWeekly,     setNotifWeekly]     = useState(true);
  const [teamOpen,        setTeamOpen]        = useState(false);
  const [showInvite,      setShowInvite]      = useState(false);

  const settingsBtnRef = useRef<HTMLButtonElement>(null);
  const popoverRef     = useRef<HTMLDivElement>(null);

  // Close popover on outside click
  useEffect(() => {
    if (!popoverOpen) return;
    function handleClick(e: MouseEvent) {
      const target = e.target as Node;
      if (
        popoverRef.current && !popoverRef.current.contains(target) &&
        settingsBtnRef.current && !settingsBtnRef.current.contains(target)
      ) {
        setPopoverOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [popoverOpen]);

  function togglePopover() {
    if (popoverOpen) { setPopoverOpen(false); return; }
    if (settingsBtnRef.current) {
      const rect = settingsBtnRef.current.getBoundingClientRect();
      setPopoverPos({ bottom: window.innerHeight - rect.bottom, left: rect.right + 8 });
    }
    setPopoverOpen(true);
  }

  const isActive = (path: string) => activePath === path || activePath.startsWith(path);

  return (
    <>
    <aside
      className="flex flex-col items-center flex-shrink-0"
      style={{
        width: '72px',
        minHeight: '100vh',
        backgroundColor: '#ffffff',
        borderRight: '1px solid #e5e7eb',
        paddingTop: '24px',
        paddingBottom: '24px',
        position: 'sticky',
        top: 0,
        height: '100vh',
        zIndex: 100,
      }}
    >
      {/* Logo */}
      <div style={{ marginBottom: '8px' }}>
        <button
          type="button"
          onClick={() => navigate(role === 'staff' ? '/staff/home' : '/admin/home')}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '50px', height: '50px',
            backgroundColor: '#1f4fd1', borderRadius: '25px', border: 'none', cursor: 'pointer',
          }}
          aria-label="InSight home"
        >
          <Eye size={22} color="#ffffff" strokeWidth={2.5} />
        </button>
      </div>

      {/* Nav items */}
      <nav className="flex flex-col items-center flex-1" style={{ paddingTop: '32px', gap: '8px' }} aria-label="Main navigation">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          return (
            <button
              key={item.path}
              type="button"
              onClick={() => navigate(item.path)}
              title={item.label}
              aria-label={item.label}
              aria-current={active ? 'page' : undefined}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '72px', height: '52px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                backgroundColor: active ? '#1f4fd1' : 'transparent',
                color: active ? '#ffffff' : '#374151',
                transition: 'background-color 0.15s',
              }}
            >
              <Icon size={22} />
            </button>
          );
        })}
      </nav>

      {/* Bottom: Settings + Avatar */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
        <button
          ref={settingsBtnRef}
          type="button"
          onClick={togglePopover}
          title="Settings"
          aria-label="Settings"
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '72px', height: '52px', borderRadius: '12px', border: 'none', cursor: 'pointer',
            backgroundColor: popoverOpen ? '#1f4fd1' : 'transparent',
            color: popoverOpen ? '#ffffff' : '#374151',
            transition: 'background-color 0.15s',
          }}
        >
          <Settings size={22} />
        </button>

        <div
          title={currentUser.name}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '36px', height: '36px', borderRadius: '50%',
            backgroundColor: '#dbe4fb', color: '#1f4fd1', fontSize: '11px', fontWeight: 600,
          }}
        >
          {currentUser.initials}
        </div>

        {/* ── Staff popover ── */}
        {popoverOpen && role === 'staff' && (
          <div
            ref={popoverRef}
            style={{
              position: 'fixed',
              bottom: `${popoverPos.bottom}px`,
              left: `${popoverPos.left}px`,
              width: '220px',
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e5e7eb',
              boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              overflow: 'hidden',
              zIndex: 1000,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '14px 16px', borderBottom: '1px solid #f3f4f6' }}>
              <div style={{ width: '34px', height: '34px', borderRadius: '50%', backgroundColor: '#dbe4fb', color: '#1f4fd1', fontSize: '11px', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                {currentUser.initials}
              </div>
              <div style={{ minWidth: 0 }}>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{currentUser.name}</p>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', fontWeight: 400 }}>{currentUser.role}</p>
              </div>
            </div>
            <div style={{ padding: '6px' }}>
              <PopoverItem icon={<User size={15} />} label="Profile &amp; settings" onClick={() => setPopoverOpen(false)} />
              <div style={{ height: '1px', backgroundColor: '#f3f4f6', margin: '4px 0' }} />
              <PopoverItem icon={<LogOut size={15} />} label="Sign out" danger onClick={() => { setPopoverOpen(false); navigate('/login'); }} />
            </div>
          </div>
        )}

        {/* ── Admin settings panel ── */}
        {popoverOpen && role === 'admin' && (
          <div
            ref={popoverRef}
            style={{
              position: 'fixed',
              bottom: `${popoverPos.bottom}px`,
              left: `${popoverPos.left}px`,
              width: '380px',
              backgroundColor: '#f3f4f6',
              borderRadius: '16px',
              border: '1px solid #e0e3e8',
              boxShadow: '0px 12px 32px 0px rgba(18,23,38,0.16)',
              zIndex: 1000,
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              padding: '8px',
              maxHeight: `calc(100vh - ${popoverPos.bottom + 16}px)`,
              overflowY: 'auto',
            }}
          >
            <SettingsCard title="Notifications">
              <SettingsRow label="Escalated cases" sub="Desktop and email, as soon as staff escalate">
                <Toggle checked={notifEscalated} onChange={setNotifEscalated} label="Escalated cases notifications" />
              </SettingsRow>
              <SettingsRow label="Camera goes offline" sub="Desktop alert after 5 min without signal" border>
                <Toggle checked={notifCamOffline} onChange={setNotifCamOffline} label="Camera offline notifications" />
              </SettingsRow>
              <SettingsRow label="Weekly workload report" sub="Emailed every Monday, 9:00 AM" border>
                <Toggle checked={notifWeekly} onChange={setNotifWeekly} label="Weekly report notifications" />
              </SettingsRow>
            </SettingsCard>

            {/* Team — collapsible */}
            <div style={{ backgroundColor: '#ffffff', borderRadius: '12px', paddingLeft: '16px', paddingRight: '16px', flexShrink: 0 }}>
              <button
                type="button"
                onClick={() => setTeamOpen((v) => !v)}
                style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '14px 0 10px' }}
              >
                <span style={{ fontSize: '16px', fontWeight: 600, color: '#121726', marginRight: 'auto' }}>Team</span>
                {/* Stacked member avatars */}
                <div style={{ display: 'flex' }}>
                  {[{ initials: 'JT', bg: '#dbe4fb', color: '#1f4fd1' }, { initials: 'RK', bg: '#d1fae5', color: '#065f46' }].map((m, i) => (
                    <div key={m.initials} style={{ width: '24px', height: '24px', borderRadius: '50%', backgroundColor: m.bg, color: m.color, fontSize: '9px', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px solid #ffffff', marginLeft: i === 0 ? 0 : '-6px' }}>
                      {m.initials}
                    </div>
                  ))}
                </div>
                <ChevronDown size={16} color="#6b7280" style={{ transform: teamOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', flexShrink: 0 }} />
              </button>
              {teamOpen && (
                <>
                  <SettingsRow label="Jane Tan" sub="Staff · jane.tan@towera.sg">
                    <button type="button" style={{ color: '#6b7380', fontSize: '14px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0 }}>Remove</button>
                  </SettingsRow>
                  <SettingsRow label="Ravi K." sub="Staff · ravi.k@towera.sg" border>
                    <button type="button" style={{ color: '#6b7380', fontSize: '14px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0 }}>Remove</button>
                  </SettingsRow>
                </>
              )}
              <SettingsRow label="Invite staff" border>
                <button type="button" onClick={() => setShowInvite(true)} style={{ color: '#1f4fd1', fontSize: '14px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0 }}>+ Invite</button>
              </SettingsRow>
              <div style={{ height: '4px' }} />
            </div>

            <SettingsCard>
              <SettingsRow label="Customise layout" sub="Choose which cards show on Home">
                <button type="button" style={{ color: '#1f4fd1', fontSize: '14px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0 }}>Edit</button>
              </SettingsRow>
            </SettingsCard>

            <SettingsCard>
              <SettingsRow label="Sam Lee" sub="sam.lee@towera.sg">
                <span style={{ color: '#6b7380', fontSize: '14px', fontWeight: 400, flexShrink: 0 }}>Admin</span>
              </SettingsRow>
              <SettingsRow label="Change password" border>
                <button type="button" style={{ color: '#1f4fd1', fontSize: '14px', fontWeight: 500, background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0 }}>Change</button>
              </SettingsRow>
            </SettingsCard>

            <button
              type="button"
              onClick={() => { setPopoverOpen(false); navigate('/login'); }}
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #edbfba',
                borderRadius: '10px',
                padding: '10px 18px',
                width: '100%',
                color: '#b22921',
                fontSize: '14px',
                fontWeight: 500,
                cursor: 'pointer',
                flexShrink: 0,
              }}
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    </aside>

    <InviteStaffModal
      isOpen={showInvite}
      onClose={() => setShowInvite(false)}
      onInvite={() => setShowInvite(false)}
    />
    </>
  );
}
