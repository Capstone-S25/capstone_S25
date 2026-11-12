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
} from 'lucide-react';
import { UserRole } from '../../types';

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
  { label: 'Home',           icon: House,             path: '/staff/home' },
  { label: 'Review History', icon: ClipboardList,     path: '/staff/review-history' },
  { label: 'Search',         icon: Search,            path: '/staff/search' },
  { label: 'Hotspots',       icon: Radio,             path: '/staff/hotspots' },
  { label: 'Cameras',        icon: Camera,            path: '/staff/cameras' },
];

const ADMIN_NAV: NavItem[] = [
  { label: 'Home',     icon: House,             path: '/admin/home' },
  { label: 'Workload', icon: BarChart2,         path: '/admin/workload' },
  { label: 'Rules',    icon: SlidersHorizontal, path: '/admin/rules' },
  { label: 'Cameras',  icon: Camera,            path: '/admin/cameras' },
];

const CURRENT_USER = { initials: 'JT', name: 'Jane Tan', role: 'Staff' };

export function Sidebar({ role, activePath }: SidebarProps) {
  const navigate = useNavigate();
  const navItems = role === 'staff' ? STAFF_NAV : ADMIN_NAV;
  const [popoverOpen, setPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close popover on outside click
  useEffect(() => {
    if (!popoverOpen) return;
    function handleClick(e: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setPopoverOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [popoverOpen]);

  const isActive = (path: string) => activePath === path || activePath.startsWith(path);

  return (
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
      }}
    >
      {/* Logo */}
      <div style={{ marginBottom: '8px' }}>
        <button
          type="button"
          onClick={() => navigate(role === 'staff' ? '/staff/home' : '/admin/home')}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '50px',
            height: '50px',
            backgroundColor: '#1f4fd1',
            borderRadius: '25px',
            border: 'none',
            cursor: 'pointer',
          }}
          aria-label="InSight home"
        >
          <Eye size={22} color="#ffffff" strokeWidth={2.5} />
        </button>
      </div>

      {/* Nav items */}
      <nav
        className="flex flex-col items-center flex-1"
        style={{ paddingTop: '32px', gap: '8px' }}
        aria-label="Main navigation"
      >
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
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '72px',
                height: '52px',
                borderRadius: '12px',
                border: 'none',
                cursor: 'pointer',
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

      {/* Bottom: Settings + Avatar — both toggle the same popover */}
      <div ref={popoverRef} style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>

        {/* Settings icon button */}
        <button
          type="button"
          onClick={() => setPopoverOpen((o) => !o)}
          title="Settings"
          aria-label="Settings"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '72px',
            height: '52px',
            borderRadius: '12px',
            border: 'none',
            cursor: 'pointer',
            backgroundColor: popoverOpen ? '#1f4fd1' : 'transparent',
            color: popoverOpen ? '#ffffff' : '#374151',
            transition: 'background-color 0.15s',
          }}
        >
          <Settings size={22} />
        </button>

        {/* Avatar button */}
        <button
          type="button"
          onClick={() => setPopoverOpen((o) => !o)}
          title={CURRENT_USER.name}
          aria-label={`User: ${CURRENT_USER.name}`}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            border: popoverOpen ? '2px solid #1f4fd1' : '2px solid transparent',
            cursor: 'pointer',
            backgroundColor: '#dbe4fb',
            color: '#1f4fd1',
            fontSize: '11px',
            fontWeight: 600,
            transition: 'border-color 0.15s',
          }}
        >
          {CURRENT_USER.initials}
        </button>

        {/* Popover */}
        {popoverOpen && (
          <div
            style={{
              position: 'absolute',
              bottom: 0,
              left: '80px',
              width: '220px',
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e5e7eb',
              boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              overflow: 'hidden',
              zIndex: 50,
            }}
          >
            {/* User info */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '14px 16px',
                borderBottom: '1px solid #f3f4f6',
              }}
            >
              <div
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '50%',
                  backgroundColor: '#dbe4fb',
                  color: '#1f4fd1',
                  fontSize: '11px',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                {CURRENT_USER.initials}
              </div>
              <div style={{ minWidth: 0 }}>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {CURRENT_USER.name}
                </p>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', fontWeight: 400 }}>
                  {CURRENT_USER.role}
                </p>
              </div>
            </div>

            {/* Menu items */}
            <div style={{ padding: '6px' }}>
              <PopoverItem
                icon={<User size={15} />}
                label="Profile &amp; settings"
                onClick={() => { setPopoverOpen(false); }}
              />
              <div style={{ height: '1px', backgroundColor: '#f3f4f6', margin: '4px 0' }} />
              <PopoverItem
                icon={<LogOut size={15} />}
                label="Sign out"
                danger
                onClick={() => { setPopoverOpen(false); navigate('/login'); }}
              />
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

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
