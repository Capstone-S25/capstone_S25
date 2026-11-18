import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { Search, Bell, ChevronDown, LogOut, UserRound } from 'lucide-react';
import { UserRole } from '../../types';

// Figma: v2/Top bar — global search, notifications and account.

const USER_BY_ROLE: Record<UserRole, { initials: string; name: string; email: string; role: string }> = {
  staff: { initials: 'JT', name: 'Jane Tan', email: 'jane.tan@towera.sg', role: 'Staff' },
  admin: { initials: 'SL', name: 'Sam Lee', email: 'sam.lee@towera.sg', role: 'Admin' },
};

interface Notification {
  id: number;
  bold: string;
  text: string;
  time: string;
  unread: boolean;
}

const NOTIFICATIONS: Record<UserRole, Notification[]> = {
  staff: [
    { id: 1, bold: 'New alert: ', text: 'Smoking · Carpark B2', time: '2 min ago', unread: true },
    { id: 2, bold: 'Sam Lee ', text: 'resolved your escalation · Smoking · Staircase B', time: '1 h ago', unread: true },
    { id: 3, bold: 'Cam 08 · Pantry L3 ', text: 'went offline', time: '6:12 PM', unread: false },
  ],
  admin: [
    { id: 1, bold: 'Jane Tan ', text: 'escalated Smoking · Carpark B2', time: '4 min ago', unread: true },
    { id: 2, bold: 'Cam 08 · Pantry L3 ', text: 'went offline', time: '6:12 PM', unread: true },
    { id: 3, bold: 'Weekly report ', text: 'is ready to view', time: 'Mon 9:00 AM', unread: false },
  ],
};

function useOutsideClose(open: boolean, close: () => void, refs: React.RefObject<HTMLElement | null>[]) {
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (refs.some((r) => r.current?.contains(e.target as Node))) return;
      close();
    }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') close(); }
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open, close, refs]);
}

export function TopBar({ role }: { role: UserRole }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const user = USER_BY_ROLE[role];
  const onSearchPage = location.pathname === '/staff/search';

  const [query, setQuery] = useState(onSearchPage ? params.get('q') ?? '' : '');
  const [notifs, setNotifs] = useState(NOTIFICATIONS[role]);
  const [bellOpen, setBellOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const bellRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  useOutsideClose(bellOpen, () => setBellOpen(false), [bellRef]);
  useOutsideClose(menuOpen, () => setMenuOpen(false), [menuRef]);

  const unread = notifs.filter((n) => n.unread).length;

  function submitSearch(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    navigate(q ? `/staff/search?q=${encodeURIComponent(q)}` : '/staff/search');
  }

  return (
    <header className="sticky top-0 z-[90] flex h-[64px] shrink-0 items-center gap-[12px] border-b border-line bg-white px-[24px]">
      {role === 'staff' && (
        <form
          onSubmit={submitSearch}
          role="search"
          className="flex w-full max-w-[520px] items-center gap-[8px] rounded-[6px] border border-line bg-white py-[5px] pr-[6px] pl-[14px] transition-colors focus-within:border-brand"
        >
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search clip IDs, places or incident types…"
            aria-label="Search clips"
            className="min-w-0 flex-1 border-none bg-transparent text-[13px] text-ink outline-none placeholder:text-faint"
          />
          <button
            type="submit"
            aria-label="Search"
            className="flex size-[30px] shrink-0 cursor-pointer items-center justify-center rounded-full border-none bg-brand text-white transition-colors hover:bg-brand-hover"
          >
            <Search size={14} strokeWidth={2.5} />
          </button>
        </form>
      )}

      <div className="flex-1" />

      {/* Notifications */}
      <div ref={bellRef} className="relative">
        <button
          type="button"
          onClick={() => setBellOpen((v) => !v)}
          aria-label={`Notifications${unread ? `, ${unread} unread` : ''}`}
          aria-expanded={bellOpen}
          className={`relative flex size-[36px] cursor-pointer items-center justify-center rounded-full border-none bg-transparent transition-colors hover:bg-head ${bellOpen ? 'text-brand' : 'text-ink'}`}
        >
          <Bell size={18} strokeWidth={1.75} />
          {unread > 0 && (
            <span className="absolute top-[2px] left-[18px] flex size-[16px] ring-2 ring-white items-center justify-center rounded-full bg-danger text-[10px] font-semibold text-white">
              {unread}
            </span>
          )}
        </button>

        {bellOpen && (
          <div className="v2-pop-in absolute top-[48px] right-0 z-[1000] w-[340px] overflow-hidden rounded-[10px] border border-line bg-white shadow-[0_12px_32px_rgba(18,26,43,0.16)]">
            <div className="flex items-center gap-[8px] border-b border-line-soft px-[16px] py-[12px]">
              <p className="m-0 text-[15px] font-medium text-ink">Notifications</p>
              {unread > 0 && <span className="text-[12px] text-muted">{unread} new</span>}
              <div className="flex-1" />
              <button
                type="button"
                disabled={unread === 0}
                onClick={() => setNotifs((ns) => ns.map((n) => ({ ...n, unread: false })))}
                className="cursor-pointer border-none bg-transparent p-0 text-[12px] font-medium text-brand hover:underline disabled:cursor-default disabled:text-faint disabled:no-underline"
              >
                Mark all read
              </button>
            </div>
            <ul className="m-0 list-none p-[6px]">
              {notifs.map((n) => (
                <li key={n.id}>
                  <button
                    type="button"
                    onClick={() => setNotifs((ns) => ns.map((x) => (x.id === n.id ? { ...x, unread: false } : x)))}
                    className="flex w-full cursor-pointer gap-[10px] rounded-[6px] border-none bg-transparent px-[10px] py-[8px] text-left transition-colors hover:bg-head"
                  >
                    <span className="flex h-[18px] w-[7px] shrink-0 items-center">
                      <span className={`size-[7px] rounded-full ${n.unread ? 'bg-brand' : 'bg-transparent'}`} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-[13px] text-ink">
                        <span className="font-medium">{n.bold}</span>
                        {n.text}
                      </span>
                      <span className="mt-[1px] block text-[12px] text-muted">{n.time}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Account */}
      <div ref={menuRef} className="relative">
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-expanded={menuOpen}
          aria-label="Account menu"
          className="flex cursor-pointer items-center gap-[10px] rounded-[6px] border-none bg-transparent py-[2px] pr-[2px] pl-[8px] transition-colors hover:bg-head"
        >
          <span className="text-[13px] font-medium whitespace-nowrap text-ink">{user.name}</span>
          <span className="flex size-[36px] items-center justify-center rounded-full bg-brand-soft text-[12px] font-semibold text-brand">
            {user.initials}
          </span>
          <ChevronDown size={14} className="text-ink-2 transition-transform" style={{ transform: menuOpen ? 'rotate(180deg)' : undefined }} />
        </button>

        {menuOpen && (
          <div className="v2-pop-in absolute top-[48px] right-0 z-[1000] w-[240px] overflow-hidden rounded-[10px] border border-line bg-white shadow-[0_12px_32px_rgba(18,26,43,0.16)]">
            <div className="border-b border-line-soft px-[16px] py-[12px]">
              <p className="m-0 truncate text-[13px] font-medium text-ink">{user.name}</p>
              <p className="m-0 truncate text-[12px] text-muted">{user.role} · {user.email}</p>
            </div>
            <div className="p-[6px]">
              <MenuItem icon={<UserRound size={15} />} onClick={() => setMenuOpen(false)}>Profile</MenuItem>
              <MenuItem icon={<LogOut size={15} />} danger onClick={() => { setMenuOpen(false); navigate('/login'); }}>Sign out</MenuItem>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}

function MenuItem({ icon, children, onClick, danger = false }: { icon: React.ReactNode; children: React.ReactNode; onClick: () => void; danger?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full cursor-pointer items-center gap-[10px] rounded-[6px] border-none bg-transparent px-[10px] py-[8px] text-left text-[13px] font-medium transition-colors hover:bg-head ${danger ? 'text-danger' : 'text-ink-2'}`}
    >
      {icon}
      {children}
    </button>
  );
}
