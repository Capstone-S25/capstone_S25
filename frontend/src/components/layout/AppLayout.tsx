import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { UserRole } from '../../types';

interface AppLayoutProps {
  role: UserRole;
  activePath: string;
  children: ReactNode;
  /**
   * 'page'  — standard padded, scrolling content column (default).
   * 'bare'  — children fill the area below the top bar with no padding/scroll,
   *           for layouts with their own panels (e.g. Home with side panel).
   */
  variant?: 'page' | 'bare';
}

export function AppLayout({ role, activePath, children, variant = 'page' }: AppLayoutProps) {
  return (
    <div className="flex h-screen bg-white">
      <Sidebar role={role} activePath={activePath} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar role={role} />
        {variant === 'bare' ? (
          <main className="flex min-h-0 flex-1">{children}</main>
        ) : (
          <main className="thin-scroll min-h-0 flex-1 overflow-y-auto">
            <div className="flex flex-col gap-[20px] px-[28px] pt-[20px] pb-[28px]">{children}</div>
          </main>
        )}
      </div>
    </div>
  );
}
