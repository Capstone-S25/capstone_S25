import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { UserRole } from '../../types';

interface AppLayoutProps {
  role: UserRole;
  activePath: string;
  children: ReactNode;
}

export function AppLayout({ role, activePath, children }: AppLayoutProps) {
  return (
    <div className="flex min-h-screen" style={{ backgroundColor: '#f3f4f6' }}>
      <Sidebar role={role} activePath={activePath} />
      <main
        className="flex-1 min-w-0"
        style={{
          backgroundColor: '#f3f4f6',
          height: '100vh',
          overflowY: 'scroll',
        }}
      >
        {children}
      </main>
    </div>
  );
}
