import { useEffect, useState } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { getMe, homePath, type Role, type SessionUser } from '../../data/auth';
import { SessionContext } from './session';

/**
 * Shows the pages inside it only to a signed-in user (with this role, if given).
 * Signed-out users go to /login, users on a temporary password to /set-password,
 * and staff opening an admin page to their own home page. The session is checked
 * again on every navigation, so an expired or revoked one is noticed.
 */
export function RequireAuth({ role }: { role?: Role }) {
  const { pathname } = useLocation();
  // undefined while the first check runs; null when nobody is signed in.
  const [user, setUser] = useState<SessionUser | null | undefined>(undefined);

  useEffect(() => {
    let current = true;
    getMe()
      .then((me) => { if (current) setUser(me); })
      .catch(() => { if (current) setUser(null); });
    return () => { current = false; };
  }, [pathname]);

  if (user === undefined) return null;
  if (user === null) return <Navigate to="/login" replace />;
  if (user.must_change_password) return <Navigate to="/set-password" replace />;
  if (role && user.role !== role) return <Navigate to={homePath(user)} replace />;
  return (
    <SessionContext.Provider value={user}>
      <Outlet />
    </SessionContext.Provider>
  );
}
