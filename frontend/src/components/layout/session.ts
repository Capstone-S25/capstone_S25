import { createContext, useContext } from 'react';
import type { SessionUser } from '../../data/auth';

export const SessionContext = createContext<SessionUser | null>(null);

/** The signed-in user, inside the pages that RequireAuth guards. */
export function useSessionUser(): SessionUser | null {
  return useContext(SessionContext);
}
