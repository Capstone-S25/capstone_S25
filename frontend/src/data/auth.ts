// Login and staff accounts (/api/v1/auth, /me, /users): cayla-backend's routes,
// served by the dashboard API. The login is an httpOnly cookie that the browser
// sends by itself, also with the incidents requests; in development Vite's proxy
// (vite.config.ts) keeps every request on the dashboard's own address.

export type Role = 'admin' | 'staff';

export interface SessionUser {
  user_id: number;
  name: string;
  email: string;
  role: Role;
  staff_id: string | null;
  must_change_password: boolean;
}

/** A failed request; message is the backend's own explanation, ready to show. */
export class AuthError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') return data.detail;
    // A 422 from FastAPI's own checks: a list, one entry per bad field.
        if (Array.isArray(data.detail) && data.detail[0]?.msg) {
      const field = data.detail[0].loc?.at(-1);
      return field ? `${data.detail[0].msg}: ${field}` : data.detail[0].msg;
    }
  } catch {
    // Not JSON: fall through to the general message.
  }
  return `Something went wrong (error ${response.status}). Please try again.`;
}

async function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      method,
      credentials: 'include',
      headers: body === undefined
        ? { Accept: 'application/json' }
        : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new AuthError("Can't reach the server. Check that the backend is running.", 0);
  }
  if (!response.ok) throw new AuthError(await errorMessage(response), response.status);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function login(email: string, password: string): Promise<SessionUser> {
  const result = await send<{ user: SessionUser }>('POST', '/auth/login', { email, password });
  return result.user;
}

/** The signed-in user; throws AuthError with status 401 when nobody is signed in. */
export function getMe(): Promise<SessionUser> {
  return send<SessionUser>('GET', '/me');
}

/** First sign-in: new + confirm only. From Settings: the current password too. */
export function changePassword(
  newPassword: string,
  confirmPassword: string,
  currentPassword?: string,
): Promise<SessionUser> {
  return send<SessionUser>('POST', '/auth/change-password', {
    new_password: newPassword,
    confirm_password: confirmPassword,
    ...(currentPassword !== undefined && { current_password: currentPassword }),
  });
}

export async function logout(): Promise<void> {
  try {
    await send<void>('POST', '/auth/logout');
  } catch {
    // Already signed out or the server is down: the dashboard goes to /login either way.
  }
}

/** Admin only. Creates the account and emails a temporary password. */
export function inviteStaff(name: string, email: string, role: Role): Promise<unknown> {
  return send('POST', '/users', { name, email, role });
}

/** Where to go after signing in. */
export function homePath(user: SessionUser): string {
  if (user.must_change_password) return '/set-password';
  return user.role === 'admin' ? '/admin/home' : '/staff/home';
}