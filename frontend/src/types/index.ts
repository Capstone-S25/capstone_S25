export type UserRole = 'staff' | 'admin';

export type Severity = 'Critical' | 'Moderate' | 'Minor';

export type IncidentType = 'Smoking' | 'Littering' | 'Loitering' | 'Unattended item';

export type IncidentOutcome = 'Accepted' | 'Dismissed' | 'Escalated' | 'Pending';

export interface Incident {
  id: string;
  camId: string;
  location: string;
  type: IncidentType;
  severity: Severity;
  dateTime: string;
  outcome: IncidentOutcome;
  confidence: number;
  description: string;
  assignedTo?: string;
  remark?: string;
  thumbnailUrl?: string;
  duration?: string;
}

export interface Camera {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline';
  rules: string[];
  lastSeen?: string;
  thumbnailUrl?: string;
}

export interface StaffMember {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  initials: string;
}

export interface MonitoringRule {
  id: string;
  location: string;
  cameraIds: string[];
  detectType: IncidentType;
  activeFrom: string;
  activeTo: string;
  severity: Severity;
  minConfidence: number;
  enabled: boolean;
}

export interface WorkloadReport {
  weekLabel: string;
  footageHours: number;
  alertsRaised: number;
  escalatedCases: number;
  confirmedRealPct: number;
  accepted: number;
  dismissed: number;
  pending: number;
  dailyData: { day: string; real: number; false: number }[];
  byType: { type: IncidentType; count: number }[];
}
