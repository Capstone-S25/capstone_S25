// Live incidents from the dashboard API (data-pipeline, /api/v1/incidents).
// In development Vite proxies /api to the API server; see vite.config.ts.
import type { Incident, IncidentOutcome, IncidentType, Severity } from '../types';
import { MOCK_CAMERAS } from './mock';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

const TYPE_LABEL: Record<string, IncidentType> = {
  smoking: 'Smoking',
  littering: 'Littering',
  loitering: 'Loitering',
  unattended_items: 'Unattended item',
  rough_sleeping: 'Rough sleeping',
};

interface ApiVerdict {
  assessment: 'confirmed' | 'rejected' | 'uncertain';
  confidence: number | null;
  description: string | null;
  model_version: string;
}

interface ApiListItem {
  id: number;
}

interface ApiDetail {
  id: number;
  camera: { id: string; name: string | null };
  location: { id: number | null; name: string | null };
  event_type: string;
  severity: string;
  first_anomaly_at: string;
  last_anomaly_at: string;
  evidence_start_at: string;
  evidence_end_at: string;
  vlm_verdict: ApiVerdict | null;
  vlm_history?: ApiVerdict[];
  review: { outcome: string; notes: string | null; reviewer_id: number } | null;
}

function capitalize<T extends string>(value: string): T {
  return (value.charAt(0).toUpperCase() + value.slice(1)) as T;
}

// "camera_01" -> "cam-01", the id format the camera pages use.
export function cameraKey(cameraId: string): string {
  const digits = cameraId.match(/(\d+)\s*$/);
  return digits ? `cam-${digits[1].padStart(2, '0')}` : cameraId;
}

function duration(startIso: string, endIso: string): string {
  const seconds = Math.max(0, Math.round((Date.parse(endIso) - Date.parse(startIso)) / 1000));
  return `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`;
}

function describe(detail: ApiDetail): { description: string; confidence: number; verdict: string } {
  const history = detail.vlm_history ?? (detail.vlm_verdict ? [detail.vlm_verdict] : []);
  const latest = history[history.length - 1];
  // The screening classifier supplies the score; Gemma supplies the wording.
  const scored = [...history].reverse().find((v) => v.confidence !== null);
  const reviewer = [...history].reverse().find((v) => v.confidence === null && v.description);
  const verdict = !latest
    ? 'No model verdict yet'
    : latest.confidence === null
      ? `Gemma ${latest.assessment} this (${latest.model_version})`
      : 'Flagged by the screening classifier; Gemma has not confirmed it';
  return {
    description: reviewer?.description ?? latest?.description ?? 'No description yet.',
    confidence: scored?.confidence ?? 0,
    verdict,
  };
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return (await response.json()) as T;
}

function toIncident(detail: ApiDetail): Incident {
  const camId = cameraKey(detail.camera.id);
  const camera = MOCK_CAMERAS.find((c) => c.id === camId);
  const { description, confidence, verdict } = describe(detail);
  return {
    id: `inc-${String(detail.id).padStart(3, '0')}`,
    camId,
    location: detail.location.name ?? camera?.location ?? detail.camera.name ?? detail.camera.id,
    type: TYPE_LABEL[detail.event_type] ?? capitalize<IncidentType>(detail.event_type.replace(/_/g, ' ')),
    severity: capitalize<Severity>(detail.severity),
    dateTime: detail.first_anomaly_at,
    outcome: detail.review ? capitalize<IncidentOutcome>(detail.review.outcome) : 'Pending',
    confidence,
    description,
    remark: detail.review?.notes ?? undefined,
    duration: duration(detail.evidence_start_at, detail.last_anomaly_at),
    thumbnailUrl: `${BASE}/api/v1/incidents/${detail.id}/thumbnail`,
    clipUrl: `${BASE}/api/v1/incidents/${detail.id}/clip`,
    verdict,
    live: true,
  };
}

/** Newest incidents with their model verdicts. Throws when the API is down. */
export async function fetchIncidents(signal?: AbortSignal, limit = 50): Promise<Incident[]> {
  const page = await getJson<{ items: ApiListItem[] }>(`/api/v1/incidents?page_size=${limit}`, signal);
  const details = await Promise.all(
    page.items.map((item) => getJson<ApiDetail>(`/api/v1/incidents/${item.id}`, signal)),
  );
  return details.map(toIncident);
}
