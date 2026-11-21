// Live incidents from the dashboard API (data-pipeline, /api/v1/incidents).
// In development Vite proxies /api to the API server; see vite.config.ts.
import type { Incident, IncidentOutcome, IncidentType, Severity } from '../types';
import { MOCK_CAMERAS } from './mock';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

const TYPE_LABEL: Record<string, IncidentType> = {
  smoking: 'Smoking',
  littering: 'Littering',
  loitering: 'Loitering',
  unattended_item: 'Unattended item',
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
  camera_id: string;
  event_type: string;
  severity: string;
  first_anomaly_at: string;
  last_anomaly_at: string;
  review_outcome?: string | null;
  card?: ApiIncidentCard | null;
}

interface ApiIncidentCard {
  evidence_start_at: string;
  review_notes?: string | null;
  screening?: ApiVerdict | null;
  confirmation?: ApiVerdict | null;
  review_count?: number;
  confirmation_count?: number;
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

function cardFromDetail(detail: ApiDetail): ApiIncidentCard {
  const history = detail.vlm_history ?? (detail.vlm_verdict ? [detail.vlm_verdict] : []);
  // The screening classifier supplies the score (confidence set); Gemma
  // supplies a verdict and description (confidence null). An incident can be
  // reviewed by Gemma several times as it continues, with screening results
  // in between, so look at all of Gemma's reviews rather than the last entry.
  const scored = [...history].reverse().find((v) => v.confidence !== null);
  const reviews = history.filter((v) => v.confidence === null);
  const confirming = [...reviews].reverse().find((v) => v.assessment === 'confirmed');
  const shown = confirming ?? reviews[reviews.length - 1];
  return {
    evidence_start_at: detail.evidence_start_at,
    review_notes: detail.review?.notes,
    screening: scored,
    confirmation: shown,
    review_count: reviews.length,
    confirmation_count: reviews.filter((v) => v.assessment === 'confirmed').length,
  };
}

function describe(card: ApiIncidentCard): { description: string; screeningReason?: string; confidence: number; verdict: string } {
  const { screening: scored, confirmation: shown, review_count = 0, confirmation_count = 0 } = card;
  const times = review_count === 1 ? '1 review' : `${review_count} reviews`;
  let verdict = 'No model verdict yet';
  if (shown && confirmation_count > 0) {
    verdict = `Gemma confirmed this in ${confirmation_count} of ${times} (${shown.model_version})`;
  } else if (shown) {
    const text = shown.assessment === 'rejected' ? 'Gemma did not confirm this' : 'Gemma could not tell';
    verdict = `${text} (${times}, ${shown.model_version})`;
  } else if (scored) {
    verdict = 'Flagged by the screening classifier; Gemma has not reviewed it yet';
  }
  return {
    description: shown?.description ?? scored?.description ?? 'No description yet.',
    screeningReason: scored?.description ?? undefined,
    confidence: scored?.confidence ?? 0,
    verdict,
  };
}

/** A failed API call; status is the HTTP status code. */
export class ApiError extends Error {
  readonly status: number;

  constructor(path: string, status: number) {
    super(`${path} returned ${status}`);
    this.status = status;
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) throw new ApiError(path, response.status);
  return (await response.json()) as T;
}

function toIncident(item: ApiListItem, card: ApiIncidentCard, location?: string | null): Incident {
  const camId = cameraKey(item.camera_id);
  const camera = MOCK_CAMERAS.find((c) => c.id === camId);
  const { description, screeningReason, confidence, verdict } = describe(card);
  return {
    id: `inc-${String(item.id).padStart(3, '0')}`,
    camId,
    location: location ?? camera?.location ?? item.camera_id,
    type: TYPE_LABEL[item.event_type] ?? capitalize<IncidentType>(item.event_type.replace(/_/g, ' ')),
    severity: capitalize<Severity>(item.severity),
    dateTime: item.first_anomaly_at,
    outcome: item.review_outcome ? capitalize<IncidentOutcome>(item.review_outcome) : 'Pending',
    confidence,
    description,
    screeningReason,
    remark: card.review_notes ?? undefined,
    duration: duration(card.evidence_start_at, item.last_anomaly_at),
    thumbnailUrl: `${BASE}/api/v1/incidents/${item.id}/thumbnail`,
    clipUrl: `${BASE}/api/v1/incidents/${item.id}/clip`,
    verdict,
    live: true,
    apiId: item.id,
  };
}

async function withCards(items: ApiListItem[], signal?: AbortSignal): Promise<Incident[]> {
  return Promise.all(items.map(async (item) => {
    if (item.card) return toIncident(item, item.card);
    // Older API deployments ignore include_card.
    const detail = await getJson<ApiDetail>(`/api/v1/incidents/${item.id}`, signal);
    return toIncident(
      { ...item, review_outcome: detail.review?.outcome },
      cardFromDetail(detail),
      detail.location.name ?? detail.camera.name,
    );
  }));
}

/** Newest incidents with their model verdicts. Throws when the API is down. */
export async function fetchIncidents(signal?: AbortSignal, limit = 50): Promise<Incident[]> {
  const page = await getJson<{ items: ApiListItem[] }>(`/api/v1/incidents?page_size=${limit}&include_card=true`, signal);
  return withCards(page.items, signal);
}

/**
 * Incidents waiting for review (API List #21), most urgent first: critical,
 * then moderate, then minor, longest waiting first within each.
 */
export async function fetchReviewQueue(signal?: AbortSignal, limit = 50): Promise<Incident[]> {
  const page = await getJson<{ items: ApiListItem[] }>(`/api/v1/reviews/queue?page_size=${limit}&include_card=true`, signal);
  return withCards(page.items, signal);
}

/** A flagged YOLO box: seconds from the start of the clip, corners as CSS percentages. */
export interface FlaggedBox {
  t: number;
  label: string;
  box: { left: string; top: string; width: string; height: string };
}

interface ApiDetection {
  offset_ms: number;
  class_label: string;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

/** Flagged boxes to draw over the clip (#18), in time order. */
export async function fetchFlaggedBoxes(apiId: number, signal?: AbortSignal): Promise<FlaggedBox[]> {
  const detections = await getJson<ApiDetection[]>(`/api/v1/incidents/${apiId}/detections`, signal);
  const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
  return detections.map((d) => ({
    t: d.offset_ms / 1000,
    label: d.class_label,
    box: { left: pct(d.bbox.x1), top: pct(d.bbox.y1), width: pct(d.bbox.x2 - d.bbox.x1), height: pct(d.bbox.y2 - d.bbox.y1) },
  }));
}

export type ReviewOutcome = 'accepted' | 'dismissed' | 'escalated';

/** Save the review for an incident (#22). Throws ApiError; 409 means already reviewed. */
export async function submitReview(apiId: number, outcome: ReviewOutcome, notes: string): Promise<void> {
  const path = `/api/v1/incidents/${apiId}/review`;
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ outcome, notes: notes || null }),
  });
  if (!response.ok) throw new ApiError(path, response.status);
}
