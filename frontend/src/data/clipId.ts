// Clip ID numbers shown across the UI. Every incident id ends in its number
// ("inc-012", the API incident 12), so the clip ID is that number: "#012".

/** The clip's number, e.g. "inc-012" -> 12, or null if the id has no number. */
export function clipNumber(id: string): number | null {
  const digits = id.match(/(\d+)\s*$/);
  return digits ? Number(digits[1]) : null;
}

/** The clip ID as shown, e.g. "#012". Falls back to the raw id. */
export function clipLabel(id: string): string {
  const n = clipNumber(id);
  return n === null ? id : `#${String(n).padStart(3, '0')}`;
}

/**
 * True when a search query names this clip's ID: "#12", "12", "012",
 * "clip 12" and "inc-012" all match clip 12.
 */
export function matchesClipId(id: string, query: string): boolean {
  const q = query.trim().toLowerCase().replace(/^(clip|inc)[\s-]*/, '').replace(/^#/, '');
  if (!/^\d+$/.test(q)) return false;
  return clipNumber(id) === Number(q);
}
