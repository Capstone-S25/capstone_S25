import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUp, ArrowRight } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { FilterSelect } from '../../components/ui/FilterSelect';
import { StatusPill } from '../../components/ui/Badge';
import { BarMeter, FieldLabel, PageHeader, SectionTitle } from '../../components/ui/Primitives';
import { RULE_TYPES } from '../../data/rules';

// ── Data ──────────────────────────────────────────────────────────────────────

type Floor = 'Level 1–3' | 'Basement 2';

interface Place {
  name: string;
  count: number;
  floor: Floor;
  // footprint on the floor plate, in grid units
  x: number; y: number; w: number; d: number;
  summary: string;
  types: string[];
  times: [number, number, number, number]; // morning, afternoon, evening, night
}

const PLACES: Place[] = [
  { name: 'Carpark B2', count: 22, floor: 'Basement 2', x: 0, y: 0, w: 5, d: 4, summary: 'Mostly smoking near the staircase doors late at night. Up slightly from last week.', types: ['Smoking', 'Loitering'], times: [0, 1, 6, 15] },
  { name: 'Lift Lobby 1', count: 16, floor: 'Level 1–3', x: 5, y: 0.5, w: 2.4, d: 3, summary: 'Littering outside the lifts, mostly in the evening rush. 35% more than last week.', types: ['Littering'], times: [2, 3, 9, 2] },
  { name: 'Corridor 2', count: 11, floor: 'Level 1–3', x: 7.6, y: 0.5, w: 2.4, d: 3, summary: 'People waiting by the fire exit for lifts. Steady week on week.', types: ['Loitering'], times: [1, 4, 5, 1] },
  { name: 'Main entrance', count: 7, floor: 'Level 1–3', x: 0, y: 0, w: 3, d: 4, summary: 'People lingering by the gantry and smoking at the drop-off point.', types: ['Loitering', 'Smoking'], times: [3, 2, 2, 0] },
  { name: 'Meeting Room 1', count: 5, floor: 'Basement 2', x: 7.6, y: 0.5, w: 2.4, d: 3, summary: 'Cups and wrappers left behind after meetings end.', types: ['Littering'], times: [2, 3, 0, 0] },
  { name: 'Staircase B', count: 4, floor: 'Basement 2', x: 5.2, y: 0.5, w: 2.2, d: 3, summary: 'Smoking on the landing with the fire door propped open.', types: ['Smoking'], times: [0, 0, 1, 3] },
  { name: 'Pantry L3', count: 3, floor: 'Level 1–3', x: 3.2, y: 0.5, w: 1.6, d: 3, summary: 'Wrappers and cups left beside the bins after lunch.', types: ['Littering'], times: [0, 3, 0, 0] },
];

const TIME_LABELS = ['Morning · 6 AM–12 PM', 'Afternoon · 12–6 PM', 'Evening · 6–10 PM', 'Night · 10 PM–6 AM'];

const TYPE_OPTIONS = [
  { value: 'all', label: 'All types' },
  ...RULE_TYPES.map((t) => ({ value: t, label: t })),
];

const PERIOD_OPTIONS = [
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: '90d', label: 'Last 90 days' },
];

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HotspotsPage() {
  const navigate = useNavigate();
  const [type, setType] = useState('all');
  const [period, setPeriod] = useState('7d');
  const [selected, setSelected] = useState('Carpark B2');

  const places = useMemo(
    () => PLACES.filter((p) => type === 'all' || p.types.includes(type)).sort((a, b) => b.count - a.count),
    [type],
  );
  const max = Math.max(1, ...places.map((p) => p.count));
  const place = places.find((p) => p.name === selected) ?? places[0];
  const periodLabel = PERIOD_OPTIONS.find((o) => o.value === period)?.label.toLowerCase();
  const maxTime = place ? Math.max(1, ...place.times) : 1;

  return (
    <AppLayout role="staff" activePath="/staff/hotspots" variant="bare">
      <section className="thin-scroll flex min-w-0 flex-1 flex-col gap-[20px] overflow-y-auto px-[28px] pt-[20px] pb-[28px]">
        <PageHeader
          title="Hotspots"
          subtitle="Where and when incidents happen · Tower A"
          actions={
            <>
              <FilterSelect label="Type" value={type} onChange={setType} options={TYPE_OPTIONS} />
              <FilterSelect label="Period" value={period} onChange={setPeriod} options={PERIOD_OPTIONS} />
            </>
          }
        />

        {/* Summary stats */}
        <div className="flex w-full items-start border-b border-line-soft pb-[16px]">
          <Stat first label="Busiest place" value="Carpark B2" note="4 more than last week, mostly smoking" />
          <Stat label="Busiest time" value="Evenings, 6–10 PM" note="41% of incidents, up from 33%" />
          <Stat label="Biggest change" value="Lift Lobby 1" note="35% more littering than last week" />
        </div>

        {/* Site map */}
        <div className="flex flex-col gap-[12px]">
          <div className="flex items-center gap-[16px]">
            <p className="m-0 text-[14px] font-medium text-ink">Site map · Tower A</p>
            <div className="flex-1" />
            <Legend color="var(--color-brand)">10+ incidents</Legend>
            <Legend color="#9db5ea">5–9</Legend>
            <Legend color="#cdd4e1">Under 5</Legend>
          </div>
          <div className="grid grid-cols-1 gap-[16px] rounded-[8px] bg-head p-[24px] lg:grid-cols-2">
            {(['Level 1–3', 'Basement 2'] as Floor[]).map((floor) => (
              <IsoFloor
                key={floor}
                label={floor}
                places={places.filter((p) => p.floor === floor)}
                selected={place?.name}
                onSelect={setSelected}
              />
            ))}
          </div>
          <p className="m-0 text-[11px] text-muted">
            Incidents in the {periodLabel}. Block height and colour scale with the count. Select a place to see when they happen.
          </p>
        </div>

        {/* Selected place */}
        {place && (
          <div className="flex w-full flex-wrap gap-x-[48px] gap-y-[20px] border-t border-line-soft pt-[20px]">
            <div className="flex min-w-[260px] flex-1 flex-col gap-[6px]">
              <FieldLabel>Selected place</FieldLabel>
              <p className="m-0 text-[18px] font-medium text-ink">{place.name}</p>
              <p className="m-0 max-w-[300px] text-[13px] text-ink-2">{place.summary}</p>
              <div className="mt-[4px] flex gap-[6px]">
                {place.types.map((t) => <StatusPill key={t} tone="neutral">{t}</StatusPill>)}
              </div>
              <button
                type="button"
                onClick={() => navigate(`/staff/search?q=${encodeURIComponent(place.name)}`)}
                className="mt-[6px] flex w-fit cursor-pointer items-center gap-[4px] border-none bg-transparent p-0 text-[13px] font-medium text-brand hover:underline"
              >
                View clips <ArrowRight size={14} />
              </button>
            </div>
            <div className="flex min-w-[320px] flex-[1.4] flex-col gap-[10px]">
              <p className="m-0 text-[12px] font-medium text-ink">When it happens</p>
              {place.times.map((n, i) => (
                <div key={i} className="flex items-center gap-[12px]">
                  <span className="w-[140px] shrink-0 text-[11px] text-ink-2">{TIME_LABELS[i]}</span>
                  <BarMeter pct={(n / maxTime) * 100} className="flex-1" />
                  <span className="w-[20px] shrink-0 text-right text-[11px] font-medium text-ink">{n}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Incidents by place */}
      <aside className="thin-scroll flex w-[340px] shrink-0 flex-col gap-[4px] overflow-y-auto border-l border-line bg-head p-[20px]">
        <div className="pb-[10px]"><SectionTitle>Incidents by place</SectionTitle></div>
        {places.map((p) => {
          const active = p.name === place?.name;
          return (
            <button
              key={p.name}
              type="button"
              onClick={() => setSelected(p.name)}
              aria-pressed={active}
              className={`flex w-full cursor-pointer flex-col gap-[10px] rounded-[6px] border px-[12px] pt-[12px] pb-[14px] text-left transition-colors ${
                active ? 'border-brand bg-white' : 'border-transparent bg-transparent hover:bg-white'
              }`}
            >
              <span className="flex w-full items-center">
                <span className="flex-1 truncate text-[13px] text-ink">{p.name}</span>
                <span className="text-[13px] font-medium text-ink">{p.count}</span>
              </span>
              <BarMeter pct={(p.count / max) * 100} className="w-full" />
            </button>
          );
        })}
      </aside>
    </AppLayout>
  );
}

// ── Isometric floor ───────────────────────────────────────────────────────────

const S = 28; // px per grid unit
const COS = Math.cos(Math.PI / 6);
const SIN = Math.sin(Math.PI / 6);
const PLATE = { w: 10, d: 4 };

function iso(x: number, y: number, z: number): [number, number] {
  return [(x - y) * COS * S, (x + y) * SIN * S - z * S];
}
const pts = (list: [number, number][]) => list.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(' ');

function shade(count: number) {
  if (count >= 10) return { top: '#0041c0', left: '#00317f', right: '#00297a', text: '#ffffff' };
  if (count >= 5) return { top: '#9db5ea', left: '#7f9bd8', right: '#6d8bcc', text: '#121a2b' };
  return { top: '#cdd4e1', left: '#b3bccc', right: '#a4aec0', text: '#121a2b' };
}

function Block({ p, max, selected, onSelect }: { p: Place; max: number; selected: boolean; onSelect: () => void }) {
  const h = 0.35 + (p.count / max) * 1.9;
  const { x, y, w, d } = p;
  const c = shade(p.count);
  const top = pts([iso(x, y, h), iso(x + w, y, h), iso(x + w, y + d, h), iso(x, y + d, h)]);
  const left = pts([iso(x, y + d, 0), iso(x + w, y + d, 0), iso(x + w, y + d, h), iso(x, y + d, h)]);
  const right = pts([iso(x + w, y, 0), iso(x + w, y + d, 0), iso(x + w, y + d, h), iso(x + w, y, h)]);
  const stroke = selected ? '#121a2b' : 'rgba(255,255,255,0.6)';

  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${p.name}, ${p.count} incidents`}
      aria-pressed={selected}
      onClick={onSelect}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(); } }}
      className="cursor-pointer outline-none [&:focus-visible_polygon]:stroke-brand"
    >
      <polygon points={left} fill={c.left} stroke={stroke} strokeWidth={selected ? 1.5 : 0.75} />
      <polygon points={right} fill={c.right} stroke={stroke} strokeWidth={selected ? 1.5 : 0.75} />
      <polygon points={top} fill={c.top} stroke={stroke} strokeWidth={selected ? 1.5 : 0.75} />
    </g>
  );
}

function labelGeom(p: Place, max: number) {
  const h = 0.35 + (p.count / max) * 1.9;
  const [cx, cy] = iso(p.x + p.w / 2, p.y + p.d / 2, h);
  const w = p.name.length * 5.6 + 14;
  return { cx, cy, w, box: { x0: cx - w / 2, x1: cx + w / 2, y0: cy - 24, y1: cy + 16 } };
}

/** Decide which name pills fit: selected place first, then busiest; overlapping ones show the count only. */
function placeLabels(places: Place[], max: number, selected?: string) {
  const order = [...places].sort((a, b) => (a.name === selected ? -1 : b.name === selected ? 1 : b.count - a.count));
  const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
  const show = new Set<string>();
  for (const p of order) {
    const { box } = labelGeom(p, max);
    if (!placed.some((b) => box.x0 < b.x1 && box.x1 > b.x0 && box.y0 < b.y1 && box.y1 > b.y0)) {
      placed.push(box);
      show.add(p.name);
    }
  }
  return show;
}

function BlockLabel({ p, max, showName, onSelect }: { p: Place; max: number; showName: boolean; onSelect: () => void }) {
  const { cx, cy, w } = labelGeom(p, max);
  return (
    <g onClick={onSelect} className="cursor-pointer" aria-hidden>
      <title>{`${p.name} · ${p.count}`}</title>
      <circle cx={cx} cy={cy - 14} r={10} fill="#ffffff" stroke="#121a2b" strokeWidth={1} />
      <text x={cx} y={cy - 10.5} textAnchor="middle" fontSize={10} fontWeight={600} fill="#121a2b">{p.count}</text>
      {showName && (
        <>
          <rect x={cx - w / 2} y={cy + 1} width={w} height={15} rx={3} fill="#ffffff" stroke="#dde3ee" />
          <text x={cx} y={cy + 11.5} textAnchor="middle" fontSize={9.5} fontWeight={500} fill="#121a2b">{p.name}</text>
        </>
      )}
    </g>
  );
}

function IsoFloor({ label, places, selected, onSelect }: { label: string; places: Place[]; selected?: string; onSelect: (n: string) => void }) {
  const max = 22;
  const slab = 0.25;
  const plateTop = pts([iso(-0.4, -0.4, 0), iso(PLATE.w + 0.4, -0.4, 0), iso(PLATE.w + 0.4, PLATE.d + 0.4, 0), iso(-0.4, PLATE.d + 0.4, 0)]);
  const plateLeft = pts([iso(-0.4, PLATE.d + 0.4, -slab), iso(PLATE.w + 0.4, PLATE.d + 0.4, -slab), iso(PLATE.w + 0.4, PLATE.d + 0.4, 0), iso(-0.4, PLATE.d + 0.4, 0)]);
  const plateRight = pts([iso(PLATE.w + 0.4, -0.4, -slab), iso(PLATE.w + 0.4, PLATE.d + 0.4, -slab), iso(PLATE.w + 0.4, PLATE.d + 0.4, 0), iso(PLATE.w + 0.4, -0.4, 0)]);
  // back walls
  const wallBack = pts([iso(-0.4, -0.4, 0), iso(PLATE.w + 0.4, -0.4, 0), iso(PLATE.w + 0.4, -0.4, 0.9), iso(-0.4, -0.4, 0.9)]);
  const wallSide = pts([iso(-0.4, -0.4, 0), iso(-0.4, PLATE.d + 0.4, 0), iso(-0.4, PLATE.d + 0.4, 0.9), iso(-0.4, -0.4, 0.9)]);

  const [minX] = iso(-0.4, PLATE.d + 0.4, 0);
  const [maxX] = iso(PLATE.w + 0.4, -0.4, 0);
  const [, minY] = iso(-0.4, -0.4, 3.2);
  const [, maxY] = iso(PLATE.w + 0.4, PLATE.d + 0.4, -slab);
  const pad = 12;
  const ordered = [...places].sort((a, b) => a.x + a.y - (b.x + b.y));
  const named = placeLabels(places, max, selected);

  return (
    <div className="flex flex-col gap-[8px]">
      <svg viewBox={`${minX - pad} ${minY - pad} ${maxX - minX + pad * 2} ${maxY - minY + pad * 2}`} className="h-auto w-full" role="group" aria-label={`Site map, ${label}`}>
        <polygon points={wallBack} fill="#e3e8f1" stroke="#cdd4e1" strokeWidth={0.75} />
        <polygon points={wallSide} fill="#dde3ee" stroke="#cdd4e1" strokeWidth={0.75} />
        <polygon points={plateLeft} fill="#cdd4e1" />
        <polygon points={plateRight} fill="#b3bccc" />
        <polygon points={plateTop} fill="#ffffff" stroke="#dde3ee" strokeWidth={0.75} />
        {ordered.map((p) => (
          <Block key={p.name} p={p} max={max} selected={p.name === selected} onSelect={() => onSelect(p.name)} />
        ))}
        {ordered.map((p) => (
          <BlockLabel key={p.name} p={p} max={max} showName={named.has(p.name)} onSelect={() => onSelect(p.name)} />
        ))}
      </svg>
      <span className="text-[11px] text-ink-2">{label}</span>
    </div>
  );
}

// ── Small parts ───────────────────────────────────────────────────────────────

function Stat({ label, value, note, first = false }: { label: string; value: string; note: string; first?: boolean }) {
  return (
    <div className={`flex min-w-0 flex-1 flex-col gap-[2px] pr-[16px] ${first ? '' : 'border-l border-line-soft pl-[20px]'}`}>
      <span className="text-[12px] text-muted">{label}</span>
      <span className="truncate text-[18px] font-medium text-ink">{value}</span>
      <span className="flex min-w-0 items-center gap-[3px]">
        <ArrowUp size={10} strokeWidth={3} className="shrink-0 text-danger" />
        <span className="truncate text-[11px] text-muted">{note}</span>
      </span>
    </div>
  );
}

function Legend({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span className="flex items-center gap-[6px] text-[11px] text-muted">
      <span className="size-[9px] rounded-[2px]" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}
