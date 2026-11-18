import type { Incident } from '../../types';
import { SeverityBadge } from '../ui/Badge';
import { ClipId, ClipThumb } from '../ui/Primitives';

/** Compact incident summary shown at the top of review modals. */
export function IncidentPreview({ incident }: { incident: Incident }) {
  return (
    <div className="flex items-center gap-[12px] rounded-[8px] border border-line bg-head p-[10px]">
      <ClipThumb seed={0} src={incident.thumbnailUrl} className="h-[42px] w-[60px] shrink-0 rounded-[4px]" iconSize={14} />
      <div className="min-w-0 flex-1">
        <p className="m-0 flex items-center gap-[6px] text-[13px] font-medium text-ink">
          <ClipId id={incident.id} variant="chip" />
          <span className="truncate">{incident.type} · {incident.location}</span>
        </p>
        <p className="m-0 mt-[2px] truncate text-[12px] text-muted">{incident.description}</p>
      </div>
      <SeverityBadge severity={incident.severity} />
    </div>
  );
}
