import { useNavigate } from 'react-router-dom';
import { Pencil, Clapperboard } from 'lucide-react';
import type { Camera } from '../../types';
import { MOCK_INCIDENTS, MOCK_RULES } from '../../data/mock';
import { SeverityBadge, OutcomeDot, StatusPill } from '../ui/Badge';
import { Button } from '../ui/Button';
import { Drawer } from '../ui/Drawer';
import { ClipId, ClipThumb } from '../ui/Primitives';

interface CameraDetailDrawerProps {
  cam: Camera | null;
  onClose: () => void;
}

function formatLastSeen(iso?: string): string {
  if (!iso) return 'Now';
  const d = new Date(iso);
  return `${d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}, ${d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}`;
}

const formatTime = (iso: string) => new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });

export default function CameraDetailDrawer({ cam, onClose }: CameraDetailDrawerProps) {
  const navigate = useNavigate();
  if (!cam) return null;

  const online = cam.status === 'online';
  const camIncidents = MOCK_INCIDENTS
    .filter((i) => i.camId === cam.id)
    .sort((a, b) => new Date(b.dateTime).getTime() - new Date(a.dateTime).getTime())
    .slice(0, 5);
  const camRules = MOCK_RULES.filter((r) => cam.rules.includes(r.id));

  return (
    <Drawer
      isOpen
      onClose={onClose}
      title={`${cam.name} · ${cam.location}`}
      titleAdornment={<StatusPill tone={online ? 'green' : 'neutral'}>{online ? 'Live' : 'Offline'}</StatusPill>}
      footer={
        <>
          <Button
            variant="secondary"
            className="flex-1"
            icon={<Clapperboard size={16} strokeWidth={1.75} />}
            onClick={() => { onClose(); navigate(`/staff/search?q=${encodeURIComponent(cam.location)}`); }}
          >
            View clips
          </Button>
          <Button variant="primary" className="flex-1" icon={<Pencil size={16} strokeWidth={1.75} />}>Edit camera</Button>
        </>
      }
    >
      <div className="flex flex-col gap-[20px]">
        {/* Feed */}
        <ClipThumb
          seed={cam.id.length}
          offline={!online}
          offlineLabel={!online ? `No signal since ${formatLastSeen(cam.lastSeen)}` : undefined}
          className="aspect-[16/9] w-full rounded-[6px]"
          iconSize={32}
        >
          {online && (
            <span className="absolute top-[10px] left-[10px] flex items-center gap-[6px] rounded-[4px] bg-[rgba(27,35,52,0.75)] px-[8px] py-[4px] text-[11px] font-medium text-white">
              <span className="size-[6px] rounded-full bg-[#4ade80]" /> {cam.name.toUpperCase()} · LIVE
            </span>
          )}
        </ClipThumb>

        {/* Facts */}
        <div className="flex border-y border-line-soft py-[12px]">
          <Fact label="Status" first>
            <span className={online ? 'text-success' : 'text-ink-2'}>{online ? 'Online' : 'Offline'}</span>
          </Fact>
          <Fact label="Last seen">{online ? 'Now' : formatLastSeen(cam.lastSeen)}</Fact>
          <Fact label="Incidents today">{camIncidents.length}</Fact>
        </div>

        {/* Rules */}
        <div className="flex flex-col gap-[8px]">
          <p className="m-0 text-[13px] font-medium text-ink">Looks for</p>
          {camRules.length === 0 ? (
            <p className="m-0 text-[12px] text-muted">No rules assigned.</p>
          ) : (
            camRules.map((rule) => (
              <div key={rule.id} className="flex items-center gap-[10px] rounded-[6px] border border-line px-[12px] py-[10px]">
                <div className="min-w-0 flex-1">
                  <p className={`m-0 text-[13px] font-medium ${rule.enabled ? 'text-ink' : 'text-faint'}`}>{rule.detectType}</p>
                  <p className="m-0 mt-[2px] text-[12px] text-muted">{rule.activeFrom} – {rule.activeTo} · {rule.severity}</p>
                </div>
                <StatusPill tone={rule.enabled ? 'green' : 'neutral'}>{rule.enabled ? 'Active' : 'Off'}</StatusPill>
              </div>
            ))
          )}
        </div>

        {/* Recent incidents */}
        <div className="flex flex-col">
          <p className="m-0 mb-[6px] text-[13px] font-medium text-ink">Recent incidents</p>
          {camIncidents.length === 0 ? (
            <p className="m-0 text-[12px] text-muted">No incidents recorded.</p>
          ) : (
            camIncidents.map((inc) => (
              <div key={inc.id} className="flex items-center gap-[10px] border-b border-line-soft py-[10px] last:border-b-0">
                <div className="min-w-0 flex-1">
                  <p className="m-0 truncate text-[13px] font-medium text-ink">{inc.type}</p>
                  <p className="m-0 mt-[2px] flex items-center gap-[6px] text-[12px] text-muted"><ClipId id={inc.id} />·<span>{formatTime(inc.dateTime)}</span></p>
                </div>
                <SeverityBadge severity={inc.severity} />
                <OutcomeDot outcome={inc.outcome} className="w-[84px]" />
              </div>
            ))
          )}
        </div>
      </div>
    </Drawer>
  );
}

function Fact({ label, children, first = false }: { label: string; children: React.ReactNode; first?: boolean }) {
  return (
    <div className={`flex min-w-0 flex-1 flex-col gap-[2px] ${first ? '' : 'border-l border-line-soft pl-[16px]'}`}>
      <span className="text-[11px] font-medium text-muted">{label}</span>
      <span className="truncate text-[13px] font-medium text-ink">{children}</span>
    </div>
  );
}
