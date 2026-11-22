import { useState, useEffect, useRef } from 'react';
import { Clock, MoreVertical, Check, Plus, ChevronDown, Copy, Trash2 } from 'lucide-react';
import { AppLayout } from '../../components/layout/AppLayout';
import { Button } from '../../components/ui/Button';
import AddRuleModal from '../../components/modals/AddRuleModal';
import { TONE } from '../../components/ui/Badge';
import { SEVERITY_TONE } from '../../components/ui/tones';
import { Toggle } from '../../components/ui/Toggle';
import { PageHeader } from '../../components/ui/Primitives';
import { RULE_TYPES, SEVERITIES, saveRules, useRules } from '../../data/rules';
import type { MonitoringRule, Severity, IncidentType } from '../../types';

const camLabels = (ids: string[]) => ids.map((id) => 'Cam ' + id.replace('cam-', '').padStart(2, '0')).join(', ');

const CELL_INPUT =
  'rounded-[6px] border border-line bg-white px-[8px] py-[6px] text-[12px] text-ink outline-none transition-colors hover:border-faint focus:border-brand disabled:cursor-not-allowed disabled:bg-head disabled:text-faint';

// Figma table header style: 11px medium muted, 1px line under header, line-soft between rows
const COLS = 'grid grid-cols-[minmax(160px,1.2fr)_150px_200px_150px_170px] items-center gap-[16px]';

type Menu = { id: string; top: number; right: number } | null;
type Toast = { text: string; restore?: { rule: MonitoringRule; index: number } } | null;

const newId = () => `rule-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;

export default function MonitoringRulesPage() {
  // Saved rules are shared with the staff screens; edits stay in this draft until "Save changes".
  const saved = useRules();
  const [draft, setDraft] = useState<MonitoringRule[]>(saved);
  const [showAddRule, setShowAddRule] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [menu, setMenu] = useState<Menu>(null);
  const [toast, setToast] = useState<Toast>(null);

  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);

  function update<K extends keyof MonitoringRule>(id: string, key: K, value: MonitoringRule[K]) {
    setDraft((prev) => prev.map((r) => (r.id === id ? { ...r, [key]: value } : r)));
  }

  function handleSave() {
    saveRules(draft);
    setJustSaved(true);
    setToast({ text: 'Rules saved · staff alerts now follow these rules' });
  }

  function openMenu(id: string, button: HTMLElement) {
    if (menu?.id === id) { setMenu(null); return; }
    const r = button.getBoundingClientRect();
    setMenu({ id, top: r.bottom + 4, right: window.innerWidth - r.right });
  }

  function duplicate(id: string) {
    const index = draft.findIndex((r) => r.id === id);
    if (index < 0) return;
    const copy: MonitoringRule = { ...draft[index], id: newId() };
    setDraft((prev) => [...prev.slice(0, index + 1), copy, ...prev.slice(index + 1)]);
    setToast({ text: `Rule duplicated · ${copy.location} · ${copy.detectType}` });
  }

  function remove(id: string) {
    const index = draft.findIndex((r) => r.id === id);
    if (index < 0) return;
    const rule = draft[index];
    setDraft((prev) => prev.filter((r) => r.id !== id));
    setToast({ text: `Rule deleted · ${rule.location} · ${rule.detectType}`, restore: { rule, index } });
  }

  function undoDelete() {
    const restore = toast?.restore;
    if (!restore) return;
    setDraft((prev) => [...prev.slice(0, restore.index), restore.rule, ...prev.slice(restore.index)]);
    setToast(null);
  }

  function addRule(rule: Omit<MonitoringRule, 'id'>) {
    setDraft((prev) => [...prev, { ...rule, id: newId() }]);
    setShowAddRule(false);
    setToast({ text: `Rule added · ${rule.location} · ${rule.detectType} · save to apply` });
  }

  useEffect(() => {
    if (!justSaved) return;
    const t = setTimeout(() => setJustSaved(false), 2000);
    return () => clearTimeout(t);
  }, [justSaved]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 5000);
    return () => clearTimeout(t);
  }, [toast]);

  const active = draft.filter((r) => r.enabled).length;

  return (
    <AppLayout role="admin" activePath="/admin/rules">
      <PageHeader
        title="Monitoring rules"
        subtitle={`${active} of ${draft.length} rules active · the AI flags matching behaviour during each rule's hours.`}
        actions={
          <>
            {dirty && <Button variant="secondary" onClick={() => setDraft(saved)}>Discard</Button>}
            <Button variant="secondary" icon={<Plus size={16} />} onClick={() => setShowAddRule(true)}>Add rule</Button>
            <Button variant="primary" onClick={handleSave} disabled={!dirty && !justSaved} icon={justSaved && !dirty ? <Check size={16} /> : undefined}>
              {justSaved && !dirty ? 'Saved' : 'Save changes'}
            </Button>
          </>
        }
      />

      <div className="w-full overflow-x-auto" role="table" aria-label="Monitoring rules">
        <div className="min-w-[900px]">
          <div role="row" className={`${COLS} border-b border-line pb-[8px] text-[11px] font-medium text-muted`}>
            <span role="columnheader">Location</span>
            <span role="columnheader">Detect</span>
            <span role="columnheader">Active hours</span>
            <span role="columnheader">Severity</span>
            <span role="columnheader">Status</span>
          </div>

          {draft.length === 0 && (
            <p className="m-0 py-[24px] text-center text-[13px] text-muted">No rules yet. Add one so the AI knows what to flag.</p>
          )}

          {draft.map((rule) => {
            const off = !rule.enabled;
            return (
              <div key={rule.id} role="row" className={`${COLS} border-b border-line-soft py-[10px] transition-colors ${off ? 'bg-head/60' : 'hover:bg-head/60'}`}>
                <span role="cell" className="flex min-w-0 flex-col gap-[2px]">
                  <span className={`truncate text-[13px] font-medium ${off ? 'text-faint' : 'text-ink'}`}>{rule.location}</span>
                  <span className={`truncate text-[12px] ${off ? 'text-faint' : 'text-muted'}`}>{camLabels(rule.cameraIds)}</span>
                </span>

                <span role="cell">
                  <select
                    aria-label={`Detect type for ${rule.location}`}
                    value={rule.detectType}
                    onChange={(e) => update(rule.id, 'detectType', e.target.value as IncidentType)}
                    disabled={off}
                    className={`${CELL_INPUT} w-full`}
                  >
                    {/* A rule saved before the list was narrowed keeps its type until it is changed. */}
                    {!(RULE_TYPES as readonly string[]).includes(rule.detectType) && <option value={rule.detectType} disabled>{rule.detectType}</option>}
                    {RULE_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </span>

                <span role="cell" className={`inline-flex w-fit items-center gap-[6px] rounded-[6px] border px-[8px] py-[6px] ${off ? 'border-line bg-head' : 'border-line bg-white focus-within:border-brand'}`}>
                  <Clock size={13} className={off ? 'text-faint' : 'text-muted'} />
                  <TimeField label={`Active from, ${rule.location}`} value={rule.activeFrom} onChange={(v) => update(rule.id, 'activeFrom', v)} disabled={off} />
                  <span className="text-[12px] text-faint">–</span>
                  <TimeField label={`Active until, ${rule.location}`} value={rule.activeTo} onChange={(v) => update(rule.id, 'activeTo', v)} disabled={off} />
                </span>

                <span role="cell">
                  <SeveritySelect
                    label={`Severity for ${rule.location}`}
                    value={rule.severity}
                    onChange={(v) => update(rule.id, 'severity', v)}
                    disabled={off}
                  />
                </span>

                <span role="cell" className="flex items-center gap-[10px]">
                  <Toggle checked={rule.enabled} onChange={(v) => update(rule.id, 'enabled', v)} label={`Enable rule for ${rule.location}`} />
                  <span className={`w-[52px] text-[12px] ${off ? 'text-faint' : 'text-ink-2'}`}>{rule.enabled ? 'Enabled' : 'Disabled'}</span>
                  <button
                    type="button"
                    aria-label={`More options for ${rule.location}`}
                    aria-haspopup="menu"
                    aria-expanded={menu?.id === rule.id}
                    data-rule-menu-button
                    onClick={(e) => openMenu(rule.id, e.currentTarget)}
                    className={`flex size-[28px] cursor-pointer items-center justify-center rounded-[6px] border-none text-muted hover:bg-head hover:text-ink ${menu?.id === rule.id ? 'bg-head text-ink' : 'bg-transparent'}`}
                  >
                    <MoreVertical size={16} />
                  </button>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <AddRuleModal isOpen={showAddRule} onClose={() => setShowAddRule(false)} onAdd={addRule} />

      {menu && (
        <RowMenu
          menu={menu}
          onClose={() => setMenu(null)}
          onDuplicate={() => { duplicate(menu.id); setMenu(null); }}
          onDelete={() => { remove(menu.id); setMenu(null); }}
        />
      )}

      {toast && (
        <div role="status" className="v2-pop-in fixed bottom-[24px] left-1/2 z-[400] flex -translate-x-1/2 items-center gap-[16px] rounded-[8px] bg-dark py-[10px] pr-[10px] pl-[16px] text-[13px] text-white shadow-[0_12px_32px_rgba(18,26,43,0.3)]">
          <span>{toast.text}</span>
          {toast.restore && (
            <button type="button" onClick={undoDelete} className="cursor-pointer rounded-[4px] border-none bg-white/15 px-[10px] py-[4px] text-[12px] font-medium text-white hover:bg-white/25">
              Undo
            </button>
          )}
        </div>
      )}
    </AppLayout>
  );
}

/**
 * A time input whose resting text is drawn with every digit in a slot as wide as
 * "0", so "11:59 PM" and "08:00 PM" take the same width (the app's font has no
 * tabular digits). While focused, the native input shows for editing.
 */
function TimeField({ label, value, onChange, disabled }: { label: string; value: string; onChange: (v: string) => void; disabled: boolean }) {
  const [editing, setEditing] = useState(false);
  const [h, m] = value.split(':').map(Number);
  const text = new Date(2000, 0, 1, h || 0, m || 0).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  // "08:00" + " PM": the AM/PM part gets the width of the wider of the two, so it never shifts.
  const split = text.search(/[^\d:.\s]/);
  const clock = split < 0 ? text : text.slice(0, split);
  const suffix = split < 0 ? '' : text.slice(split);
  const dayParts = suffix ? [new Date(2000, 0, 1, 1).toLocaleTimeString([], { hour: '2-digit' }), new Date(2000, 0, 1, 13).toLocaleTimeString([], { hour: '2-digit' })].map((s) => s.replace(/[\d\s]/g, '')) : [];
  return (
    <span className="relative inline-flex">
      <input
        type="time"
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setEditing(true)}
        onBlur={() => setEditing(false)}
        disabled={disabled}
        className={`w-[64px] border-none bg-transparent p-0 text-[12px] outline-none [&::-webkit-calendar-picker-indicator]:hidden ${editing ? (disabled ? 'text-faint' : 'text-ink') : 'text-transparent'}`}
      />
      {!editing && (
        <span aria-hidden className={`pointer-events-none absolute inset-y-0 left-0 flex items-center text-[12px] whitespace-pre ${disabled ? 'text-faint' : 'text-ink'}`}>
          {[...clock].map((ch, i) => (/\d/.test(ch) ? <span key={i} className="inline-block w-[1ch] text-center">{ch}</span> : <span key={i}>{ch}</span>))}
          {suffix && (
            <span className="inline-grid">
              <span className="col-start-1 row-start-1">{suffix}</span>
              {dayParts.map((p) => <span key={p} className="invisible col-start-1 row-start-1">{p}</span>)}
            </span>
          )}
        </span>
      )}
    </span>
  );
}

/** One coloured severity chip that is itself the dropdown, like a spreadsheet's dropdown cell. */
function SeveritySelect({ label, value, onChange, disabled }: { label: string; value: Severity; onChange: (v: Severity) => void; disabled: boolean }) {
  const t = TONE[disabled ? 'neutral' : SEVERITY_TONE[value]];
  return (
    <span
      className={`relative inline-flex w-[112px] items-center rounded-full transition-[filter] focus-within:ring-2 focus-within:ring-brand/40 ${disabled ? 'opacity-70' : 'hover:brightness-95'}`}
      style={{ backgroundColor: t.bg, color: t.fg }}
    >
      <span className="pointer-events-none absolute left-[10px] size-[6px] rounded-full" style={{ backgroundColor: t.dot }} />
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value as Severity)}
        disabled={disabled}
        className="w-full cursor-pointer appearance-none border-none bg-transparent bg-none py-[5px] pr-[26px] pl-[22px] text-[12px] font-medium text-inherit outline-none disabled:cursor-not-allowed"
      >
        {SEVERITIES.map((o) => <option key={o} value={o} className="bg-white text-ink">{o}</option>)}
      </select>
      <ChevronDown size={14} className="pointer-events-none absolute right-[8px]" aria-hidden />
    </span>
  );
}

/**
 * The row's "more options" menu. Fixed to the viewport so the table's scroll area
 * can't clip it; closes on an outside click, Escape, scrolling or resizing.
 */
function RowMenu({ menu, onClose, onDuplicate, onDelete }: { menu: NonNullable<Menu>; onClose: () => void; onDuplicate: () => void; onDelete: () => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus();
  }, []);

  useEffect(() => {
    function onDown(e: MouseEvent) {
      const target = e.target as Element;
      if (ref.current?.contains(target) || target.closest('[data-rule-menu-button]')) return;
      onClose();
    }
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose(); }
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', onClose);
    window.addEventListener('scroll', onClose, true);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onClose);
      window.removeEventListener('scroll', onClose, true);
    };
  }, [onClose]);

  const item = 'flex w-full cursor-pointer items-center gap-[10px] rounded-[6px] border-none bg-transparent px-[10px] py-[8px] text-left text-[13px]';
  return (
    <div
      ref={ref}
      role="menu"
      style={{ top: menu.top, right: menu.right }}
      className="v2-pop-in fixed z-[1000] w-[168px] rounded-[10px] border border-line bg-white p-[6px] shadow-[0_12px_32px_rgba(18,26,43,0.16)]"
    >
      <button type="button" role="menuitem" onClick={onDuplicate} className={`${item} text-ink hover:bg-head focus-visible:bg-head focus-visible:outline-none`}>
        <Copy size={15} className="text-muted" />Duplicate
      </button>
      <button type="button" role="menuitem" onClick={onDelete} className={`${item} text-danger hover:bg-danger-soft focus-visible:bg-danger-soft focus-visible:outline-none`}>
        <Trash2 size={15} />Delete
      </button>
    </div>
  );
}
