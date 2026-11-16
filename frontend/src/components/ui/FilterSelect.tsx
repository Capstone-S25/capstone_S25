import { ChevronDown } from 'lucide-react';

// Figma: Filter/Severity, Filter/Location, Filter/Date
// White pill with muted label + medium value + chevron. A transparent native
// <select> sits on top so keyboard and screen-reader behaviour stays native.

export interface FilterOption {
  value: string;
  label: string;
}

interface FilterSelectProps {
  label: string;
  value: string;
  options: FilterOption[];
  onChange: (value: string) => void;
  className?: string;
}

export function FilterSelect({ label, value, options, onChange, className = '' }: FilterSelectProps) {
  const current = options.find((o) => o.value === value)?.label ?? value;
  return (
    <label
      className={`relative inline-flex shrink-0 cursor-pointer items-center gap-[6px] rounded-[6px] border border-line bg-white py-[7px] pr-[8px] pl-[12px] text-[12px] whitespace-nowrap transition-colors hover:border-faint focus-within:border-brand ${className}`}
    >
      <span className="text-muted">{label}</span>
      <span className="font-medium text-ink">{current}</span>
      <ChevronDown size={14} className="text-ink-2" aria-hidden />
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="absolute inset-0 cursor-pointer opacity-0"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}
