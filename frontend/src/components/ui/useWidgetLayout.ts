import { useEffect, useState } from 'react';

export interface WidgetConfig<T extends string> {
  id: T;
  visible: boolean;
}

export type WidgetMeta<T extends string> = Record<T, { label: string; hint: string }>;

function readLayout<T extends string>(key: string, defaults: WidgetConfig<T>[], meta: WidgetMeta<T>): WidgetConfig<T>[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return defaults;
    const saved = JSON.parse(raw) as WidgetConfig<T>[];
    const known = saved.filter((w) => w.id in meta);
    // append any widgets added since the layout was saved
    const missing = defaults.filter((d) => !known.some((w) => w.id === d.id));
    return [...known, ...missing];
  } catch {
    return defaults;
  }
}

/** Widget order + visibility, remembered per browser. */
export function useWidgetLayout<T extends string>(key: string, defaults: WidgetConfig<T>[], meta: WidgetMeta<T>) {
  const [layout, setLayout] = useState<WidgetConfig<T>[]>(() => readLayout(key, defaults, meta));
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(layout)); } catch { /* storage unavailable */ }
  }, [key, layout]);
  return [layout, setLayout] as const;
}
