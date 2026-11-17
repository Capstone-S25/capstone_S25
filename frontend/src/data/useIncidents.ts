import { useEffect, useState } from 'react';
import type { Incident } from '../types';
import { fetchIncidents } from './api';
import { MOCK_INCIDENTS } from './mock';

export interface IncidentFeed {
  incidents: Incident[];
  /** 'live' once the API has answered; 'demo' while loading or when it is down. */
  source: 'live' | 'demo';
}

/**
 * Incidents for the dashboard pages. Live data replaces the demo set as soon
 * as the API answers and refreshes every `refreshMs`; if the API is not
 * running the pages keep showing the demo set. `load` picks which incidents,
 * e.g. fetchReviewQueue for the review queue.
 */
export function useIncidents(
  refreshMs = 5000,
  load: (signal: AbortSignal) => Promise<Incident[]> = fetchIncidents,
): IncidentFeed {
  const [feed, setFeed] = useState<IncidentFeed>({ incidents: MOCK_INCIDENTS, source: 'demo' });

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = async () => {
      try {
        const incidents = await load(controller.signal);
        setFeed({ incidents, source: 'live' });
      } catch {
        if (controller.signal.aborted) return;
        setFeed((previous) => (previous.source === 'live' ? previous : { incidents: MOCK_INCIDENTS, source: 'demo' }));
      }
      if (!controller.signal.aborted) timer = setTimeout(refresh, refreshMs);
    };
    refresh();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [refreshMs, load]);

  return feed;
}
