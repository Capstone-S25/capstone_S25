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
 * running the pages keep showing the demo set.
 */
export function useIncidents(refreshMs = 5000): IncidentFeed {
  const [feed, setFeed] = useState<IncidentFeed>({ incidents: MOCK_INCIDENTS, source: 'demo' });

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const incidents = await fetchIncidents(controller.signal);
        setFeed({ incidents, source: 'live' });
      } catch {
        if (controller.signal.aborted) return;
        setFeed((previous) => (previous.source === 'live' ? previous : { incidents: MOCK_INCIDENTS, source: 'demo' }));
      }
      if (!controller.signal.aborted) timer = setTimeout(load, refreshMs);
    };
    load();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [refreshMs]);

  return feed;
}
