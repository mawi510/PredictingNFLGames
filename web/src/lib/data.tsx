"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  getSlate,
  getStatus,
  getTrackRecord,
  type Slate,
  type Status,
  type TrackRecord,
} from "./api";

// One load per endpoint. `stale` means the API failed and the value came
// from the browser's last good copy.
export interface Loaded<T> {
  data: T | null;
  stale: boolean;
  failed: boolean;
  loading: boolean;
}

interface AppData {
  status: Loaded<Status>;
  slate: Loaded<Slate>;
  record: Loaded<TrackRecord>;
}

const initial = <T,>(): Loaded<T> => ({
  data: null,
  stale: false,
  failed: false,
  loading: true,
});

const DataContext = createContext<AppData>({
  status: initial(),
  slate: initial(),
  record: initial(),
});

function readCache<T>(key: string): T | null {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeCache(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be blocked; the page still renders from the live response.
  }
}

async function load<T>(key: string, fetcher: () => Promise<T>): Promise<Loaded<T>> {
  try {
    const data = await fetcher();
    writeCache(key, data);
    return { data, stale: false, failed: false, loading: false };
  } catch {
    const cached = readCache<T>(key);
    return { data: cached, stale: cached !== null, failed: true, loading: false };
  }
}

export function DataProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Loaded<Status>>(initial);
  const [slate, setSlate] = useState<Loaded<Slate>>(initial);
  const [record, setRecord] = useState<Loaded<TrackRecord>>(initial);

  useEffect(() => {
    load("pmp:status", getStatus).then(setStatus);
    load("pmp:slate", getSlate).then(setSlate);
    load("pmp:record", getTrackRecord).then(setRecord);
  }, []);

  return (
    <DataContext.Provider value={{ status, slate, record }}>
      {children}
    </DataContext.Provider>
  );
}

export const useAppData = () => useContext(DataContext);
