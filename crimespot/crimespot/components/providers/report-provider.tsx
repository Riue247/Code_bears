"use client";
import { createContext, useContext, useState } from "react";

export type AnalysisReport = {
  id: string;
  query: string;
  generatedAt: number;
  mapSummary: string;
  modelAnalysis: string;
  hotspotAnalysis: {
    enabled: boolean;
    is_hotspot: boolean | null;
    risk_score: number | null;
  } | null;
  shelters: any[];
  universities: any[];
};

type ReportContextValue = {
  report: AnalysisReport | null;
  setReport: (report: AnalysisReport) => void;
};

const ReportContext = createContext<ReportContextValue | undefined>(undefined);

export function ReportProvider({ children }: { children: React.ReactNode }) {
  const [report, setReport] = useState<AnalysisReport | null>(null);

  return (
    <ReportContext.Provider value={{ report, setReport }}>
      {children}
    </ReportContext.Provider>
  );
}

export function useReport() {
  const ctx = useContext(ReportContext);
  if (!ctx) {
    throw new Error("useReport must be used within a ReportProvider");
  }
  return ctx;
}
