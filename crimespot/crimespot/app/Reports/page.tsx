"use client";
import { useMemo, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import Header from "@/components/ui/header";
import PageTransition from "../../components/PageTransition";
import { useReport } from "@/components/providers/report-provider";

type UserReport = {
	id: string;
	type: string;
	location: string;
	time: string;
	createdAt: number;
};

function readReports(): UserReport[] {
	if (typeof window === "undefined") return [];
	try {
		const raw = localStorage.getItem("scrolink_reports");
		return raw ? (JSON.parse(raw) as UserReport[]) : [];
	} catch {
		return [];
	}
}

export default function ReportsPage() {
	const { report } = useReport();
	const [reports, setReports] = useState<UserReport[]>(() => readReports());

	function handleDelete(id: string) {
		const updated = reports.filter(r => r.id !== id);
		localStorage.setItem("scrolink_reports", JSON.stringify(updated));
		setReports(updated);
	}

	function handleClear() {
		localStorage.removeItem("scrolink_reports");
		setReports([]);
	}

	const sorted = useMemo(
		() => [...reports].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0)),
		[reports]
	);

	return (
		<div className="p-8 bg-zinc-50 dark:bg-black min-h-screen flex flex-col">
			<PageTransition>
				<main className="flex flex-col gap-4 items-center rounded-xl bg-white border border-zinc-200 dark:border-zinc-800 dark:bg-black w-full flex-1 overflow-visible">
					<Header/>
					<div className="w-full max-w-3xl px-4 pb-6">
						<div className="mb-4">
							<h1 className="text-2xl font-semibold mb-2">Latest Analysis</h1>
							{report ? (
								<div className="rounded-md border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
									<p className="text-sm text-zinc-600 dark:text-zinc-300">
										Query: <span className="font-medium">{report.query}</span>
									</p>
									<p className="text-xs text-zinc-500 dark:text-zinc-400 mb-2">
										Generated: {new Date(report.generatedAt).toLocaleString()}
									</p>
									<p className="text-sm mb-2">{report.mapSummary}</p>
									{report.modelAnalysis && (
										<p className="text-sm mb-2">{report.modelAnalysis}</p>
									)}
									{report.hotspotAnalysis && (
										<p className="text-sm">
											Hotspot:{" "}
											{report.hotspotAnalysis.is_hotspot
												? "Potential hotspot detected"
												: "No hotspot detected"}{" "}
											(risk {report.hotspotAnalysis.risk_score != null ? Math.round(report.hotspotAnalysis.risk_score * 100) + "%" : "n/a"})
										</p>
									)}
									{report.shelters?.length > 0 && (
										<p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
											Shelters: {report.shelters.slice(0, 3).map((s) => s.name).join(", ")}
										</p>
									)}
									{report.universities?.length > 0 && (
										<p className="text-xs text-zinc-500 dark:text-zinc-400">
											Universities: {report.universities.slice(0, 3).map((u) => u.name).join(", ")}
										</p>
									)}
								</div>
							) : (
								<p className="text-sm text-zinc-600 dark:text-zinc-400">
									Run a search on the Dashboard to generate a data-backed report.
								</p>
							)}
						</div>
						<div className="mb-4 flex items-center justify-between">
							<h1 className="text-2xl font-semibold">User Reports</h1>
							<Button variant="outline" onClick={handleClear} disabled={!sorted.length}>Clear all</Button>
						</div>
						<ScrollArea className="h-[75vh] rounded-md border border-zinc-200 p-3 dark:border-zinc-800">
							<div className="flex flex-col gap-3">
								{sorted.map((r) => {
									const when = r.time
										? new Date(r.time).toLocaleString()
										: new Date(r.createdAt).toLocaleString();
									return (
										<div key={r.id} className="rounded-md border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
											<div className="flex items-start justify-between">
												<div>
													<h3 className="font-medium">{r.type}</h3>
													<p className="text-sm text-zinc-600 dark:text-zinc-400">{r.location}</p>
													<p className="text-xs text-zinc-500 dark:text-zinc-400">{when}</p>
												</div>
												<Button variant="outline" onClick={() => handleDelete(r.id)}>Delete</Button>
											</div>
										</div>
									);
								})}
								{!sorted.length && (
									<p className="text-sm text-zinc-600 dark:text-zinc-400">No reports yet. Use “Report Incident” on the Dashboard to add one.</p>
								)}
							</div>
						</ScrollArea>
					</div>
				</main>
			</PageTransition>
		</div>
	);
}

