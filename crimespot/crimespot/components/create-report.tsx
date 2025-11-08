"use client";
import { useEffect, useMemo, useState } from "react";
import { Button } from "./ui/button";

type UserReport = {
	id: string;
	type: string;
	location: string;
	time: string; // ISO string from input[type=datetime-local]
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

function writeReports(list: UserReport[]) {
	if (typeof window === "undefined") return;
	localStorage.setItem("scrolink_reports", JSON.stringify(list));
}

export default function CreateReport({ onSaved }: { onSaved?: (r: UserReport) => void }) {
	const [type, setType] = useState("");
	const [location, setLocation] = useState("");
	const [time, setTime] = useState("");
	const [submitting, setSubmitting] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [success, setSuccess] = useState<string | null>(null);

	const isValid = useMemo(() => type.trim() && location.trim() && time.trim(), [type, location, time]);

	useEffect(() => {
		if (success) {
			const t = setTimeout(() => setSuccess(null), 2500);
			return () => clearTimeout(t);
		}
	}, [success]);

	async function handleSubmit(e: React.FormEvent) {
		e.preventDefault();
		setError(null);
		setSuccess(null);
		if (!isValid) {
			setError("Please fill in all fields.");
			return;
		}
		setSubmitting(true);
		try {
			const report: UserReport = {
				id: `r_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
				type: type.trim(),
				location: location.trim(),
				time,
				createdAt: Date.now(),
			};
			const current = readReports();
			current.unshift(report); // newest first
			writeReports(current);
			setSuccess("Report submitted.");
			setType("");
			setLocation("");
			setTime("");
			onSaved?.(report);
		} catch {
			setError("Failed to save report. Please try again.");
		} finally {
			setSubmitting(false);
		}
	}

	return (
		<form onSubmit={handleSubmit} className="flex flex-col gap-3">
			<div className="flex flex-col gap-1">
				<label className="text-sm font-medium">Type of crime</label>
				<select
					value={type}
					onChange={(e) => setType(e.target.value)}
					className="w-full rounded-md border border-zinc-300 bg-white p-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-zinc-700 dark:bg-zinc-900"
				>
					<option value="" disabled>Select type…</option>
					<option value="Assault">Assault</option>
					<option value="Robbery">Robbery</option>
					<option value="Burglary">Burglary</option>
					<option value="Theft">Theft</option>
					<option value="Auto Theft">Auto Theft</option>
					<option value="Vandalism">Vandalism</option>
					<option value="Other">Other</option>
				</select>
			</div>
			<div className="flex flex-col gap-1">
				<label className="text-sm font-medium">Where it happened</label>
				<input
					type="text"
					placeholder="Address or description of location"
					value={location}
					onChange={(e) => setLocation(e.target.value)}
					className="w-full rounded-md border border-zinc-300 bg-white p-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-zinc-700 dark:bg-zinc-900"
				/>
			</div>
			<div className="flex flex-col gap-1">
				<label className="text-sm font-medium">Time it happened</label>
				<input
					type="datetime-local"
					value={time}
					onChange={(e) => setTime(e.target.value)}
					className="w-full rounded-md border border-zinc-300 bg-white p-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-zinc-700 dark:bg-zinc-900"
				/>
			</div>
			{error && <p className="text-sm text-red-600">{error}</p>}
			{success && <p className="text-sm text-green-600">{success}</p>}
			<div className="mt-1 flex justify-end">
				<Button type="submit" disabled={!isValid || submitting} className="bg-blue-600 text-white hover:bg-blue-600/90">
					{submitting ? "Submitting..." : "Submit Report"}
				</Button>
			</div>
		</form>
	);
}