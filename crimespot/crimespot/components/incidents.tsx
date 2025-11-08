type Incident = {
    id: string;
    lat: number;
    lng: number;
    description: string;
    datetime: number;
    neighborhood: string;
};

import { AnimatePresence, motion } from "motion/react";

function getCategoryColor(description: string): { color: string; label: string } {
	const d = (description || "").toUpperCase();
	const violent = ["ASSAULT", "HOMICIDE", "ROBBERY", "RAPE", "SHOOT", "WEAPON"];
	const property = ["BURGLARY", "LARCENY", "THEFT", "AUTO", "MOTOR VEHICLE", "ARSON", "VANDALISM"];
	if (violent.some(k => d.includes(k))) return { color: "#ef4444", label: "Violent" }; // red
	if (property.some(k => d.includes(k))) return { color: "#f59e0b", label: "Property" }; // orange
	return { color: "#3b82f6", label: "Other" }; // blue
}

function timeAgo(ms: number): string {
	if (!Number.isFinite(ms) || ms <= 0) return "Unknown time";
	const diff = Date.now() - ms;
	if (diff < 0) return new Date(ms).toLocaleString();
	const minutes = Math.floor(diff / 60000);
	if (minutes < 1) return "just now";
	if (minutes < 60) return `${minutes}m ago`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours}h ago`;
	const days = Math.floor(hours / 24);
	return `${days}d ago`;
}

export default function Incidents({incidents}: {incidents: Incident[]}) {
	return(
		<div className="flex flex-col gap-4">
			<div className="flex items-center justify-between mt-8">
				<h2 className="text-lg font-semibold">Incidents</h2>
				<span className="text-xs text-zinc-500 dark:text-zinc-400">{incidents.length} total</span>
			</div>
			<div className="flex flex-col gap-3">
				<AnimatePresence initial={false}>
					{incidents.map((incident: Incident, idx: number) => {
						const category = getCategoryColor(incident.description);
						const when = timeAgo(incident.datetime);
						return (
							<motion.div
								key={incident.id}
								initial={{ opacity: 0, y: 8, scale: 0.98 }}
								animate={{ opacity: 1, y: 0, scale: 1 }}
								exit={{ opacity: 0, y: -8, scale: 0.98 }}
								transition={{ duration: 0.25, delay: Math.min(idx * 0.02, 0.2) }}
								whileHover={{ scale: 1.01 }}
								className="relative overflow-hidden rounded-md border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
								style={{
									boxShadow: `0 1px 0 0 ${category.color}22 inset`,
								}}
							>
								<div className="absolute left-0 top-0 h-full w-1" style={{ backgroundColor: category.color }} />
								<div className="p-3">
									<div className="flex items-start justify-between gap-3">
										<div className="flex items-start gap-2">
											<span
												className="mt-1 inline-block h-2.5 w-2.5 rounded-full"
												style={{ backgroundColor: category.color }}
											/>
											<h3 className="font-medium leading-5">{incident.description || "No description"}</h3>
										</div>
										<span className="rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-600 dark:text-zinc-300"
											style={{ borderColor: `${category.color}55` }}
										>
											{category.label}
										</span>
									</div>
									<div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
										<span className="rounded-full bg-zinc-100 px-2 py-1 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">{when}</span>
										<span className="rounded-full bg-zinc-100 px-2 py-1 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">{incident.neighborhood || "Unknown"}</span>
										<span className="rounded-full bg-zinc-100 px-2 py-1 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
											{isFinite(incident.lat) && isFinite(incident.lng)
												? `${incident.lat.toFixed(4)}, ${incident.lng.toFixed(4)}`
												: "Coords N/A"}
										</span>
									</div>
								</div>
							</motion.div>
						);
					})}
				</AnimatePresence>
				{!incidents?.length && (
					<motion.p
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						className="text-sm text-zinc-600 dark:text-zinc-400"
					>
						No incidents found.
					</motion.p>
				)}
			</div>
		</div>
	)
}