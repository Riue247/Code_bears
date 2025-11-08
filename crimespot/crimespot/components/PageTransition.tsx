"use client";
import { AnimatePresence, motion } from "motion/react";

export default function PageTransition({ children }: { children: React.ReactNode }) {
	return (
		<AnimatePresence mode="wait">
			<motion.div
				key="page"
				initial={{ opacity: 0, y: 12 }}
				animate={{ opacity: 1, y: 0 }}
				exit={{ opacity: 0, y: -12 }}
				transition={{ duration: 0.35, ease: "easeOut" }}
				className="overflow-visible w-full"
			>
				{children}
			</motion.div>
		</AnimatePresence>
	);
}

