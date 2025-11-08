"use client";
import Link from "next/link";
import { Button } from "./button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "./dialog";
import { useState } from "react";
import CreateReport from "../create-report";
import { LayoutDashboard, MapPin, FileText, AlertTriangle } from "lucide-react";

const routes = [
    { label: "Dashboard", href: "/Dashboard", icon: LayoutDashboard },
    { label: "Map", href: "/Dashboard#map-section", icon: MapPin },
    { label: "Reports", href: "/Reports", icon: FileText },
];

export default function Header() {
    const [open, setOpen] = useState(false);
    return (
        <header className="flex justify-between items-center p-4 w-full">
            <h1 className="text-2xl font-medium">Scrolink</h1>
            <div className="flex gap-3">
                {routes.map(({ label, href, icon: Icon }) => (
                    <Link key={label} href={href} className="flex items-center gap-1.5 text-zinc-800 hover:text-blue-600 dark:text-zinc-100">
                        {Icon && <Icon className="size-4" />}
                        <span>{label}</span>
                    </Link>
                ))}
            </div>
            <div className="flex gap-2">
                <Dialog open={open} onOpenChange={setOpen}>
                    <DialogTrigger asChild>
                        <Button className="flex items-center gap-2">
                            <AlertTriangle className="size-4" />
                            <span>Report Incident</span>
                        </Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Report an Incident</DialogTitle>
                        </DialogHeader>
                        <CreateReport onSaved={() => setOpen(false)} />
                    </DialogContent>
                </Dialog>
            </div>
        </header>
    )
}
