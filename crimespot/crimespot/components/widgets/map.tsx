"use client";
import {useEffect, useRef} from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const token = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN as string | undefined;
mapboxgl.accessToken = token ?? "";

type Incident = {
	 id: string;
	 lat: number;
	 lng: number;
	 description: string;
	 datetime: number;
	 neighborhood: string;
};

export default function Map({location, incidents = []}: {location: {lng: number, lat: number}, incidents?: Incident[]}) {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const marker = useRef<mapboxgl.Marker | null>(null);
    const incidentMarkersRef = useRef<mapboxgl.Marker[]>([]);
    // Example POIs can be added here later if needed

    const getColorByDescription = (desc: string): string => {
        const d = (desc || "").toUpperCase();
        const violent = ["ASSAULT", "HOMICIDE", "ROBBERY", "RAPE", "SHOOT", "WEAPON"];
        const property = ["BURGLARY", "LARCENY", "THEFT", "AUTO", "MOTOR VEHICLE", "ARSON", "VANDALISM"];
        if (violent.some(k => d.includes(k))) return "#ef4444"; // red
        if (property.some(k => d.includes(k))) return "#f59e0b"; // orange
        return "#3b82f6"; // blue
    };
    useEffect(() => {
        if (!map.current) {
            if (!token) {
                console.error("Missing NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN. Map cannot load tiles.");
            }
            map.current = new mapboxgl.Map({
                container: mapContainerRef.current as HTMLElement,
                style: "mapbox://styles/mapbox/streets-v12",
                center: [-76.613, 39.305],
                zoom: 12,
            });
            map.current.on("load", () => {
                map.current?.resize();
            });
        }

    }, []);

    useEffect(() => {
        if (location && map.current) {
            map.current.flyTo({center:[location.lng, location.lat], zoom: 14})

            if (marker.current) marker.current.remove();

            marker.current = new mapboxgl.Marker({ color: "orange" })
            .setLngLat([location.lng, location.lat])
            .addTo(map.current!);
        }
    }, [location]);

    // Render incident markers
    useEffect(() => {
        if (!map.current) return;
        // Clear previous markers
        incidentMarkersRef.current.forEach(m => m.remove());
        incidentMarkersRef.current = [];

        incidents
            .filter(i => Number.isFinite(i.lat) && Number.isFinite(i.lng))
            .forEach((i) => {
                const color = getColorByDescription(i.description);
                const popupHtml = `
                    <div style="font-size:12px;line-height:1.2;">
                        <div><strong>${i.description || "Incident"}</strong></div>
                        <div>${i.neighborhood || "Unknown neighborhood"}</div>
                        <div>${(i.datetime && Number.isFinite(i.datetime)) ? new Date(i.datetime).toLocaleString() : "Unknown time"}</div>
                    </div>
                `;
                const popup = new mapboxgl.Popup({ closeButton: true }).setHTML(popupHtml);
                const m = new mapboxgl.Marker({ color }).setLngLat([i.lng, i.lat]).setPopup(popup).addTo(map.current!);
                incidentMarkersRef.current.push(m);
            });
    }, [incidents]);

    return (
        <div className="relative w-[45vw] h-[60vh] rounded-md">
            <div ref={mapContainerRef} className="w-full h-full rounded-md" />
            <div className="absolute left-2 top-2 rounded-md bg-white/90 p-2 text-xs shadow-sm border border-zinc-200 dark:bg-zinc-900 dark:border-zinc-800">
                <div className="flex items-center gap-2"><span className="inline-block h-2 w-2 rounded-full" style={{backgroundColor:"#ef4444"}}></span> Violent</div>
                <div className="flex items-center gap-2"><span className="inline-block h-2 w-2 rounded-full" style={{backgroundColor:"#f59e0b"}}></span> Property</div>
                <div className="flex items-center gap-2"><span className="inline-block h-2 w-2 rounded-full" style={{backgroundColor:"#3b82f6"}}></span> Other</div>
            </div>
        </div>
    )
}