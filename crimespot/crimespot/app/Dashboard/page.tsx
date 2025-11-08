"use client";
import PageTransition from "../../components/PageTransition";
import Incidents from "@/components/incidents";
import { Button } from "@/components/ui/button";
import Header from "@/components/ui/header";
import { Item, ItemActions, ItemContent, ItemDescription, ItemTitle } from "@/components/ui/item";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import ChatBox from "@/components/widgets/chat-box";
import Map from "@/components/widgets/map";
import Search from "@/components/widgets/search";
import { useState } from "react";
import { useReport } from "@/components/providers/report-provider";

type Incident = {
  id: string;
  lat: number;
  lng: number;
  description: string;
  datetime: number;
  neighborhood: string;
};

type ArcGisFeature = {
  attributes?: { [key: string]: unknown };
  properties?: { [key: string]: unknown };
};

const newId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random()}`;
};

export default function Dashboard() {
    const { setReport } = useReport();
  const [location, setLocation] = useState<{ lng: number; lat: number }>({ lng: 0, lat: 0 });
  const [danger, setDanger] = useState<string>("");
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [summary, setSummary] = useState<string>("");
  const [modelAnalysis, setModelAnalysis] = useState<string>("");
  const [hotspotResult, setHotspotResult] = useState<{ enabled: boolean; is_hotspot: boolean | null; risk_score: number | null } | null>(null);
  const [shelters, setShelters] = useState<any[]>([]);
  const [universities, setUniversities] = useState<any[]>([]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState<string>("");

  async function handleSearch(query: string) {
    if (!query) return;
    try {
      const q = encodeURIComponent(query);
      const geocodeRes = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/${q}.json?access_token=${process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN}&limit=1`
      );
      if (!geocodeRes.ok) throw new Error(`Geocode failed: ${geocodeRes.status}`);
      const geoData = await geocodeRes.json();
      if (!geoData.features?.length) return;
      const [lng, lat] = geoData.features[0].center;
      setLocation({ lng, lat });

      const radius = 500;
      const end = Date.now();
      const start = end - 24 * 60 * 60 * 1000;
      const arcgisRes = await fetch(
        `https://services1.arcgis.com/UWYHeuuJISiGmgXx/arcgis/rest/services/NIBRS_GroupA_Crime_Data/FeatureServer/0/query?` +
          `geometry=${lng},${lat}&geometryType=esriGeometryPoint&spatialRel=esriSpatialRelIntersects&distance=${radius}` +
          `&units=esriSRUnit_Meter&inSR=4326&outFields=*&where=1%3D1&returnGeometry=false&f=json&resultRecordCount=50` +
          `&time=${start},${end}`
      );
      if (!arcgisRes.ok) throw new Error(`Crime query failed: ${arcgisRes.status}`);
      const arcgisData = await arcgisRes.json();

      const parsed: Incident[] = (arcgisData.features ?? []).map((f: ArcGisFeature, i: number) => {
        const a = (f.attributes ?? f.properties ?? {}) as Record<string, unknown>;
        let latNum = Number(a["Latitude"] as string | number | undefined);
        let lngNum = Number(a["Longitude"] as string | number | undefined);
        if (!Number.isFinite(latNum) || !Number.isFinite(lngNum)) {
          const geoLoc = a["GeoLocation"] as string | undefined;
          const m = typeof geoLoc === "string" ? geoLoc.match(/\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)/) : null;
          if (m) {
            latNum = Number(m[1]);
            lngNum = Number(m[2]);
          }
        }
        return {
          id: String((a["CCNumber"] ?? a["RowID"] ?? a["OBJECTID"] ?? `row-${i}`) as string | number),
          lat: latNum,
          lng: lngNum,
          description: (a["Description"] as string | undefined) ?? "",
          datetime: Number((a["CrimeDateTime"] as number | string | undefined) ?? 0),
          neighborhood: (a["Neighborhood"] as string | undefined) ?? (a["Neighbourhood"] as string | undefined) ?? "",
        };
      });
      setIncidents(parsed);

      const last24Count = parsed.length;
      const dangerLevel = last24Count > 10 ? "High" : last24Count > 5 ? "Medium" : "Low";
      setDanger(dangerLevel);

      const categorize = (d: string) => {
        const u = (d || "").toUpperCase();
        const violent = ["ASSAULT", "HOMICIDE", "ROBBERY", "RAPE", "SHOOT", "WEAPON"];
        const property = ["BURGLARY", "LARCENY", "THEFT", "AUTO", "MOTOR VEHICLE", "ARSON", "VANDALISM"];
        if (violent.some((k) => u.includes(k))) return "violent";
        if (property.some((k) => u.includes(k))) return "property";
        return "other";
      };
      const counts = parsed.reduce((acc, i) => {
        const c = categorize(i.description);
        acc[c] = (acc[c] ?? 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      const byNeighborhood = parsed.reduce((acc, i) => {
        const key = i.neighborhood || "Unknown";
        acc[key] = (acc[key] ?? 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      const topNeighborhoods = Object.entries(byNeighborhood)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([n, c]) => `${n} (${c})`)
        .join(", ");
      const s = `Last 24h near search area: ${last24Count} incidents. Breakdown — violent: ${counts.violent ?? 0}, property: ${counts.property ?? 0}, other: ${counts.other ?? 0}. Top neighborhoods: ${topNeighborhoods || "n/a"}.`;
      setSummary(s);

      setBackendLoading(true);
      setBackendError("");
      setModelAnalysis("");
      setHotspotResult(null);
      setShelters([]);
      setUniversities([]);

            try {
                const backendRes = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query }),
                });
        if (!backendRes.ok) {
          throw new Error(`Backend error: ${backendRes.status}`);
        }
        const backendData = await backendRes.json();
                const sheltersData = backendData.extraSummaries?.homeless_shelters || [];
                const universitiesData = backendData.extraSummaries?.universities || [];
                const analysisText = backendData.response || "";
                const hotspot = backendData.hotspotAnalysis || null;
                setModelAnalysis(analysisText);
                setHotspotResult(hotspot);
                setShelters(sheltersData);
                setUniversities(universitiesData);
                setReport({
                    id: newId(),
                    query,
                    generatedAt: Date.now(),
                    mapSummary: s,
                    modelAnalysis: analysisText,
                    hotspotAnalysis: hotspot,
                    shelters: sheltersData,
                    universities: universitiesData,
                });
            } catch (err) {
                console.error(err);
                setBackendError("Unable to load Gemini risk analysis.");
      } finally {
        setBackendLoading(false);
      }
    } catch (err) {
      console.error(err);
      setBackendError("Search failed. Please try again.");
    }
  }

  const shelterHighlights = shelters.slice(0, 3).map((s) => `${s.name} (${s.beds_total ?? "n/a"} beds)`).join(", ");
  const universityHighlights = universities.slice(0, 3).map((u) => u.name).join(", ");
  const riskPercent = hotspotResult?.risk_score != null ? `${Math.round(hotspotResult.risk_score * 100)}%` : "n/a";

  return (
    <div className="p-8 bg-zinc-50 dark:bg-black min-h-screen flex flex-col">
      <PageTransition>
        <main className="flex flex-col gap-4 items-center rounded-xl bg-white border border-zinc-200 dark:border-zinc-800 dark:bg-black w-full flex-1 overflow-visible">
          <Header />
            <div className="flex gap-5 w-full justify-center">
                <div className="w-1/2 space-y-2">
                    <ChatBox incidents={incidents} summary={modelAnalysis || summary} />
                    <Search onSearch={handleSearch} />
                    {backendError && <p className="text-sm text-red-500">{backendError}</p>}
                </div>
                <div className="w-1/2 space-y-2" id="map-section">
                    <Map location={location} incidents={incidents} />
                    {danger && (
                <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md mt-2">
                  <ItemContent>
                    <ItemTitle>Danger Level (Map data)</ItemTitle>
                    <ItemDescription>{danger}</ItemDescription>
                    <p>{incidents.length} incidents reported in the last 24h</p>
                    <ItemActions>
                      <Sheet>
                        <SheetTrigger asChild>
                          <Button>View Incidents</Button>
                        </SheetTrigger>
                        <SheetContent className="w-[400px]">
                          <ScrollArea className="h-[95vh] pr-2">
                            <Incidents incidents={incidents} />
                          </ScrollArea>
                        </SheetContent>
                      </Sheet>
                    </ItemActions>
                  </ItemContent>
                </Item>
              )}
              {backendLoading && (
                <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md">
                  <ItemContent>
                    <ItemTitle>Gemini Risk Analysis</ItemTitle>
                    <ItemDescription>Loading insight from the crime dataset…</ItemDescription>
                  </ItemContent>
                </Item>
              )}
              {modelAnalysis && !backendLoading && (
                <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md">
                  <ItemContent>
                    <ItemTitle>Gemini Risk Analysis</ItemTitle>
                    <ItemDescription>{modelAnalysis}</ItemDescription>
                  </ItemContent>
                </Item>
              )}
              {hotspotResult?.enabled && (
                <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md">
                  <ItemContent>
                    <ItemTitle>Hotspot Prediction</ItemTitle>
                    <ItemDescription>
                      {hotspotResult.is_hotspot ? "Potential hotspot detected" : "No hotspot detected"}
                    </ItemDescription>
                    <p>Risk score: {riskPercent}</p>
                  </ItemContent>
                </Item>
              )}
              {(shelters.length > 0 || universities.length > 0) && (
                <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md">
                  <ItemContent>
                    <ItemTitle>Contextual Layers</ItemTitle>
                    <ItemDescription>
                      {shelters.length > 0 && <p className="mb-1 text-sm">Shelters: {shelterHighlights || shelters.length}</p>}
                      {universities.length > 0 && (
                        <p className="text-sm">Universities: {universityHighlights || universities.length}</p>
                      )}
                    </ItemDescription>
                  </ItemContent>
                </Item>
              )}
            </div>
          </div>
        </main>
      </PageTransition>
    </div>
  );
}
