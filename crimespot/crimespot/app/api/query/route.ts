import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

const backendDir = path.join(process.cwd(), "backend");
const pipelineScript = path.join(backendDir, "rag_pipeline.py");
const PYTHON_BIN = process.env.PYTHON_BIN || "python";

function runPython(args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [pipelineScript, ...args]);
    let stdout = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data) => {
      console.error(data.toString());
    });

    child.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(`Python script exited with code ${code}`));
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (err: any) {
        reject(new Error(`Invalid JSON from Python script: ${err.message}`));
      }
    });
  });
}

export async function POST(request: NextRequest) {
  const { query } = await request.json().catch(() => ({}));
  if (!query || typeof query !== "string") {
    return NextResponse.json({ error: "Missing query" }, { status: 400 });
  }

  try {
    const result = await runPython(["--query", query, "--skip-save"]);
    return NextResponse.json({
      prompt: result.prompt,
      response: result.response,
      incidents: result.incidents,
      hotspotAnalysis: result.hotspot_analysis,
      extraSummaries: result.extra_summaries,
    });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
