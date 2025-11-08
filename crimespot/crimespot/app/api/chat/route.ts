import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

const backendDir = path.join(process.cwd(), "backend");
const chatScript = path.join(backendDir, "gemini_chat.py");
const PYTHON_BIN = process.env.PYTHON_BIN || "python";

function runPython(args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [chatScript, ...args]);
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
  const { prompt } = await request.json().catch(() => ({}));
  if (!prompt || typeof prompt !== "string") {
    return NextResponse.json({ error: "Missing prompt" }, { status: 400 });
  }

  try {
    const result = await runPython(["--message", prompt]);
    return NextResponse.json(result);
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
