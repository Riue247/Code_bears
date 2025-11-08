const express = require("express");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const pipelinePath = path.join(__dirname, "rag_pipeline.py");
const chatPath = path.join(__dirname, "gemini_chat.py");

app.use(express.json());

function runPython(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn("python", [scriptPath, ...args]);
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
      } catch (err) {
        reject(new Error(`Invalid JSON from Python script: ${err.message}`));
      }
    });
  });
}

app.post("/query", async (req, res) => {
  const userQuery = req.body.query;
  if (!userQuery) {
    return res.status(400).json({ error: "Missing query" });
  }

  try {
    const result = await runPython(pipelinePath, ["--query", userQuery, "--skip-save"]);
    return res.json({
      prompt: result.prompt,
      response: result.response,
      incidents: result.incidents,
      hotspotAnalysis: result.hotspot_analysis,
      extraSummaries: result.extra_summaries,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

app.post("/chat", async (req, res) => {
  const message = req.body.message;
  if (!message) {
    return res.status(400).json({ error: "Missing message" });
  }

  try {
    const result = await runPython(chatPath, ["--message", message]);
    return res.json(result);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
