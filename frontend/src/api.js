/* API helpers — all calls go through Vite proxy → FastAPI backend */

const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

/** Smart input — backend auto-detects analysis vs Q&A */
export function submitQuery(input) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({ input }),
  });
}

/** Start a full analysis job */
export function startAnalysis(ticker, ragLevel = "intermediate") {
  return request("/analysis", {
    method: "POST",
    body: JSON.stringify({ ticker, rag_level: ragLevel }),
  });
}

/** Poll analysis result */
export function getAnalysis(jobId) {
  return request(`/analysis/${jobId}`);
}

/** SSE stream for analysis progress */
export function streamAnalysis(jobId, onEvent) {
  const source = new EventSource(`${BASE}/analysis/${jobId}/stream`);

  source.addEventListener("phase", (e) => {
    onEvent({ type: "phase", data: JSON.parse(e.data) });
  });

  source.addEventListener("agent_completed", (e) => {
    onEvent({ type: "agent_completed", data: JSON.parse(e.data) });
  });

  source.addEventListener("done", (e) => {
    onEvent({ type: "done", data: JSON.parse(e.data) });
    source.close();
  });

  source.addEventListener("error", (e) => {
    // SSE "error" may just be the connection closing
    if (source.readyState === EventSource.CLOSED) return;
    onEvent({ type: "error", data: { message: "Stream connection lost" } });
    source.close();
  });

  return source; // caller can close early if needed
}

/** Get indexed companies */
export function getCompanies() {
  return request("/companies");
}

/** Direct Q&A */
export function askQuestion(ticker, question) {
  return request("/qa", {
    method: "POST",
    body: JSON.stringify({ ticker, question }),
  });
}

/** SEC download */
export function downloadSEC(ticker) {
  return request("/sec/download", {
    method: "POST",
    body: JSON.stringify({ ticker }),
  });
}

/** SEC index */
export function indexSEC(ticker) {
  return request("/sec/index", {
    method: "POST",
    body: JSON.stringify({ ticker }),
  });
}

/** SEC status */
export function getSECStatus(ticker) {
  return request(`/sec/status/${ticker}`);
}
