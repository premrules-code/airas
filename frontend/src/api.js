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

/** SSE stream for analysis progress with auto-reconnect */
export function streamAnalysis(jobId, onEvent) {
  let retries = 0;
  const MAX_RETRIES = 3;
  let done = false;
  let source = null;

  async function connect() {
    // Pre-check: verify job still exists before opening SSE
    // (EventSource can't distinguish 404 from network error)
    try {
      const check = await fetch(`${BASE}/analysis/${jobId}`);
      if (check.status === 404) {
        onEvent({ type: "error", data: { message: "Analysis job expired. Please start a new query." } });
        return;
      }
    } catch {
      // Network error — still try SSE, polling fallback will handle it
    }

    source = new EventSource(`${BASE}/analysis/${jobId}/stream`);

    source.addEventListener("phase", (e) => {
      retries = 0;
      onEvent({ type: "phase", data: JSON.parse(e.data) });
    });

    source.addEventListener("agent_completed", (e) => {
      retries = 0;
      onEvent({ type: "agent_completed", data: JSON.parse(e.data) });
    });

    source.addEventListener("done", (e) => {
      done = true;
      onEvent({ type: "done", data: JSON.parse(e.data) });
      source.close();
    });

    source.addEventListener("error", () => {
      if (done) return;
      source.close();

      retries++;
      if (retries <= MAX_RETRIES) {
        setTimeout(connect, 2000 * retries);
      } else {
        // Stop retrying — polling fallback is already running
        onEvent({ type: "error", data: { message: "Stream connection lost — using polling fallback" } });
      }
    });
  }

  connect();

  return {
    close() {
      done = true;
      if (source) source.close();
    },
  };
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
