import React, { useState, useRef, useCallback } from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import AgentGrid from "./components/AgentGrid";
import AgentReport from "./components/AgentReport";
import RecommendationPanel from "./components/RecommendationPanel";
import QAResult from "./components/QAResult";
import { submitQuery, streamAnalysis, getAnalysis } from "./api";

export default function App() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Analysis state
  const [mode, setMode] = useState(null); // null | "analysis" | "qa"
  const [ticker, setTicker] = useState(null);
  const [agents, setAgents] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  const [phase, setPhase] = useState(null); // downloading | indexing | routing | context_gathered | done
  const [phaseMessage, setPhaseMessage] = useState(null);

  // QA state
  const [qaAnswer, setQaAnswer] = useState(null);
  const [qaSources, setQaSources] = useState([]);

  const sseRef = useRef(null);
  const pollRef = useRef(null);

  const resetState = useCallback(() => {
    setMode(null);
    setAgents([]);
    setRecommendation(null);
    setPhase(null);
    setPhaseMessage(null);
    setQaAnswer(null);
    setQaSources([]);
    setError(null);
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  function startPolling(jobId) {
    pollRef.current = setInterval(async () => {
      try {
        const result = await getAnalysis(jobId);
        if (result.agents?.length) {
          setAgents(result.agents);
        }
        if (result.recommendation) {
          setRecommendation(result.recommendation);
        }
        if (result.status === "done" || result.status === "error") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setLoading(false);
          if (result.status === "error" && result.errors?.length) {
            setError(result.errors.join("; "));
          }
        }
      } catch {
        // Retry silently
      }
    }, 5000);
  }

  async function handleSubmit(override) {
    const query = (typeof override === "string" ? override : input).trim();
    if (!query) return;
    resetState();
    setLoading(true);

    try {
      const res = await submitQuery(query);

      if (res.mode === "qa") {
        setMode("qa");
        setTicker(res.ticker || null);
        setQaAnswer(res.answer);
        setQaSources(res.sources || []);
        setLoading(false);
        return;
      }

      // Analysis mode
      setMode("analysis");
      setTicker(res.ticker);

      // Open SSE stream
      sseRef.current = streamAnalysis(res.job_id, (evt) => {
        switch (evt.type) {
          case "phase":
            setPhase(evt.data.phase);
            if (evt.data.message) setPhaseMessage(evt.data.message);
            break;
          case "agent_completed":
            setAgents((prev) => [...prev, evt.data]);
            break;
          case "done":
            if (evt.data && evt.data.ticker) {
              setRecommendation(evt.data);
            }
            setPhase("done");
            setLoading(false);
            break;
          case "error":
            setError(evt.data?.message || "Analysis failed");
            setLoading(false);
            break;
        }
      });

      // Fallback polling in case SSE drops
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }

  return (
    <>
      <Navbar />

      <Hero
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        loading={loading}
        compact={mode !== null}
      />

      {error && <div className="error-box">{error}</div>}

      {/* Analysis view */}
      {mode === "analysis" && (
        <div className="analysis-container">
          {loading && phase && phase !== "done" && (
            <div className="phase-status">
              {phase === "downloading" && (
                <><div className="spinner" style={{ margin: "0 auto 12px" }} />Downloading SEC 10-K filings for {ticker}...</>
              )}
              {phase === "indexing" && (
                <><div className="spinner" style={{ margin: "0 auto 12px" }} />Building vector index from filings...</>
              )}
              {phase === "index_done" && "Index ready. Starting analysis..."}
              {(phase === "download_error" || phase === "index_error") && (
                <span style={{ color: "var(--amber)" }}>{phaseMessage || "Proceeding without filing data..."}</span>
              )}
              {phase === "routing" && "Routing query to agents..."}
              {phase === "context_gathered" && "Context gathered - agents analyzing..."}
            </div>
          )}

          <AgentGrid completedAgents={agents} />

          {recommendation && <RecommendationPanel rec={recommendation} />}

          {/* Full report below the grid once agents have completed */}
          {agents.length > 0 && <AgentReport agents={agents} />}
        </div>
      )}

      {/* QA view */}
      {mode === "qa" && (
        <QAResult answer={qaAnswer} sources={qaSources} ticker={ticker} query={input} />
      )}
    </>
  );
}
