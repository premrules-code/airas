import React from "react";
import AvailableCompanies from "./AvailableCompanies";

const POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"];

export default function Hero({ value, onChange, onSubmit, loading, compact }) {
  function handleKey(e) {
    if (e.key === "Enter" && value.trim()) onSubmit();
  }

  function fillTicker(t) {
    onChange(t);
  }

  // Compact mode: just the search bar + indexed companies
  if (compact) {
    return (
      <section className="hero compact">
        <div className="search-box">
          <input
            type="text"
            placeholder={'Try "AAPL" or "What is Apple\'s revenue?"'}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKey}
          />
          <button onClick={onSubmit} disabled={loading || !value.trim()}>
            {loading ? "..." : "Go"}
          </button>
        </div>
        <AvailableCompanies onSelect={fillTicker} />
      </section>
    );
  }

  return (
    <section className="hero">
      <h1>AIRAS Investment Analyst</h1>
      <p>
        Enter a ticker for full 10-agent analysis, or ask any question about a
        company's financials.
      </p>

      <div className="search-box">
        <input
          type="text"
          placeholder={'Try "AAPL" or "What is Apple\'s revenue?"'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          autoFocus
        />
        <button onClick={onSubmit} disabled={loading || !value.trim()}>
          {loading ? "..." : "Go"}
        </button>
      </div>

      <div className="tickers-row">
        {POPULAR.map((t) => (
          <button key={t} className="ticker-chip" onClick={() => fillTicker(t)}>
            {t}
          </button>
        ))}
      </div>

      <AvailableCompanies onSelect={fillTicker} />

      <div className="features-row">
        <span className="feature-badge">10 AI Agents</span>
        <span className="feature-badge">SEC Filing RAG</span>
        <span className="feature-badge">Real-time Streaming</span>
        <span className="feature-badge">Claude + GPT-4</span>
      </div>
    </section>
  );
}
