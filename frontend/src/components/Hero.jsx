import React from "react";
import AvailableCompanies from "./AvailableCompanies";

const POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"];

const EXAMPLE_QUERIES = [
  {
    category: "Full Analysis",
    icon: "chart",
    queries: [
      "AAPL",
      "NVDA",
      "TSLA",
    ],
  },
  {
    category: "Financial Q&A",
    icon: "question",
    queries: [
      "What is Apple's revenue growth?",
      "Tesla debt to equity ratio",
      "NVDA profit margins and trends",
    ],
  },
  {
    category: "Comparisons",
    icon: "compare",
    queries: [
      "AAPL vs MSFT revenue",
      "Apple vs Nvidia profit margins",
      "Compare Tesla and Ford financials",
    ],
  },
  {
    category: "Deep Dive",
    icon: "dive",
    queries: [
      "What are Microsoft's key risks?",
      "AAPL competitive advantages",
      "Tesla earnings highlights and outlook",
    ],
  },
];

export default function Hero({ value, onChange, onSubmit, loading, compact }) {
  function handleKey(e) {
    if (e.key === "Enter" && value.trim()) onSubmit();
  }

  function fillTicker(t) {
    onChange(t);
  }

  function fillAndSubmit(q) {
    onChange(q);
    onSubmit(q);
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

      <div className="example-queries">
        <h3 className="example-queries-title">Try asking</h3>
        <div className="example-grid">
          {EXAMPLE_QUERIES.map((group) => (
            <div key={group.category} className="example-group">
              <div className={`example-group-header ${group.icon}`}>
                <span className="example-group-icon">
                  {group.icon === "chart" && "\u25B2"}
                  {group.icon === "question" && "?"}
                  {group.icon === "compare" && "\u21C4"}
                  {group.icon === "dive" && "\u25CE"}
                </span>
                <span className="example-group-label">{group.category}</span>
              </div>
              <ul className="example-list">
                {group.queries.map((q) => (
                  <li key={q}>
                    <button
                      className="example-query-btn"
                      onClick={() => fillAndSubmit(q)}
                      disabled={loading}
                    >
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      <div className="features-row">
        <span className="feature-badge">10 AI Agents</span>
        <span className="feature-badge">SEC Filing RAG</span>
        <span className="feature-badge">Real-time Streaming</span>
        <span className="feature-badge">Claude + GPT-4</span>
      </div>
    </section>
  );
}
