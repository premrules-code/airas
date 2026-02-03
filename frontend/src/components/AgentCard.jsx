import React, { useState } from "react";
import ScoreBadge from "./ScoreBadge";

const ICON_BG = {
  "#10b981": "var(--green-bg)",
  "#3b82f6": "var(--blue-bg)",
  "#f59e0b": "var(--amber-bg)",
  "#ef4444": "var(--red-bg)",
};

const ICON_MAP = {
  "chart-bar": "\u2593",
  "newspaper": "\u2709",
  "chart-line": "\u2197",
  "shield": "\u26E8",
  "users": "\u2694",
  "user-check": "\u2713",
  "layers": "\u25A3",
  "message-circle": "\u2709",
  "dollar-sign": "$",
  "star": "\u2605",
  "zap": "\u26A1",
  circle: "\u25CF",
};

export default function AgentCard({ agent, pending }) {
  const [expanded, setExpanded] = useState(false);

  if (pending || !agent) {
    return (
      <div className="agent-card pending">
        <div className="card-header">
          <div
            className="card-icon"
            style={{ background: "var(--border)", color: "var(--text-dim)" }}
          >
            {ICON_MAP[agent?.icon] || "\u25CF"}
          </div>
        </div>
        <div className="card-title">{agent?.label || "Waiting..."}</div>
        <div className="card-desc">{agent?.description || ""}</div>
        <div className="loading-bar" />
      </div>
    );
  }

  const bg = ICON_BG[agent.color] || "var(--blue-bg)";
  const hasDetails =
    agent.summary || agent.strengths?.length > 0 || agent.weaknesses?.length > 0;

  return (
    <div
      className={`agent-card ${expanded ? "expanded" : ""}`}
      onClick={() => hasDetails && setExpanded(!expanded)}
      style={{ cursor: hasDetails ? "pointer" : "default" }}
    >
      <div className="card-header">
        <div className="card-icon" style={{ background: bg, color: agent.color }}>
          {ICON_MAP[agent.icon] || "\u25CF"}
        </div>
        <ScoreBadge signal={agent.signal} />
      </div>
      <div className="card-title">{agent.label}</div>
      <div className="card-desc">{agent.description}</div>
      <div className="card-score" style={{ color: agent.color }}>
        {agent.display_score}
      </div>

      {/* Collapsed: show truncated summary */}
      {!expanded && agent.summary && (
        <div className="card-summary-preview">{agent.summary}</div>
      )}

      {/* Expanded: full detail */}
      {expanded && (
        <div className="card-expanded">
          {agent.summary && (
            <div className="card-full-summary">{agent.summary}</div>
          )}
          {agent.strengths?.length > 0 && (
            <div className="card-section">
              <div className="card-section-title strengths-title">Strengths</div>
              <ul className="card-detail-list">
                {agent.strengths.map((s, i) => (
                  <li key={i} className="strength-item">{s}</li>
                ))}
              </ul>
            </div>
          )}
          {agent.weaknesses?.length > 0 && (
            <div className="card-section">
              <div className="card-section-title weaknesses-title">Weaknesses</div>
              <ul className="card-detail-list">
                {agent.weaknesses.map((w, i) => (
                  <li key={i} className="weakness-item">{w}</li>
                ))}
              </ul>
            </div>
          )}
          {agent.sources?.length > 0 && (
            <div className="card-section">
              <div className="card-section-title sources-title">Sources</div>
              <div className="card-sources">
                {agent.sources.map((src, i) => (
                  <span key={i} className="card-source-tag">{src}</span>
                ))}
              </div>
            </div>
          )}
          <div className="card-confidence">
            Confidence: {Math.round((agent.confidence || 0) * 100)}%
          </div>
        </div>
      )}

      {hasDetails && !expanded && (
        <div className="card-expand-hint">Click to expand</div>
      )}
    </div>
  );
}
