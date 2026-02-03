import React from "react";
import ScoreBadge from "./ScoreBadge";

export default function AgentReport({ agents }) {
  // Only show completed agents that have content (exclude pending placeholders)
  const completed = agents.filter((a) => a.display_score != null);
  if (completed.length === 0) return null;

  return (
    <div className="agent-report">
      <h2 className="report-heading">Agent Reports</h2>
      {completed.map((agent) => (
        <div className="report-section" key={agent.agent_name}>
          <div className="report-section-header">
            <div className="report-section-left">
              <span className="report-agent-name">{agent.label}</span>
              <span className="report-agent-desc">{agent.description}</span>
            </div>
            <div className="report-section-right">
              <span className="report-score" style={{ color: agent.color }}>
                {agent.display_score}
              </span>
              <ScoreBadge signal={agent.signal} />
            </div>
          </div>

          {agent.summary && (
            <p className="report-summary">{agent.summary}</p>
          )}

          <div className="report-columns">
            {agent.strengths?.length > 0 && (
              <div className="report-col">
                <h4 className="report-col-title strengths-title">Strengths</h4>
                <ul className="report-col-list">
                  {agent.strengths.map((s, i) => (
                    <li key={i} className="strength-item">{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {agent.weaknesses?.length > 0 && (
              <div className="report-col">
                <h4 className="report-col-title weaknesses-title">Weaknesses</h4>
                <ul className="report-col-list">
                  {agent.weaknesses.map((w, i) => (
                    <li key={i} className="weakness-item">{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {agent.sources?.length > 0 && (
            <div className="report-sources">
              <span className="report-sources-label">Data Sources:</span>
              {agent.sources.map((src, i) => (
                <span key={i} className="report-source-tag">{src}</span>
              ))}
            </div>
          )}

          {agent.confidence != null && (
            <div className="report-confidence">
              Confidence: {Math.round(agent.confidence * 100)}%
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
