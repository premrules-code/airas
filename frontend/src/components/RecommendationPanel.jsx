import React from "react";
import DonutChart from "./DonutChart";

function catColor(score) {
  if (score >= 75) return "var(--green)";
  if (score >= 60) return "var(--blue)";
  if (score >= 45) return "var(--amber)";
  return "var(--red)";
}

export default function RecommendationPanel({ rec }) {
  if (!rec) return null;

  const badgeCls = rec.recommendation.toLowerCase().replace(" ", "-");

  return (
    <div className="recommendation-panel">
      <div className="rec-header">
        <h2>{rec.ticker} Analysis</h2>
        <span className={`rec-badge ${badgeCls}`}>{rec.recommendation}</span>
      </div>

      {/* Category scores */}
      <div className="category-scores">
        {[
          { label: "Financial", value: rec.financial_score },
          { label: "Technical", value: rec.technical_score },
          { label: "Sentiment", value: rec.sentiment_score },
          { label: "Risk", value: rec.risk_score },
        ].map((c) => (
          <div className="cat-score" key={c.label}>
            <div className="cat-value" style={{ color: catColor(c.value) }}>
              {c.value}
            </div>
            <div className="cat-label">{c.label}</div>
          </div>
        ))}
      </div>

      <div className="rec-body">
        <DonutChart
          score={rec.overall_score}
          color={rec.overall_color}
        />

        <div className="rec-details">
          {rec.bullish_factors?.length > 0 && (
            <>
              <h3>Strengths</h3>
              <ul className="rec-list strengths">
                {rec.bullish_factors.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </>
          )}

          {rec.risks?.length > 0 && (
            <>
              <h3>Risks</h3>
              <ul className="rec-list risks">
                {rec.risks.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </>
          )}

          {rec.thesis && <div className="rec-thesis">{rec.thesis}</div>}
        </div>
      </div>
    </div>
  );
}
