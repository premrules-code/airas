import React from "react";

export default function PriceTargetBar({ low, median, high, current }) {
  if (!low && !high) return null;

  const min = low || median || 0;
  const max = high || median || 100;
  const range = max - min || 1;

  const medianPct = median ? ((median - min) / range) * 100 : 50;
  const currentPct = current ? ((current - min) / range) * 100 : null;

  return (
    <div className="price-target-bar">
      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-muted)", marginBottom: 4 }}>
        Price Target Range
      </div>
      <div className="ptb-track">
        <div className="ptb-range" style={{ left: 0, width: "100%" }} />
        {median != null && (
          <div
            className="ptb-marker"
            style={{ left: `${Math.min(100, Math.max(0, medianPct))}%`, background: "var(--accent)" }}
            title={`Median: $${median}`}
          />
        )}
        {currentPct != null && (
          <div
            className="ptb-marker"
            style={{ left: `${Math.min(100, Math.max(0, currentPct))}%`, background: "var(--text)" }}
            title={`Current: $${current}`}
          />
        )}
      </div>
      <div className="ptb-labels">
        <span>${low ?? "N/A"}</span>
        <span>Median: ${median ?? "N/A"}</span>
        <span>${high ?? "N/A"}</span>
      </div>
    </div>
  );
}
