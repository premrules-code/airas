import React from "react";

export default function ScoreBadge({ signal }) {
  if (!signal) return null;
  const cls = signal.toLowerCase().replace(" ", "-");
  return <span className={`score-badge ${cls}`}>{signal}</span>;
}
