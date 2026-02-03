import React, { useState, useEffect } from "react";
import { getCompanies } from "../api";

export default function AvailableCompanies({ onSelect }) {
  const [companies, setCompanies] = useState([]);

  useEffect(() => {
    getCompanies()
      .then((data) => setCompanies(data.companies || []))
      .catch(() => {});
  }, []);

  if (companies.length === 0) return null;

  return (
    <div className="tickers-row" style={{ marginTop: 12 }}>
      <span style={{ fontSize: 12, color: "var(--text-dim)", marginRight: 4 }}>
        Indexed:
      </span>
      {companies.map((c) => (
        <button
          key={c.ticker}
          className="ticker-chip indexed"
          onClick={() => onSelect(c.ticker)}
          title={`${c.files_count} files indexed`}
        >
          {c.ticker}
        </button>
      ))}
    </div>
  );
}
