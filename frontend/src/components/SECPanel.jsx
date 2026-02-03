import React, { useState, useEffect } from "react";
import { downloadSEC, indexSEC, getSECStatus } from "../api";

export default function SECPanel({ ticker }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    getSECStatus(ticker).then(setStatus).catch(() => {});
  }, [ticker]);

  // Poll while running
  useEffect(() => {
    if (
      !ticker ||
      !status ||
      (status.download_status !== "running" && status.index_status !== "running")
    )
      return;

    const interval = setInterval(() => {
      getSECStatus(ticker).then(setStatus).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, [ticker, status?.download_status, status?.index_status]);

  async function handleDownload() {
    setLoading(true);
    try {
      await downloadSEC(ticker);
      setStatus((s) => ({ ...s, download_status: "running" }));
    } catch (e) {
      setStatus((s) => ({ ...s, download_status: "error", error: e.message }));
    }
    setLoading(false);
  }

  async function handleIndex() {
    setLoading(true);
    try {
      await indexSEC(ticker);
      setStatus((s) => ({ ...s, index_status: "running" }));
    } catch (e) {
      setStatus((s) => ({ ...s, index_status: "error", error: e.message }));
    }
    setLoading(false);
  }

  if (!ticker) return null;

  return (
    <div className="sec-panel">
      <h3>SEC Filings - {ticker}</h3>
      <div className="sec-actions">
        <button
          className="sec-btn"
          onClick={handleDownload}
          disabled={loading || status?.download_status === "running"}
        >
          {status?.download_status === "running" ? "Downloading..." : "Download 10-K"}
        </button>
        <button
          className="sec-btn"
          onClick={handleIndex}
          disabled={
            loading ||
            status?.index_status === "running" ||
            status?.download_status === "running"
          }
        >
          {status?.index_status === "running" ? "Indexing..." : "Build Index"}
        </button>
      </div>

      {status && (
        <div className="sec-status">
          Download: {status.download_status} | Index: {status.index_status}
          {status.files_count > 0 && ` | ${status.files_count} files`}
          {status.error && (
            <div style={{ color: "var(--red)", marginTop: 4 }}>{status.error}</div>
          )}
        </div>
      )}
    </div>
  );
}
