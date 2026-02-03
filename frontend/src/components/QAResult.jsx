import React, { useState } from "react";

/**
 * Parse answer text into structured blocks:
 * - Paragraphs (split on double newlines)
 * - Inline **bold** → <strong>
 * - Inline [Source N] → clickable citation superscript
 * - Lines starting with - or * → list items
 */
function renderAnswer(text, onCiteClick) {
  if (!text) return null;

  // Split into paragraphs on double newline
  const paragraphs = text.split(/\n\n+/);

  return paragraphs.map((para, pi) => {
    const trimmed = para.trim();
    if (!trimmed) return null;

    // Markdown headings: ## or ###
    const h2Match = trimmed.match(/^##\s+(.+)$/);
    if (h2Match) {
      return <h3 className="qa-heading" key={pi}>{renderInline(h2Match[1], onCiteClick)}</h3>;
    }
    const h3Match = trimmed.match(/^###\s+(.+)$/);
    if (h3Match) {
      return <h4 className="qa-subheading" key={pi}>{renderInline(h3Match[1], onCiteClick)}</h4>;
    }

    // Check if this paragraph is a list (bullet or numbered)
    const blockEl = renderBlock(trimmed, pi, onCiteClick);
    if (blockEl) return blockEl;

    // Mixed: heading lines + body text in the same block
    // (e.g. "## Title\nSome text" without a blank line between)
    const lines = trimmed.split("\n");
    if (/^#{2,3}\s+/.test(lines[0])) {
      const headLine = lines[0];
      const rest = lines.slice(1).join("\n").trim();
      const level = headLine.startsWith("###") ? "h4" : "h3";
      const headText = headLine.replace(/^#{2,3}\s+/, "");
      const HeadTag = level;
      const cls = level === "h3" ? "qa-heading" : "qa-subheading";

      // The rest might itself be a list
      const restBlock = rest ? renderBlock(rest, `${pi}-rest`, onCiteClick) : null;
      return (
        <React.Fragment key={pi}>
          <HeadTag className={cls}>{renderInline(headText, onCiteClick)}</HeadTag>
          {restBlock || (rest && <p className="qa-para">{renderInline(rest, onCiteClick)}</p>)}
        </React.Fragment>
      );
    }

    // Multi-line paragraph without heading — join lines
    return (
      <p className="qa-para" key={pi}>
        {renderInline(trimmed.replace(/\n/g, " "), onCiteClick)}
      </p>
    );
  });
}

/** Render inline formatting: **bold**, [Source N] citations */
function renderInline(text, onCiteClick) {
  // Split on **bold** and [Source N]
  const parts = text.split(/(\*\*[^*]+\*\*|\[Source \d+\])/g);

  return parts.map((part, i) => {
    // Bold
    const boldMatch = part.match(/^\*\*(.+)\*\*$/);
    if (boldMatch) {
      return <strong key={i}>{boldMatch[1]}</strong>;
    }
    // Citation
    const citeMatch = part.match(/^\[Source (\d+)\]$/);
    if (citeMatch) {
      const num = parseInt(citeMatch[1], 10);
      return (
        <button
          key={i}
          className="cite-link"
          onClick={(e) => {
            e.stopPropagation();
            onCiteClick(num);
          }}
          title={`View Source ${num}`}
        >
          {num}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

/** Render a block that may contain mixed lines: numbered lists, bullet lists, text */
function renderBlock(trimmed, pi, onCiteClick) {
  const lines = trimmed.split("\n");

  // All numbered list (1. 2. etc.)
  const isNumbered = lines.every((l) => /^\s*\d+[.)]\s/.test(l) || !l.trim());
  if (isNumbered) {
    return (
      <ol className="qa-olist" key={pi}>
        {lines
          .filter((l) => l.trim())
          .map((l, li) => (
            <li key={li}>{renderInline(l.replace(/^\s*\d+[.)]\s*/, ""), onCiteClick)}</li>
          ))}
      </ol>
    );
  }

  // All bullet list
  const isBullet = lines.every((l) => /^\s*[-*]\s/.test(l) || !l.trim());
  if (isBullet) {
    return (
      <ul className="qa-list" key={pi}>
        {lines
          .filter((l) => l.trim())
          .map((l, li) => (
            <li key={li}>{renderInline(l.replace(/^\s*[-*]\s*/, ""), onCiteClick)}</li>
          ))}
      </ul>
    );
  }

  return null;
}

export default function QAResult({ answer, sources, ticker, query }) {
  const [expanded, setExpanded] = useState({});

  if (!answer) return null;

  function toggleSource(id) {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <div className="qa-container">
      {/* Answer card — the hero of the page */}
      <div className="qa-card">
        <div className="qa-card-bar" />
        <div className="qa-card-inner">
          <div className="qa-card-top">
            <div>
              <span className="qa-ticker">{ticker}</span>
              <span className="qa-badge">AI Answer</span>
              {query && <div className="qa-query">{query}</div>}
            </div>
            {sources?.length > 0 && (
              <span className="qa-src-count">{sources.length} sources</span>
            )}
          </div>

          <div className="qa-answer-body">
            {renderAnswer(answer, toggleSource)}
          </div>
        </div>
      </div>

      {/* Source citations */}
      {sources?.length > 0 && (
        <div className="qa-citations">
          <div className="qa-citations-header">
            <span className="qa-citations-title">Referenced SEC Filings</span>
          </div>

          {sources.map((s) => (
            <div
              key={s.id}
              className={`qa-cite-card ${expanded[s.id] ? "open" : ""}`}
            >
              <button
                className="qa-cite-toggle"
                onClick={() => toggleSource(s.id)}
              >
                <span className="qa-cite-num">{s.id}</span>
                <div className="qa-cite-info">
                  <span className="qa-cite-label">{s.label || s.file}</span>
                  <span className="qa-cite-sub">
                    {[s.filing_type, s.filing_date, s.section]
                      .filter(Boolean)
                      .join("  \u00B7  ")}
                  </span>
                </div>
                <span className="qa-cite-chevron">{expanded[s.id] ? "\u25B2" : "\u25BC"}</span>
              </button>

              {expanded[s.id] && s.text && (
                <div className="qa-cite-text">{s.text}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
