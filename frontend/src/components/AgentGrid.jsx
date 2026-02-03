import React from "react";
import AgentCard from "./AgentCard";

/** Static metadata for the 11 agent slots (10 agents + synthesis) */
const AGENT_SLOTS = [
  { agent_name: "financial_analyst", label: "Financial Analyst", icon: "chart-bar", description: "Revenue, margins, balance sheet health" },
  { agent_name: "news_sentiment", label: "News Sentiment", icon: "newspaper", description: "Recent news tone and impact" },
  { agent_name: "technical_analyst", label: "Technical Analyst", icon: "chart-line", description: "Price trends, indicators, patterns" },
  { agent_name: "risk_assessment", label: "Risk Assessment", icon: "shield", description: "Volatility, downside risk factors" },
  { agent_name: "competitive_analysis", label: "Competitive Analysis", icon: "users", description: "Market position vs peers" },
  { agent_name: "insider_activity", label: "Insider Activity", icon: "user-check", description: "Executive buying/selling patterns" },
  { agent_name: "options_analysis", label: "Options Analysis", icon: "layers", description: "Options flow, implied volatility" },
  { agent_name: "social_sentiment", label: "Social Sentiment", icon: "message-circle", description: "Social media buzz and tone" },
  { agent_name: "earnings_analysis", label: "Earnings Analysis", icon: "dollar-sign", description: "Earnings quality, beat/miss history" },
  { agent_name: "analyst_ratings", label: "Analyst Ratings", icon: "star", description: "Wall Street consensus and targets" },
  { agent_name: "synthesis", label: "Enhanced Synthesis", icon: "zap", description: "Weighted composite score" },
];

export default function AgentGrid({ completedAgents }) {
  const agentMap = {};
  for (const a of completedAgents) {
    agentMap[a.agent_name] = a;
  }

  return (
    <div className="agent-grid">
      {AGENT_SLOTS.map((slot) => {
        const completed = agentMap[slot.agent_name];
        return (
          <AgentCard
            key={slot.agent_name}
            agent={completed || slot}
            pending={!completed}
          />
        );
      })}
    </div>
  );
}
