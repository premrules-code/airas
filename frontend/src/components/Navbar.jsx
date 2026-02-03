import React from "react";

export default function Navbar() {
  return (
    <nav className="navbar">
      <a href="/" className="navbar-brand">
        <span>AIRAS</span>
        <span className="navbar-subtitle">AI Investment Research & Analysis System</span>
      </a>
      <div className="navbar-status">Online</div>
    </nav>
  );
}
