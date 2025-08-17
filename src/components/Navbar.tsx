"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { CSSProperties } from "react";

const linkStyle = (active: boolean): CSSProperties => ({
  padding: "8px 10px",
  borderRadius: 8,
  textDecoration: "none",
  color: active ? "#fff" : "#334155",
  background: active ? "#2563eb" : "transparent",
  fontWeight: 600,
});

export default function Navbar() {
  const pathname = usePathname() || "/";

  return (
    <nav style={{ borderBottom: "1px solid #e5e7eb", background: "#fff" }}>
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
        }}
      >
        {/* Left: logo + title + suffix */}
        <div style={{ display: "flex", alignItems: "center" }}>
          <img src="/logo-eg.svg" alt="Company logo" width={28} height={28} />
          <h1 style={{ fontWeight: 700, marginLeft: 8, fontSize: 28 }}>
            - Healthcare
          </h1>
        </div>

        {/* Right: nav links */}
        <div style={{ display: "flex", gap: 8 }}>
          <Link href="/" style={linkStyle(pathname === "/")}>
            Home
          </Link>
          <Link href="/docs" style={linkStyle(pathname.startsWith("/docs"))}>
            Docs
          </Link>
        </div>
      </div>
    </nav>
  );
}
