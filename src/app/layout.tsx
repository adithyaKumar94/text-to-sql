import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Clinical SQL Chat",
  description: "RAG-powered SQL agent for clinical schema",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ background: "#f8fafc" }}>
        <Navbar />
        {children}
      </body>
    </html>
  );
}
