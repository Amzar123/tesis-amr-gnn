import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMR Graph Visualizer",
  description: "Interactive 2D/3D visualizer for AMR-based GNN graphs (.pt files)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
