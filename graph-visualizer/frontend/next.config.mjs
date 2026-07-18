/** @type {import('next').NextConfig} */
const nextConfig = {
  // These packages ship as pure ESM (.mjs) — Next.js 14 must transpile them
  transpilePackages: [
    "react-force-graph-2d",
    "react-force-graph-3d",
    "force-graph",
    "3d-force-graph",
    "three-forcegraph",
    "three-render-objects",
    "d3-force-3d",
    "react-kapsule",
    "kapsule",
    "accessor-fn",
  ],
  webpack: (config) => {
    // Prevent canvas from being bundled server-side
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    return config;
  },
};

export default nextConfig;
