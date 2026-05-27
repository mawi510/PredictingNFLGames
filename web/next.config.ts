import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a fully static site (out/) that nginx serves on EC2 — no Node runtime.
  // The app is client-rendered and fetches the API at runtime, so this is safe.
  output: "export",
  trailingSlash: true,
  devIndicators: false
};

export default nextConfig;
