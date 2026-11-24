/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      manifest: {
        name: "FPL AI Coach",
        short_name: "Coach",
        theme_color: "#0E1A15",
        background_color: "#0E1A15",
        display: "standalone",
        icons: [{ src: "icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any maskable" }],
      },
      workbox: {
        runtimeCaching: [{ urlPattern: /\/api\/entries\/.*\/snapshot/, handler: "StaleWhileRevalidate", options: { cacheName: "snapshots" } }],
      },
    }),
  ],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { env: { TZ: "UTC" }, environment: "jsdom", globals: true, setupFiles: ["./src/test/setup.ts"], include: ["src/**/*.test.{ts,tsx}"] },
});
