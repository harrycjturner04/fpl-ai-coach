/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import type { Plugin } from "vite";

/** Preloads the latin Archivo font so it downloads alongside the CSS, not after it. */
const preloadFont = (): Plugin => ({
  name: "preload-font",
  transformIndexHtml: (_html, ctx) =>
    Object.keys(ctx.bundle ?? {}).filter((f) => /archivo-latin-wdth-normal-.*\.woff2$/.test(f)).map((f) => ({
      tag: "link", injectTo: "head", attrs: { rel: "preload", as: "font", type: "font/woff2", crossorigin: "", href: `/${f}` },
    })),
});

export default defineConfig({
  plugins: [
    react(),
    preloadFont(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: "script-defer",
      manifest: {
        name: "FPL AI Coach",
        short_name: "Coach",
        theme_color: "#0E1A15",
        background_color: "#0E1A15",
        display: "standalone",
        icons: [
          { src: "icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any maskable" },
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
        ],
      },
      workbox: {
        runtimeCaching: [{ urlPattern: ({ url }) => /^\/api\/entries\/\d+\/snapshot$/.test(url.pathname), handler: "NetworkFirst", options: { cacheName: "snapshots", networkTimeoutSeconds: 5 } }],
      },
    }),
  ],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { env: { TZ: "UTC" }, environment: "jsdom", globals: true, setupFiles: ["./src/test/setup.ts"], include: ["src/**/*.test.{ts,tsx}"] },
});
