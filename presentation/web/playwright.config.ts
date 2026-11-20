import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://127.0.0.1:5173" },
  webServer: { command: "npm run dev", url: "http://127.0.0.1:5173", reuseExistingServer: !process.env.CI },
  projects: [
    { name: "iPhone SE", use: { ...devices["iPhone SE"] } },
    { name: "Pixel 7", use: { ...devices["Pixel 7"] } },
    { name: "Desktop Chrome", use: { ...devices["Desktop Chrome"] } },
  ],
});
