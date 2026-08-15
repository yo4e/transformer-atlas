import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://yo4e.github.io",
  base: "/transformer-atlas",
  integrations: [mdx(), sitemap()],
  vite: {
    server: {
      allowedHosts: [".manus.computer"]
    }
  },
  markdown: {
    shikiConfig: {
      theme: "github-light"
    }
  }
});
