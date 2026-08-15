import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const output = "dist";
const base = "/transformer-atlas";

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => entry.isDirectory() ? files(path.join(directory, entry.name)) : [path.join(directory, entry.name)]));
  return nested.flat();
}

function withBase(url) {
  if (!url.startsWith("/") || url.startsWith("//") || url === base || url.startsWith(`${base}/`)) return url;
  return `${base}${url}`;
}

const targets = (await files(output)).filter((file) => file.endsWith(".html") || file.endsWith(".css"));
let rewrites = 0;
for (const file of targets) {
  const original = await readFile(file, "utf8");
  let updated = original.replace(/\b(href|src)="(\/[^\"]*)"/g, (_full, attribute, url) => {
    const replacement = withBase(url);
    if (replacement !== url) rewrites += 1;
    return `${attribute}="${replacement}"`;
  });
  updated = updated.replace(/url\((['"]?)(\/[^)'\"]*)\1\)/g, (_full, quote, url) => {
    const replacement = withBase(url);
    if (replacement !== url) rewrites += 1;
    return `url(${quote}${replacement}${quote})`;
  });
  if (updated !== original) await writeFile(file, updated, "utf8");
}

console.log(`GitHub Pages base normalization complete: ${rewrites} internal references rewritten across ${targets.length} files.`);
