import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const children = await Promise.all(entries.map(async (entry) => entry.isDirectory() ? files(path.join(directory, entry.name)) : [path.join(directory, entry.name)]));
  return children.flat();
}

const output = "dist";
const html = (await files(output)).filter((file) => file.endsWith(".html"));
const failures = [];
for (const file of html) {
  const source = await readFile(file, "utf8");
  for (const match of source.matchAll(/href="([^"#?]+)"/g)) {
    const href = match[1];
    if (!href.startsWith("/") || href.startsWith("//")) continue;
    const local = decodeURIComponent(href.replace(/^\//, ""));
    const targets = local === "" ? [path.join(output, "index.html")] : [path.join(output, local), path.join(output, local, "index.html")];
    const exists = await Promise.all(targets.map(async (target) => stat(target).then(() => true).catch(() => false)));
    if (!exists.some(Boolean)) failures.push(`${file}: ${href}`);
  }
}
if (failures.length) {
  console.error(`Broken internal links:\n${failures.join("\n")}`);
  process.exit(1);
}
console.log(`internal link check passed across ${html.length} HTML files.`);
