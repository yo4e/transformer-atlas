import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = "src/content";
const requiredSections = ["## 問題", "## 直感", "## 実際の機構", "## 小さな例", "## この比喩の限界", "## 実際の LLM との接続", "## 理解を確かめる", "## 参照"];

async function markdownFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return entries.filter((entry) => entry.isFile() && entry.name.endsWith(".md") && entry.name !== "README.md").map((entry) => path.join(directory, entry.name));
}

const chapters = await markdownFiles(path.join(root, "chapters"));
const mathPages = await markdownFiles(path.join(root, "math"));
const failures = [];

if (chapters.length !== 28) failures.push(`Expected 28 chapters, found ${chapters.length}.`);
if (mathPages.length < 10) failures.push(`Expected at least 10 math pages, found ${mathPages.length}.`);

for (const file of chapters) {
  const content = await readFile(file, "utf8");
  if (!content.startsWith("---\n")) failures.push(`${file}: missing frontmatter.`);
  for (const section of requiredSections) if (!content.includes(section)) failures.push(`${file}: missing ${section}.`);
  if (!content.includes("status: complete")) failures.push(`${file}: incomplete status.`);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`content checks passed: ${chapters.length} chapters, ${mathPages.length} math pages.`);
