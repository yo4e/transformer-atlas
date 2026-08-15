import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = "src/content";
const coreOrders = new Set([3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27]);
const requiredSignals = ["問題", "直感", "数値", "Python", "誤解", "確認"];

async function markdownFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return entries.filter((entry) => entry.isFile() && entry.name.endsWith(".md") && entry.name !== "README.md").map((entry) => path.join(directory, entry.name)).sort();
}

const chapters = await markdownFiles(path.join(root, "chapters"));
const mathPages = await markdownFiles(path.join(root, "math"));
const failures = [];
const paragraphs = new Map();

if (chapters.length !== 28) failures.push(`Expected 28 chapters, found ${chapters.length}.`);
if (mathPages.length < 10) failures.push(`Expected at least 10 math pages, found ${mathPages.length}.`);

for (const file of chapters) {
  const content = await readFile(file, "utf8");
  const order = Number(path.basename(file).slice(0, 2));
  const body = content.replace(/^---[\s\S]*?---\s*/, "");
  const minCharacters = coreOrders.has(order) ? 5500 : 4500;
  if (!content.startsWith("---\n")) failures.push(`${file}: missing frontmatter.`);
  if (!content.includes("status: complete")) failures.push(`${file}: incomplete status.`);
  if (body.length < minCharacters) failures.push(`${file}: only ${body.length} characters; expected at least ${minCharacters}.`);
  for (const signal of requiredSignals) if (!content.includes(signal)) failures.push(`${file}: missing explanatory signal ${signal}.`);
  if (!/```python[\s\S]+?```/.test(body)) failures.push(`${file}: missing transparent Python example.`);
  if (!/https?:\/\//.test(content)) failures.push(`${file}: missing source URL.`);
  if (!/\d/.test(body)) failures.push(`${file}: missing numeric worked example.`);
  if (/next_value\s*=\s*transform\(/.test(body)) failures.push(`${file}: contains first-draft placeholder code.`);
  for (const paragraph of body.split(/\n\s*\n/).map((item) => item.replace(/[`*_>#]/g, "").replace(/\s+/g, " ").trim()).filter((item) => item.length > 120 && !item.startsWith("http"))) {
    const entries = paragraphs.get(paragraph) ?? [];
    entries.push(file);
    paragraphs.set(paragraph, entries);
  }
}

for (const [paragraph, files] of paragraphs) if (files.length > 2) failures.push(`Repeated long boilerplate across ${files.length} chapters: ${paragraph.slice(0, 80)}…`);
for (const file of mathPages) {
  const body = (await readFile(file, "utf8")).replace(/^---[\s\S]*?---\s*/, "");
  if (body.length < 3000) failures.push(`${file}: math note is only ${body.length} characters.`);
  if (!/```python[\s\S]+?```/.test(body)) failures.push(`${file}: missing Python example.`);
}
const attention = await readFile(path.join(root, "chapters", "08-self-attention-を最後まで通す.md"), "utf8");
for (const signal of [/[QＱ]/, /[KＫ]/, /[VＶ]/, /Q\s*K\^?T/, /softmax/i, /output/i]) if (!signal.test(attention)) failures.push(`Chapter 08: complete attention path missing ${signal}.`);

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`second-draft checks passed: ${chapters.length} chapters, ${mathPages.length} math notes, ${[...paragraphs.values()].length} unique substantive paragraphs.`);
