#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import axe from "axe-core";
import { chromium } from "playwright";

const args = parseArgs(process.argv.slice(2));

if (!args.html || !args.out) {
  console.error("Usage: node scripts/run-axe.mjs --html <path> --out <path>");
  process.exit(1);
}

const htmlPath = path.resolve(args.html);
const outPath = path.resolve(args.out);
const html = await fs.readFile(htmlPath, "utf8");
const preparedHtml = prepareHtmlSnapshot(html);

const browser = await chromium.launch();
try {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 633 },
  });

  await page.setContent(preparedHtml, {
    waitUntil: "domcontentloaded",
  });
  await page.addScriptTag({ content: axe.source });

  const results = await page.evaluate(async () => {
    return window.axe.run(document, {
      runOnly: {
        type: "tag",
        values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"],
      },
      resultTypes: ["violations", "passes", "incomplete", "inapplicable"],
      reporter: "v1",
    });
  });

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, `${JSON.stringify(results, null, 2)}\n`);
  console.log(`Wrote axe report to ${path.relative(process.cwd(), outPath)}`);
  console.log(`${results.violations.reduce((count, violation) => count + violation.nodes.length, 0)} violations across ${results.violations.length} rules`);
} finally {
  await browser.close();
}

function parseArgs(argv) {
  const parsed = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--html" || arg === "--out") {
      parsed[arg.slice(2)] = argv[index + 1];
      index += 1;
    }
  }

  return parsed;
}

function prepareHtmlSnapshot(input) {
  const baseTag = '<base href="https://www.bt.com/" />';
  const withoutScripts = input.replaceAll("<script", '<script type="application/x-blocked"');

  if (withoutScripts.includes("<head>")) {
    return withoutScripts.replace("<head>", `<head>${baseTag}`);
  }

  return `${baseTag}${withoutScripts}`;
}
