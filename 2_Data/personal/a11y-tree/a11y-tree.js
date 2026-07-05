const { chromium } = require("playwright");
const fs = require("fs");
const YAML = require("yaml");

// Parse arguments
const args = process.argv.slice(2);
const isYaml = args.includes("--yaml") || args.includes("-y");

// Filter out flags to find the target site URL
const site = args.find(arg => !arg.startsWith("-"));

if (!site) {
  console.error("Usage: node a11y-tree.js <site> [--yaml | -y]");
  process.exit(1);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto(site, { waitUntil: "networkidle" });

  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Accessibility.enable");

  const { nodes } = await cdp.send("Accessibility.getFullAXTree");

  // Determine format and filename based on the flag
  const filename = isYaml ? "a11y-tree.yaml" : "a11y-tree.json";
  const outputData = isYaml ? YAML.stringify(nodes) : JSON.stringify(nodes, null, 2);

  fs.writeFileSync(filename, outputData);

  console.log(`Fetched: ${site}`);
  console.log(`Saved ${nodes.length} AX nodes to ${filename}`);

  await browser.close();
})();