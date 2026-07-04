const { chromium } = require("playwright");
const fs = require("fs");

const site = process.argv[2];

if (!site) {
  console.error("Usage: node a11y-tree.js <site>");
  process.exit(1);
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
  });

  const page = await browser.newPage();

  await page.goto(site, {
    waitUntil: "networkidle",
  });

  const cdp = await page.context().newCDPSession(page);

  await cdp.send("Accessibility.enable");

  const { nodes } = await cdp.send(
    "Accessibility.getFullAXTree"
  );

  fs.writeFileSync(
    "a11y-tree.json",
    JSON.stringify(nodes, null, 2)
  );

  console.log(`Fetched: ${site}`);
  console.log(`Saved ${nodes.length} AX nodes`);

  await browser.close();
})();