#!/usr/bin/env node
/**
 * Playwright browser tests for the MCP-RAG SPA.
 *
 * Usage:
 *   BASE_URL=http://localhost:18060 node spa-tests.js
 *
 * Each test outputs a line starting with PASS: or FAIL: for the e2e runner.
 */

const { chromium } = require('playwright');

const BASE_URL = process.env.BASE_URL || 'http://localhost:18060';

let passed = 0;
let failed = 0;

function pass(msg) {
  console.log(`PASS: ${msg}`);
  passed++;
}

function fail(msg, detail) {
  console.log(`FAIL: ${msg}${detail ? ' — ' + detail : ''}`);
  failed++;
}

async function test(name, fn) {
  try {
    await fn();
    pass(name);
  } catch (e) {
    fail(name, e.message);
  }
}

async function main() {
  console.log(`Playwright browser tests (base: ${BASE_URL})`);
  console.log('');

  // Use system Chrome (google-chrome-stable) when available,
  // fall back to Playwright's bundled chromium.
  let launchOptions = { headless: true };
  try {
    const browser = await chromium.launch({ ...launchOptions, channel: 'chrome' });
    await browser.close();
    launchOptions.channel = 'chrome';
    console.log('  Using system Chrome');
  } catch {
    console.log('  Using Playwright bundled Chromium');
  }

  const browser = await chromium.launch(launchOptions);
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });

  try {
    // ── Health endpoint (no page needed) ──────────────────────
    const healthPage = await context.newPage();
    await test('GET /health returns status=healthy', async () => {
      const resp = await healthPage.goto(`${BASE_URL}/health`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      const json = await resp.json();
      if (json.status !== 'healthy') throw new Error(`status=${json.status}`);
      if (!json.healthy) throw new Error('missing healthy field');
      if (typeof json.config_revision !== 'number') throw new Error('missing config_revision');
    });
    await healthPage.close();

    // ── OpenAPI spec ──────────────────────────────────────────
    const apiPage = await context.newPage();
    await test('GET /openapi.json returns valid spec', async () => {
      const resp = await apiPage.goto(`${BASE_URL}/openapi.json`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      const json = await resp.json();
      if (json.openapi !== '3.0.3') throw new Error(`openapi version=${json.openapi}`);
      if (!json.info?.title) throw new Error('missing info.title');
    });
    await apiPage.close();

    // ── SPA: /app loads ───────────────────────────────────────
    const appPage = await context.newPage();
    await test('GET /app loads SPA with title', async () => {
      const resp = await appPage.goto(`${BASE_URL}/app`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      const title = await appPage.title();
      if (!title) throw new Error('page has no title');
      console.log(`    SPA title: "${title}"`);
    });

    await test('SPA: sidebar navigation links visible', async () => {
      // Wait for sidebar to render (Vite SPA structure)
      await appPage.waitForTimeout(2000);
      // Look for common nav elements
      const navLinks = await appPage.locator('nav a, [role="navigation"] a, .sidebar a').all();
      if (navLinks.length === 0) {
        // Try broader: any link on the page
        const allLinks = await appPage.locator('a').all();
        if (allLinks.length === 0) {
          throw new Error('no navigation links found');
        }
        console.log(`    Found ${allLinks.length} links (no dedicated nav)`);
      } else {
        console.log(`    Found ${navLinks.length} sidebar links`);
      }
    });

    await test('SPA: page content renders', async () => {
      // Get body text
      const text = await appPage.locator('body').innerText();
      if (!text || text.trim().length === 0) {
        throw new Error('page body is empty');
      }
      console.log(`    Body text length: ${text.length} chars`);
    });

    // ── SPA: /documents page ──────────────────────────────────
    await test('GET /app/documents loads', async () => {
      const resp = await appPage.goto(`${BASE_URL}/app/documents`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      await appPage.waitForTimeout(1000);
      const text = await appPage.locator('body').innerText();
      if (!text || text.trim().length === 0) {
        throw new Error('documents page is empty');
      }
    });

    // ── SPA: /config page ─────────────────────────────────────
    await test('GET /app/config loads', async () => {
      const resp = await appPage.goto(`${BASE_URL}/app/config`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      await appPage.waitForTimeout(1000);
      const text = await appPage.locator('body').innerText();
      if (!text || text.trim().length === 0) {
        throw new Error('config page is empty');
      }
    });

    // ── SPA: /mcp page ───────────────────────────────────────
    await test('GET /app/mcp loads', async () => {
      const resp = await appPage.goto(`${BASE_URL}/app/mcp`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      await appPage.waitForTimeout(1000);
      const text = await appPage.locator('body').innerText();
      if (!text || text.trim().length === 0) {
        throw new Error('MCP page is empty');
      }
    });

    // ── Docs page ─────────────────────────────────────────────
    const docsPage = await context.newPage();
    await test('GET /docs loads Scalar API docs', async () => {
      const resp = await docsPage.goto(`${BASE_URL}/docs`, { waitUntil: 'networkidle' });
      if (resp.status() !== 200) throw new Error(`HTTP ${resp.status()}`);
      // Scalar should have rendered something
      await docsPage.waitForTimeout(1500);
      const html = await docsPage.content();
      if (html.length < 200) throw new Error('docs page too small, likely not rendered');
    });
    await docsPage.close();

    // ── Redirects ─────────────────────────────────────────────
    const redirectPage = await context.newPage();
    await test('GET / redirects to /app', async () => {
      await redirectPage.goto(`${BASE_URL}/`, { waitUntil: 'commit' });
      const finalUrl = redirectPage.url();
      if (!finalUrl.includes('/app')) {
        throw new Error(`landed at ${finalUrl}, expected /app`);
      }
    });

    await test('GET /doc redirects to /docs', async () => {
      await redirectPage.goto(`${BASE_URL}/doc`, { waitUntil: 'commit' });
      const finalUrl = redirectPage.url();
      if (!finalUrl.includes('/docs')) {
        throw new Error(`landed at ${finalUrl}, expected /docs`);
      }
    });

    await test('GET /documents-page redirects to /app/documents', async () => {
      await redirectPage.goto(`${BASE_URL}/documents-page`, { waitUntil: 'commit' });
      const finalUrl = redirectPage.url();
      if (!finalUrl.includes('/app/documents')) {
        throw new Error(`landed at ${finalUrl}, expected /app/documents`);
      }
    });

    await test('GET /config-page redirects to /app/config', async () => {
      await redirectPage.goto(`${BASE_URL}/config-page`, { waitUntil: 'commit' });
      const finalUrl = redirectPage.url();
      if (!finalUrl.includes('/app/config')) {
        throw new Error(`landed at ${finalUrl}, expected /app/config`);
      }
    });
    await redirectPage.close();

    // ── Screenshot ────────────────────────────────────────────
    await test('Screenshot: /app page captured', async () => {
      await appPage.goto(`${BASE_URL}/app`, { waitUntil: 'networkidle' });
      await appPage.waitForTimeout(1000);
      await appPage.screenshot({ path: '/tmp/e2e-spa-screenshot.png', fullPage: false });
      console.log('    Screenshot saved to /tmp/e2e-spa-screenshot.png');
    });

  } finally {
    await context.close();
    await browser.close();
  }

  console.log('');
  console.log(`Playwright results: ${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error(`FAIL: Playwright fatal error: ${e.message}`);
  process.exit(1);
});
