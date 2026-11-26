const path = require('node:path');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { createInterface } = require('node:readline');
const { test, expect } = require('@playwright/test');

const repo = path.resolve(__dirname, '../../..');
const python = process.env.PYTHON || path.join(
  repo, '.venv', ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']),
);
const modelPath = '/output/browser-triangle/browser-triangle.glb';

async function withDeadline(promise, milliseconds, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function runStudio(info, use) {
  const server = spawn(python, ['-B', path.join(__dirname, 'serve_fixture.py')], { cwd: repo });
  const lines = createInterface({ input: server.stdout });
  const closed = new Promise((resolve) => server.once('close', resolve));
  let log = '';
  server.on('error', (error) => { log += `${error.message}\n`; });
  server.stderr.setEncoding('utf8').on('data', (message) => { log += message; });
  try {
    const starting = Promise.race([
      once(lines, 'line').then(([line]) => line),
      closed.then((code) => { throw new Error(`Studio fixture exited (${code})\n${log}`); }),
    ]);
    const origin = await withDeadline(starting, 15_000, 'Studio fixture did not start within 15 seconds');
    expect(origin).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    await use(origin);
  } finally {
    lines.close();
    server.stdin.end();
    let exitCode;
    try {
      exitCode = await withDeadline(closed, 5_000, 'Studio fixture did not exit after stdin closed');
    } catch (error) {
      server.kill('SIGKILL');
      await withDeadline(closed, 5_000, 'Studio fixture did not exit after SIGKILL');
      throw error;
    } finally {
      if (log) await info.attach('studio-server.log', { body: log, contentType: 'text/plain' });
    }
    if (info.status === 'passed') expect(exitCode, 'Studio fixture exit code').toBe(0);
  }
}

async function magentaPixels(page, screenshot) {
  // Decode a screenshot of the composited canvas after the real WebGL draw.
  // Reading the WebGL buffer directly can see discarded pixels between frames.
  return page.evaluate(async (bytes) => {
    const bitmap = await createImageBitmap(new Blob([Uint8Array.from(bytes)], { type: 'image/png' }));
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0);
    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    bitmap.close();
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 100 && data[i + 2] > 100 && data[i + 1] < 65) count++;
    }
    return count;
  }, [...screenshot]);
}

async function saveScreenshot(page, info, name) {
  const screenshot = info.outputPath(`${name}.png`);
  await page.screenshot({ path: screenshot });
  await info.attach(name, { path: screenshot, contentType: 'image/png' });
}

test('Studio loads and draws a library model, toggles wireframe, and reports a broken file', async ({ page, request }, info) => {
  await runStudio(info, async (origin) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('response', (response) => {
      if (response.status() >= 400) errors.push(`HTTP ${response.status()} ${response.url()}`);
    });
    page.on('requestfailed', (request) => {
      errors.push(`${request.failure().errorText} ${request.url()}`);
    });

    const releaseModel = Promise.withResolvers();
    await page.route('**/*', async (route) => {
      const { origin: target, protocol, pathname } = new URL(route.request().url());
      if (target !== origin || route.request().method() !== 'GET') {
        if (protocol === 'blob:') return route.continue();
        errors.push(`Unexpected browser request: ${route.request().method()} ${route.request().url()}`);
        return route.abort();
      }
      // Pause the genuine file request to prove the loading state is observable.
      if (pathname === modelPath) await releaseModel.promise;
      await route.continue();
    });

    const library = await request.get(`${origin}/api/assets`);
    expect(library.ok()).toBe(true);
    const { assets } = await library.json();
    expect(assets).toHaveLength(1);
    expect(assets[0].model).toBe(modelPath.slice('/output/'.length));

    const loading = page.getByText('Loading model…', { exact: true });
    const canvas = page.locator('.sv-pane canvas');
    let model;
    try {
      await page.goto(`${origin}/viewer/studio.html#asset=${encodeURIComponent(assets[0].id)}`, { waitUntil: 'domcontentloaded' });
      await expect(loading).toBeVisible();
      model = page.waitForResponse((response) => new URL(response.url()).pathname === modelPath);
    } catch (error) {
      if (errors.length) await info.attach('studio-browser-errors.log', { body: errors.join('\n'), contentType: 'text/plain' });
      throw error;
    } finally {
      releaseModel.resolve();
    }
    expect((await model).ok()).toBe(true);
    await expect(loading).toBeHidden();
    await expect(canvas).toBeVisible();
    const renderer = await canvas.evaluate((element) => {
      const gl = element.getContext('webgl2');
      const extension = gl?.getExtension('WEBGL_debug_renderer_info');
      return extension && gl.getParameter(extension.UNMASKED_RENDERER_WEBGL);
    });
    expect(renderer, 'Studio must run WebGL on SwiftShader').toMatch(/SwiftShader/i);

    const wireframe = page.getByRole('button', { name: 'Wireframe', exact: true });
    let filledPixels;
    await expect.poll(async () => {
      filledPixels = await magentaPixels(page, await canvas.screenshot());
      return filledPixels;
    }).toBeGreaterThan(1000);
    await saveScreenshot(page, info, 'studio-model');

    await wireframe.click();
    await expect(wireframe).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(async () => {
      const pixels = await magentaPixels(page, await canvas.screenshot());
      return pixels > 50 && pixels < filledPixels / 4;
    }, { message: 'Wireframe should draw magenta edges without filling the triangle' }).toBe(true);
    await saveScreenshot(page, info, 'studio-wireframe');

    await page.locator('#openFile').setInputFiles({
      name: 'broken.glb',
      mimeType: 'model/gltf-binary',
      buffer: Buffer.from('not a GLB'),
    });
    await expect(page.getByText(/^This model didn't load:/)).toBeVisible();
    await expect(loading).toBeHidden();
    await saveScreenshot(page, info, 'studio-model-error');
    expect(errors).toEqual([]);
  });
});

test('the winding demo boots from its shipped page and reverses a triangle', async ({ page }, info) => {
  await runStudio(info, async (origin) => {
    const errors = [];
    const imports = new Map([
      ['https://unpkg.com/three@0.170.0/build/three.module.js',
        path.join(repo, 'viewer/vendor/three.module.js')],
      ['https://unpkg.com/three@0.170.0/examples/jsm/controls/OrbitControls.js',
        path.join(repo, 'viewer/vendor/controls/OrbitControls.js')],
      ['https://unpkg.com/three@0.170.0/examples/jsm/renderers/CSS2DRenderer.js',
        path.join(__dirname, 'fixtures/CSS2DRenderer.js')],
    ]);
    const loadedImports = new Set();
    page.on('pageerror', (error) => errors.push(error.message));
    await page.route('**/*', (route) => {
      const url = route.request().url();
      if (new URL(url).origin === origin) return route.continue();
      if (!imports.has(url)) {
        errors.push(`Unexpected winding demo request: ${url}`);
        return route.abort();
      }
      loadedImports.add(url);
      return route.fulfill({
        path: imports.get(url),
        contentType: 'text/javascript',
        headers: { 'access-control-allow-origin': '*' },
      });
    });
    await page.goto(`${origin}/understanding_3d/mesh-winding-demo.html`);
    await expect(page.locator('#canvas-container canvas')).toBeVisible();
    const index = page.locator('#index-display');
    const status = page.locator('#status');
    const flip = page.getByRole('button', { name: 'Flip Winding of Highlighted Face' });
    await expect(index).toHaveText('0, 1, 2');
    await flip.click();
    await expect(index).toHaveText('0, 2, 1');
    await expect(status).toContainText('Clockwise (back');
    await flip.click();
    await expect(index).toHaveText('0, 1, 2');
    await expect(status).toContainText('Counter-clockwise (front');
    expect([...loadedImports].sort()).toEqual([...imports.keys()].sort());
    expect(errors).toEqual([]);
  });
});

test('Studio lets the user choose among saved animations in an opened model', async ({ page, request }, info) => {
  await runStudio(info, async (origin) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const { assets } = await (await request.get(`${origin}/api/assets`)).json();
    await page.goto(`${origin}/viewer/studio.html#asset=${encodeURIComponent(assets[0].id)}`);
    await expect(page.locator('.sv-pane canvas')).toBeVisible();
    await expect(page.getByText('Loading model…', { exact: true })).toBeHidden();
    const animated = await request.get(`${origin}/browser-moves.glb`);
    expect(animated.ok()).toBe(true);
    const input = page.locator('#openFile');
    await input.setInputFiles({
      name: 'browser-moves.glb', mimeType: 'model/gltf-binary', buffer: await animated.body(),
    });
    const picker = page.getByRole('combobox', { name: 'Choose a move' });
    const frame = page.locator('[data-frame]');
    const play = page.getByRole('button', { name: 'Play or pause' });
    await expect(picker.locator('option')).toHaveText(['Choose a move', 'Attack', 'Attack.001', 'Idle']);
    await expect(picker).toHaveValue('Choose a move');
    await expect(play).toBeDisabled();
    await picker.selectOption({ label: 'Attack.001' });
    await expect(frame).toHaveText(/^frame \d+ \/ 60$/);
    await expect(play).toHaveAttribute('data-tip', 'Pause');
    await picker.selectOption({ label: 'Attack' });
    await expect(frame).toHaveText(/^frame \d+ \/ 30$/);
    await input.setInputFiles({
      name: 'browser-triangle.glb', mimeType: 'model/gltf-binary',
      buffer: await (await request.get(`${origin}${modelPath}`)).body(),
    });
    await expect(picker).toHaveCount(0);
    await expect(page.locator('.sv-pane canvas')).toBeVisible();
    expect(errors).toEqual([]);
  });
});
