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
    const model = page.waitForResponse((response) => new URL(response.url()).pathname === modelPath);
    try {
      await page.goto(`${origin}/viewer/studio.html#asset=${encodeURIComponent(assets[0].id)}`, { waitUntil: 'domcontentloaded' });
      await expect(loading).toBeVisible();
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
