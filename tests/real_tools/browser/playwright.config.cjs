const path = require('node:path');
const { defineConfig } = require('@playwright/test');

const results = path.resolve(__dirname, '../../../real-tool-results/browser');

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: 'studio.spec.cjs',
  outputDir: path.join(results, 'test-results'),
  reporter: [['list'], ['html', { outputFolder: path.join(results, 'report'), open: 'never' }]],
  timeout: 60_000,
  globalTimeout: 180_000,
  workers: 1,
  retries: 0,
  forbidOnly: Boolean(process.env.CI),
  use: {
    browserName: 'chromium',
    viewport: { width: 1280, height: 840 },
    launchOptions: {
      args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
    screenshot: 'only-on-failure',
    trace: 'on',
  },
});
