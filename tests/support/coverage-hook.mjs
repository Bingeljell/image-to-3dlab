import { readFileSync, writeFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { threadId } from 'node:worker_threads';
import { instrument } from './coverage-instrument.mjs';

const inventory = process.env.AF_COVERAGE_INVENTORY;
const rawDir = process.env.AF_COVERAGE_JS_RAW;
const root = process.env.AF_COVERAGE_ROOT || process.cwd();

if (inventory && rawDir) {
  const files = new Set(JSON.parse(readFileSync(inventory, 'utf8')).javascript);
  const instrumenter = instrument();
  registerHooks({
    load(url, context, nextLoad) {
      const result = nextLoad(url, context);
      if (!url.startsWith('file:')) return result;
      const file = relative(root, fileURLToPath(url)).split(sep).join('/');
      if (!files.has(file)) return result;
      const source = result.source ?? readFileSync(fileURLToPath(url));
      return { ...result, source: instrumenter.instrumentSync(source.toString(), file) };
    },
  });

  process.on('exit', () => {
    if (globalThis.__coverage__) {
      const name = `${process.pid}-${threadId}.json`;
      writeFileSync(join(rawDir, name), JSON.stringify(globalThis.__coverage__));
    }
  });
}
