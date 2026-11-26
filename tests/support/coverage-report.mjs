import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import istanbulCoverage from 'istanbul-lib-coverage';
import { instrument } from './coverage-instrument.mjs';

const [root, rawDir, ...files] = process.argv.slice(2);
const coverage = istanbulCoverage.createCoverageMap();
const instrumenter = instrument();

// Parse every source file to get its real statement and branch census, even
// when no test evaluated the module. Hooks and reporting use the same parser.
for (const file of files) {
  instrumenter.instrumentSync(readFileSync(join(root, file), 'utf8'), file);
  coverage.addFileCoverage(instrumenter.lastFileCoverage());
}

for (const filename of readdirSync(rawDir).filter(name => name.endsWith('.json'))) {
  const hits = JSON.parse(readFileSync(join(rawDir, filename), 'utf8'));
  for (const [file, record] of Object.entries(hits)) {
    const census = coverage.fileCoverageFor(file);
    for (const key of ['statementMap', 'fnMap', 'branchMap']) {
      if (JSON.stringify(record[key]) !== JSON.stringify(census[key])) {
        throw new Error(`Coverage locations changed while measuring ${file}`);
      }
    }
  }
  coverage.merge(hits);
}

function counts(summary) {
  return Object.fromEntries(['lines', 'branches'].map(metric => {
    const { covered, total } = summary[metric];
    return [metric, { covered, total }];
  }));
}

const results = {};
for (const file of coverage.files()) {
  const measured = coverage.fileCoverageFor(file);
  results[file] = {
    ...counts(measured.toSummary()),
    missing_lines: measured.getUncoveredLines().map(Number),
    branch_locations: measured.branchMap,
    branch_hits: measured.b,
  };
}

console.log(JSON.stringify({ files: results, totals: counts(coverage.getCoverageSummary()) }, null, 2));
