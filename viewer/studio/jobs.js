// Running a step: start a job, follow it, report progress, a live log, and how it ended.
// Every step's API answers a start with {job_id, status_url | events_url} and its status
// endpoint with {status, last_event, log_tail}; this follows any of them the same way.

export const TERMINAL = new Set(['done', 'error', 'cancelled']);
const STALL_SECONDS = 180;

/** The best ready 3D engine first, then the other ready ones (from GET /api/catalog). */
export function readyEngines(catalog) {
  return (catalog?.backends || [])
    .filter((b) => b.kind === '3d' && b.supported_here !== false && b.build_present !== false
      && (b.weights || []).every((w) => w.present !== false))
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
    .map((b) => ({ id: b.id, label: b.label, license: b.license?.name || '', caveat: b.caveat || null }));
}

/** Where to poll a job: the start response's status_url, or the conventional path. */
export function statusUrlFor(kind, start) {
  if (start.status_url) return start.status_url;
  return `/api/${kind}/${start.job_id}/status`;
}

/** A progress number from an event, if the step reports one (0-100), else null. */
export function percentOf(event) {
  for (const key of ['overall_pct', 'pct', 'percent', 'progress']) {
    const value = Number(event?.[key]);
    if (Number.isFinite(value)) return Math.max(0, Math.min(100, value <= 1 && key === 'progress' ? value * 100 : value));
  }
  return null;
}

/** Turn a raw failure into words a person can act on. */
export function plainError(message = '') {
  const text = String(message);
  const rules = [
    [/memory|MemoryError|out of memory|Killed/i, 'This Mac ran out of memory. Close other big apps and try again, or pick a lower detail or texture size.'],
    [/not installed|is not installed\/ready|needs_setup/i, 'This engine is not installed yet. Install it from Setup, then try again.'],
    [/is running; wait|wait for it to finish/i, 'Another job is running. This one can start when it finishes.'],
    [/humanoid|map.*SOMA|could not map/i, "This doesn't look like a humanoid, so the preset moves can't fit it. You can still download the rig and animate it in Blender."],
    [/Blender/i, 'Blender stopped with an error. The full log is below; "Copy details" makes it easy to report.'],
  ];
  for (const [pattern, words] of rules) if (pattern.test(text)) return words;
  return text || 'The step stopped without saying why. The log below has the details.';
}

/**
 * Follow a started job until it ends. Calls onUpdate({status, event, percent, log, stalled})
 * every poll, then resolves with the final status payload. Polling (not an event stream)
 * because a dropped stream that never reconnects makes a finished job look hung.
 */
export function followJob(statusUrl, { onUpdate, interval = 1200, fetchImpl = fetch, now = () => Date.now() } = {}) {
  let stopped = false;
  let lastChange = now(), lastKey = '';
  const done = new Promise((resolve, reject) => {
    const tick = async () => {
      if (stopped) return;
      try {
        const response = await fetchImpl(statusUrl);
        const payload = await response.json();
        const event = payload.last_event || {};
        const key = JSON.stringify(event) + (payload.log_tail || '').length;
        if (key !== lastKey) { lastKey = key; lastChange = now(); }
        const stalled = (now() - lastChange) / 1000 > STALL_SECONDS;
        onUpdate?.({ status: payload.status, event, percent: percentOf(event), log: payload.log_tail || '', stalled });
        if (TERMINAL.has(payload.status)) { resolve(payload); return; }
      } catch (error) {
        // a missed poll is not a failure; the next one catches up
      }
      setTimeout(tick, interval);
    };
    tick().catch(reject);
  });
  return { done, stop() { stopped = true; } };
}

// engine chatter that means nothing to a person: kernel compiles, verbose/debug lines, blanks
// engine chatter, and the picture model's tokenizer dumps (<|im_start|>, token lists with Ġ/Ċ markers)
const NOISE = /\[(VERBOSE|DEBUG)\]|ggml_|kernel_|compile_pipeline|th_max|^\s*[|\-=]*\s*$|^\s*0x[0-9a-f]+|<\|im_(start|end)\|>|\btokens \[|[\u0120\u010a]/i;

/** The last `count` lines of a log worth showing to a person. */
export function readableLog(text, count = 4) {
  return String(text || '').split('\n').map((line) => line.trimEnd()).filter((line) => line && !NOISE.test(line)).slice(-count).join('\n');
}

/**
 * How a finished step reads: a cancel is calm and has no log (nothing to debug); a failure
 * says why in plain words and keeps its log.
 */
export function jobEnd(final) {
  if (final.status === 'cancelled') return { cancelled: true, why: 'Cancelled. Everything finished before this step is kept.', log: '' };
  return { cancelled: false, why: plainError(final.last_event?.message || final.log_tail || ''), log: final.log_tail || '' };
}
