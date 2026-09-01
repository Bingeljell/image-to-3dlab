export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return 'estimating…';
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return `${value}s`;
  const minutes = Math.round(value / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** Shared stage-list and overall-progress renderer for local workshop jobs. */
export class JobProgressPanel {
  constructor({ stages, bar, label, eta }) {
    this.elements = { stages, bar, label, eta };
    this.phases = [];
    this.rows = new Map();
  }

  configure(meta) {
    const normalized = meta || { stages: ['running'], stage_labels: { running: 'Running' } };
    this.phases = normalized.stages.map((phase) => [
      phase,
      normalized.stage_labels[phase] || phase,
    ]);
    this.rows.clear();
    this.elements.stages.innerHTML = '';
    for (const [phase, label] of this.phases) {
      const row = document.createElement('div');
      row.className = 'stage-row';
      row.dataset.phase = phase;
      row.innerHTML = `<span class="stage-dot">○</span><span>${label}</span>` +
        '<span class="stage-detail">queued</span>';
      this.elements.stages.appendChild(row);
      this.rows.set(phase, row);
    }
  }

  reset() {
    for (const row of this.rows.values()) {
      row.className = 'stage-row';
      row.querySelector('.stage-dot').textContent = '○';
      row.querySelector('.stage-detail').textContent = 'queued';
    }
  }

  apply(event) {
    const phase = event.phase;
    if (!this.rows.has(phase)) return;

    const current = this.phases.findIndex(([name]) => name === phase);
    for (let index = 0; index < current; index++) {
      const row = this.rows.get(this.phases[index][0]);
      row.className = 'stage-row done';
      row.querySelector('.stage-dot').textContent = '✓';
      row.querySelector('.stage-detail').textContent = 'done';
    }

    const row = this.rows.get(phase);
    const percent = event.stage_pct == null ? '' : `${event.stage_pct}%`;
    const steps = event.step != null ? `${event.step}/${event.total}` : percent;
    const done = event.stage_pct === 100;
    row.className = done ? 'stage-row done' : 'stage-row active';
    row.querySelector('.stage-dot').textContent = done ? '✓' : '●';
    row.querySelector('.stage-detail').textContent = done
      ? 'done'
      : `${steps} · ~${formatDuration(event.stage_eta_seconds)}`;
    this.elements.bar.style.width = `${Math.max(0, Math.min(100, event.overall_pct || 0))}%`;
    this.elements.label.textContent = event.message || phase;
    this.elements.eta.textContent = event.total_eta_seconds == null
      ? 'Total still estimating'
      : `Total ~${formatDuration(event.total_eta_seconds)}`;
  }
}

