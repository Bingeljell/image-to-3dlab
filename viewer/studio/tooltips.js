/** Pick a tooltip side that has room in the current viewport. */
export function tooltipPlacement(rect, viewport, preferred = 'down') {
  const vertical = preferred === 'up' && rect.top < 150
    ? 'down'
    : preferred === 'down' && viewport.height - rect.bottom < 150
      ? 'up'
      : preferred;
  const horizontal = rect.left < 140 ? 'left' : viewport.width - rect.right < 140 ? 'right' : '';
  return { vertical, horizontal };
}

/** Reposition shared data-tip pseudo-tooltips whenever their target is hovered or focused. */
export function installTooltips(root = document) {
  const place = (event) => {
    const target = event.target?.closest?.('[data-tip]');
    if (!target) return;
    const preferred = target.classList.contains('tip-up') ? 'up' : 'down';
    const { vertical, horizontal } = tooltipPlacement(target.getBoundingClientRect(), {
      width: window.innerWidth,
      height: window.innerHeight,
    }, preferred);
    target.dataset.tipV = vertical;
    if (horizontal) target.dataset.tipH = horizontal;
    else delete target.dataset.tipH;
  };
  root.addEventListener('pointerover', place);
  root.addEventListener('focusin', place);
}
