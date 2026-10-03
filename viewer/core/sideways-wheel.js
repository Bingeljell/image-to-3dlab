// The menu bar scrolls sideways when the window is narrow, with its scrollbar hidden so it
// stays one 42px row. A trackpad swipes it sideways on its own; a plain mouse wheel only
// turns up and down, and would never reach the last tabs. This turns that into sideways
// scroll: the new scrollLeft, or null to leave the wheel to the browser.
const LINE = 16;

export function sidewaysScroll({ deltaX, deltaY, deltaMode = 0 },
  { scrollLeft, scrollWidth, clientWidth }) {
  const room = scrollWidth - clientWidth;
  if (room <= 0 || Math.abs(deltaX) >= Math.abs(deltaY)) return null;
  const step = deltaMode === 1 ? deltaY * LINE : deltaMode === 2 ? deltaY * clientWidth : deltaY;
  const next = Math.max(0, Math.min(room, scrollLeft + step));
  return next === scrollLeft ? null : next;
}
