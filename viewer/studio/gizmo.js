// Axis map maths: where each axis end sits in the little circle for a camera angle.
// No DOM here, so Node tests import this exact file.

export const AXES = [
  { key: '+x', vector: [1, 0, 0], axis: 'x', view: 'right', label: 'Right side (+X)' },
  { key: '-x', vector: [-1, 0, 0], axis: 'x', view: 'left', label: 'Left side (−X)' },
  { key: '+y', vector: [0, 1, 0], axis: 'y', view: 'top', label: 'Top (+Y)' },
  { key: '-y', vector: [0, -1, 0], axis: 'y', view: 'bottom', label: 'Bottom (−Y)' },
  { key: '+z', vector: [0, 0, 1], axis: 'z', view: 'front', label: 'Front (+Z)' },
  { key: '-z', vector: [0, 0, -1], axis: 'z', view: 'back', label: 'Back (−Z)' },
];

/**
 * Project the six axis ends onto the screen for a camera looking at the origin from
 * `cameraDirection` (a unit vector from the target to the camera) with `up` as screen up.
 * Returns screen x (right) and y (down) in [-1, 1], plus depth (towards the viewer is
 * positive), sorted far to near so near ends draw on top.
 */
export function projectAxes(cameraDirection, up = [0, 1, 0]) {
  const back = normalise(cameraDirection);
  let right = cross(up, back);
  if (length(right) < 1e-6) right = cross([0, 0, -1], back); // looking straight down or up
  right = normalise(right);
  const trueUp = cross(back, right);
  return AXES.map((axis) => ({
    ...axis,
    x: dot(axis.vector, right),
    y: -dot(axis.vector, trueUp),
    depth: dot(axis.vector, back),
  })).sort((a, b) => a.depth - b.depth);
}

function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function length(a) { return Math.hypot(a[0], a[1], a[2]); }
function normalise(a) { const n = length(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }
