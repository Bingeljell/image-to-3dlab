import * as THREE from '../vendor/three.module.js';

const DIRECTIONS = {
  front: { direction: [0, 0, 1], up: [0, 1, 0] },
  back: { direction: [0, 0, -1], up: [0, 1, 0] },
  left: { direction: [-1, 0, 0], up: [0, 1, 0] },
  right: { direction: [1, 0, 0], up: [0, 1, 0] },
  top: { direction: [0, 1, 0], up: [0, 0, -1] },
  bottom: { direction: [0, -1, 0], up: [0, 0, 1] },
};

const DEFAULT_POSITION = new THREE.Vector3(1.45, 0.6, 2.45);
const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);

/** Snap a model viewport camera without changing model, pose, or rig-edit state. */
export function setCameraView(view, name) {
  if (!view?.camera || !view?.controls) return false;
  const preset = DIRECTIONS[name];
  if (name !== 'reset' && !preset) return false;

  const target = name === 'reset' ? DEFAULT_TARGET : view.controls.target;
  const currentDistance = view.camera.position.distanceTo(view.controls.target);
  const distance = Number.isFinite(currentDistance) && currentDistance > 0.1
    ? currentDistance
    : DEFAULT_POSITION.length();

  if (name === 'reset') {
    view.controls.target.copy(DEFAULT_TARGET);
    view.camera.position.copy(DEFAULT_POSITION);
    view.camera.up.set(0, 1, 0);
  } else {
    const direction = new THREE.Vector3(...preset.direction);
    view.camera.position.copy(target).addScaledVector(direction, distance);
    view.camera.up.set(...preset.up);
  }
  view.camera.lookAt(view.controls.target);
  view.camera.updateProjectionMatrix();
  view.controls.update();
  return true;
}

/** Create the standard six-view and camera-reset toolbar used by 3D workshop rooms. */
export function createCameraViewControls({ container, getView, onChange }) {
  const toolbar = document.createElement('div');
  toolbar.className = 'viewport-camera-controls';
  toolbar.setAttribute('aria-label', 'Camera views');

  const buttons = [
    ['front', 'Front', 'F'],
    ['back', 'Back', 'B'],
    ['left', 'Left', 'L'],
    ['right', 'Right', 'R'],
    ['top', 'Top', 'T'],
    ['bottom', 'Bottom', 'D'],
    ['reset', 'Reset View', 'Reset'],
  ].map(([name, label, shortLabel]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.cameraView = name;
    button.textContent = shortLabel;
    button.title = label;
    button.setAttribute('aria-label', label);
    button.disabled = true;
    button.onclick = () => {
      if (setCameraView(getView(), name)) onChange?.(name);
    };
    toolbar.appendChild(button);
    return button;
  });

  container.appendChild(toolbar);
  return {
    element: toolbar,
    setEnabled(enabled) { buttons.forEach((button) => { button.disabled = !enabled; }); },
    dispose() { toolbar.remove(); },
  };
}
