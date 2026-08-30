const AXIS_INDEX = { X: 0, Y: 1, Z: 2 };
const EPSILON = 1e-7;

/** Own editable fit-joint corrections without mutating the loaded sidecar. */
export class RigCorrectionSession {
  constructor(sidecar) {
    this.source = clone(sidecar);
    this.corrections = clone(sidecar.corrections || {});
    this.undoStack = [];
    this.redoStack = [];
  }

  target(id) {
    this.requireJoint(id);
    return [...(this.corrections[id]?.targetPosition || this.source.joints[id].position)];
  }

  setTarget(id, targetPosition, { mirror = false } = {}) {
    this.requireJoint(id);
    const target = vector3(targetPosition);
    const before = clone(this.corrections);
    this.writeCorrection(id, target, false);

    const mirrorId = mirror ? this.mirrorPartner(id) : null;
    if (mirrorId) this.writeCorrection(mirrorId, this.reflect(target), true);
    this.record(before);
    return mirrorId;
  }

  reset(id, { mirror = false } = {}) {
    this.requireJoint(id);
    const before = clone(this.corrections);
    delete this.corrections[id];
    const mirrorId = mirror ? this.mirrorPartner(id) : null;
    if (mirrorId) delete this.corrections[mirrorId];
    this.record(before);
    return mirrorId;
  }

  resetAll() {
    const before = clone(this.corrections);
    this.corrections = {};
    this.record(before);
  }

  undo() {
    if (!this.undoStack.length) return false;
    this.redoStack.push(clone(this.corrections));
    this.corrections = this.undoStack.pop();
    return true;
  }

  redo() {
    if (!this.redoStack.length) return false;
    this.undoStack.push(clone(this.corrections));
    this.corrections = this.redoStack.pop();
    return true;
  }

  toSidecar() {
    return { ...clone(this.source), corrections: clone(this.corrections) };
  }

  mirrorPartner(id) {
    const direct = this.source.joints[id].mirrorOf;
    if (direct) return direct;
    return Object.keys(this.source.joints)
      .find((candidate) => this.source.joints[candidate].mirrorOf === id) || null;
  }

  reflect(position) {
    const reflected = [...position];
    const index = AXIS_INDEX[this.source.mirror.axis];
    const origin = this.source.mirror.origin;
    reflected[index] = 2 * origin - reflected[index];
    return reflected;
  }

  writeCorrection(id, target, mirrored) {
    const sourcePosition = this.source.joints[id].position;
    if (sameVector(sourcePosition, target)) {
      delete this.corrections[id];
      return;
    }
    this.corrections[id] = {
      sourcePosition: [...sourcePosition],
      targetPosition: [...target],
      delta: target.map((value, axis) => value - sourcePosition[axis]),
      mirrored,
    };
  }

  record(before) {
    if (JSON.stringify(before) === JSON.stringify(this.corrections)) return;
    this.undoStack.push(before);
    this.redoStack = [];
  }

  requireJoint(id) {
    if (!this.source.joints[id]) throw new Error(`Unknown fit joint: ${id}`);
  }
}

function vector3(value) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(item))) {
    throw new Error('Joint target must contain exactly three finite numbers');
  }
  return value.map(Number);
}

function sameVector(left, right) {
  return left.every((value, axis) => Math.abs(value - right[axis]) <= EPSILON);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}
