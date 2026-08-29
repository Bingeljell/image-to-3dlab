import * as THREE from '../vendor/three.module.js';

const DEFAULT_COLOR = new THREE.Color(0x65b7e8);
const SELECTED_COLOR = new THREE.Color(0xffc857);
const INSTANCE_MATRIX = new THREE.Matrix4();
const WORLD_POSITION = new THREE.Vector3();

/** Selectable joint markers layered over a deform skeleton. */
export class BonePicker {
  constructor({ scene, camera, canvas, bones, onSelect, markerRadius = 0.014 }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.bones = [...bones];
    this.onSelect = onSelect || (() => {});
    this.selectedIndex = -1;
    this.pointerStart = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    this.geometry = new THREE.SphereGeometry(markerRadius, 12, 8);
    this.material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.92,
    });
    this.markers = new THREE.InstancedMesh(this.geometry, this.material, this.bones.length);
    this.markers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.markers.frustumCulled = false;
    this.markers.renderOrder = 20;
    for (let index = 0; index < this.bones.length; index++) {
      this.markers.setColorAt(index, DEFAULT_COLOR);
    }
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    this.scene.add(this.markers);

    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
    this.canvas.addEventListener('pointerdown', this.handlePointerDown);
    this.canvas.addEventListener('pointerup', this.handlePointerUp);
    this.update();
  }

  update() {
    for (let index = 0; index < this.bones.length; index++) {
      this.bones[index].getWorldPosition(WORLD_POSITION);
      INSTANCE_MATRIX.makeTranslation(WORLD_POSITION.x, WORLD_POSITION.y, WORLD_POSITION.z);
      this.markers.setMatrixAt(index, INSTANCE_MATRIX);
    }
    this.markers.instanceMatrix.needsUpdate = true;
  }

  pick(clientX, clientY) {
    if (!this.markers.visible || !this.bones.length) return null;
    const bounds = this.canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return null;
    this.pointer.set(
      ((clientX - bounds.left) / bounds.width) * 2 - 1,
      -((clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObject(this.markers, false)[0];
    if (hit?.instanceId == null) {
      this.clear();
      return null;
    }
    return this.select(hit.instanceId);
  }

  select(index) {
    if (index < 0 || index >= this.bones.length) return null;
    if (this.selectedIndex >= 0) this.markers.setColorAt(this.selectedIndex, DEFAULT_COLOR);
    this.selectedIndex = index;
    this.markers.setColorAt(index, SELECTED_COLOR);
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    const bone = this.bones[index];
    this.onSelect(bone, index);
    return bone;
  }

  clear() {
    if (this.selectedIndex >= 0) {
      this.markers.setColorAt(this.selectedIndex, DEFAULT_COLOR);
      if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    }
    this.selectedIndex = -1;
    this.onSelect(null, -1);
  }

  setVisible(visible) {
    this.markers.visible = visible;
  }

  handlePointerDown(event) {
    if (event.button !== 0) return;
    this.pointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY };
  }

  handlePointerUp(event) {
    if (!this.pointerStart || this.pointerStart.id !== event.pointerId) return;
    const distance = Math.hypot(
      event.clientX - this.pointerStart.x,
      event.clientY - this.pointerStart.y,
    );
    this.pointerStart = null;
    // OrbitControls uses the same pointer. A drag rotates the camera; only a click selects.
    if (distance <= 4) this.pick(event.clientX, event.clientY);
  }

  dispose() {
    this.canvas.removeEventListener('pointerdown', this.handlePointerDown);
    this.canvas.removeEventListener('pointerup', this.handlePointerUp);
    this.markers.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }

  get selectedBone() {
    return this.selectedIndex >= 0 ? this.bones[this.selectedIndex] : null;
  }
}

