import * as THREE from '../vendor/three.module.js';

const NORMAL = new THREE.Color(0xf0a95b);
const SELECTED = new THREE.Color(0xffe08a);
const MATRIX = new THREE.Matrix4();

/** Render a validated armature-local fit skeleton over the normalized workshop model. */
export class FitSkeletonOverlay {
  constructor({ scene, sidecar, bones, camera = null, canvas = null, onSelect = null,
    onMove = null, onDragChange = null, markerRadius = 0.018 }) {
    this.scene = scene;
    this.sidecar = sidecar;
    this.bonesByName = new Map(bones.map((bone) => [bone.name, bone]));
    this.jointIds = Object.keys(sidecar.joints);
    this.jointIndex = new Map(this.jointIds.map((id, index) => [id, index]));
    this.selectedId = null;
    this.positions = new Map();
    this.camera = camera;
    this.canvas = canvas;
    this.onSelect = onSelect;
    this.onMove = onMove;
    this.onDragChange = onDragChange;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.dragPlane = new THREE.Plane();
    this.dragPoint = new THREE.Vector3();
    this.dragOffset = new THREE.Vector3();
    this.dragTarget = new THREE.Vector3();
    this.dragJointId = null;
    this.handlePointerDown = (event) => {
      if (event.button !== 0) return;
      const id = this.pick(event.clientX, event.clientY);
      if (!id) return;
      this.consume(event);
      this.onSelect?.(id);
      this.dragJointId = id;
      const position = this.positions.get(id);
      this.camera.getWorldDirection(this.dragTarget);
      this.dragPlane.setFromNormalAndCoplanarPoint(this.dragTarget, position);
      this.setRay(event.clientX, event.clientY);
      this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint);
      this.dragOffset.subVectors(position, this.dragPoint);
      this.canvas.setPointerCapture?.(event.pointerId);
      this.canvas.style.cursor = 'grabbing';
      this.onDragChange?.(true);
    };
    this.handlePointerMove = (event) => {
      if (!this.dragJointId) return;
      this.consume(event);
      this.setRay(event.clientX, event.clientY);
      if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) return;
      this.dragTarget.copy(this.dragPoint).add(this.dragOffset);
      const local = this.armatureNode.worldToLocal(this.dragTarget.clone()).toArray();
      this.setJointLocalPosition(this.dragJointId, local);
      this.onMove?.(this.dragJointId, local, false);
    };
    this.handlePointerUp = (event) => {
      if (!this.dragJointId) return;
      this.consume(event);
      const id = this.dragJointId;
      const local = this.armatureNode.worldToLocal(this.positions.get(id).clone()).toArray();
      this.dragJointId = null;
      this.canvas.releasePointerCapture?.(event.pointerId);
      this.canvas.style.cursor = '';
      this.onMove?.(id, local, true);
      this.onDragChange?.(false);
    };

    const mappedBone = this.jointIds
      .map((id) => this.bonesByName.get(sidecar.joints[id].sourceBone))
      .find(Boolean);
    this.armatureNode = mappedBone ? findArmatureNode(mappedBone) : null;
    if (!this.armatureNode) throw new Error('Rig sidecar does not map to any deform bone in this GLB');
    this.armatureNode.updateWorldMatrix(true, false);

    for (const id of this.jointIds) {
      const joint = sidecar.joints[id];
      const values = sidecar.corrections[id]?.targetPosition || joint.position;
      const position = new THREE.Vector3(...values);
      this.armatureNode.localToWorld(position);
      this.positions.set(id, position);
    }

    this.markerGeometry = new THREE.SphereGeometry(markerRadius, 12, 8);
    this.markerMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.95,
    });
    this.markers = new THREE.InstancedMesh(
      this.markerGeometry,
      this.markerMaterial,
      this.jointIds.length,
    );
    this.markers.frustumCulled = false;
    this.markers.renderOrder = 30;
    this.jointIds.forEach((id, index) => {
      MATRIX.makeTranslation(...this.positions.get(id).toArray());
      this.markers.setMatrixAt(index, MATRIX);
      this.markers.setColorAt(index, NORMAL);
    });
    this.markers.instanceMatrix.needsUpdate = true;
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    this.scene.add(this.markers);

    const linePositions = [];
    for (const id of this.jointIds) {
      const parentId = sidecar.joints[id].parent;
      if (!parentId || !this.positions.has(parentId)) continue;
      linePositions.push(...this.positions.get(parentId).toArray(), ...this.positions.get(id).toArray());
    }
    this.lineGeometry = new THREE.BufferGeometry();
    this.lineGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(linePositions, 3),
    );
    this.lineMaterial = new THREE.LineBasicMaterial({
      color: NORMAL,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.72,
    });
    this.lines = new THREE.LineSegments(this.lineGeometry, this.lineMaterial);
    this.lines.renderOrder = 29;
    this.scene.add(this.lines);
    this.canvas?.addEventListener('pointerdown', this.handlePointerDown, true);
    this.canvas?.addEventListener('pointermove', this.handlePointerMove, true);
    this.canvas?.addEventListener('pointerup', this.handlePointerUp, true);
    this.canvas?.addEventListener('pointercancel', this.handlePointerUp, true);
  }

  pick(clientX, clientY) {
    if (!this.camera || !this.canvas || !this.markers.visible) return null;
    this.setRay(clientX, clientY);
    const hit = this.raycaster.intersectObject(this.markers, false)[0];
    return Number.isInteger(hit?.instanceId) ? this.jointIds[hit.instanceId] : null;
  }

  setRay(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  consume(event) {
    event.preventDefault?.();
    event.stopImmediatePropagation?.();
  }

  selectJoint(id) {
    if (this.selectedId && this.jointIndex.has(this.selectedId)) {
      this.markers.setColorAt(this.jointIndex.get(this.selectedId), NORMAL);
    }
    this.selectedId = this.jointIndex.has(id) ? id : null;
    if (this.selectedId) this.markers.setColorAt(this.jointIndex.get(this.selectedId), SELECTED);
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
  }

  jointsForBone(boneName) {
    return this.jointIds.filter((id) => this.sidecar.joints[id].sourceBone === boneName);
  }

  setJointLocalPosition(id, values) {
    const index = this.jointIndex.get(id);
    if (index == null) throw new Error(`Unknown fit joint: ${id}`);
    const position = new THREE.Vector3(...values);
    this.armatureNode.localToWorld(position);
    this.positions.set(id, position);
    MATRIX.makeTranslation(...position.toArray());
    this.markers.setMatrixAt(index, MATRIX);
    this.markers.instanceMatrix.needsUpdate = true;
    this.updateLines();
  }

  updateLines() {
    const values = this.lineGeometry.attributes.position.array;
    let offset = 0;
    for (const id of this.jointIds) {
      const parentId = this.sidecar.joints[id].parent;
      if (!parentId || !this.positions.has(parentId)) continue;
      for (const position of [this.positions.get(parentId), this.positions.get(id)]) {
        values[offset++] = position.x;
        values[offset++] = position.y;
        values[offset++] = position.z;
      }
    }
    this.lineGeometry.attributes.position.needsUpdate = true;
    this.lineGeometry.computeBoundingSphere();
  }

  setVisible(visible) {
    this.markers.visible = visible;
    this.lines.visible = visible;
  }

  setXray(xray) {
    this.markerMaterial.depthTest = !xray;
    this.markerMaterial.needsUpdate = true;
    this.lineMaterial.depthTest = !xray;
    this.lineMaterial.needsUpdate = true;
  }

  dispose() {
    this.canvas?.removeEventListener('pointerdown', this.handlePointerDown, true);
    this.canvas?.removeEventListener('pointermove', this.handlePointerMove, true);
    this.canvas?.removeEventListener('pointerup', this.handlePointerUp, true);
    this.canvas?.removeEventListener('pointercancel', this.handlePointerUp, true);
    this.markers.removeFromParent();
    this.lines.removeFromParent();
    this.markerGeometry.dispose();
    this.markerMaterial.dispose();
    this.lineGeometry.dispose();
    this.lineMaterial.dispose();
  }
}

function findArmatureNode(bone) {
  let rootBone = bone;
  while (rootBone.parent?.isBone) rootBone = rootBone.parent;
  return rootBone.parent || rootBone;
}
