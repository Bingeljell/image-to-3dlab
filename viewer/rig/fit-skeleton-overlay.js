import * as THREE from '../vendor/three.module.js';

const JOINT_COLOR = new THREE.Color(0xf0a95b);
const BONE_COLOR = new THREE.Color(0xa6afbb);
const SELECTED = new THREE.Color(0x35e7ff);
const MATRIX = new THREE.Matrix4();
const ORIENTATION = new THREE.Quaternion();
const SCALE = new THREE.Vector3();
const MIDPOINT = new THREE.Vector3();
const DIRECTION = new THREE.Vector3();
const AXIS_UP = new THREE.Vector3(0, 1, 0);
const AXIS_VECTORS = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};
const AXIS_COLORS = { x: 0xf05d5e, y: 0x63c174, z: 0x579dff };

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
    this.dragStartPoint = new THREE.Vector3();
    this.dragStartPosition = new THREE.Vector3();
    this.dragAxisWorld = new THREE.Vector3();
    this.dragJointId = null;
    this.dragAxis = null;
    this.jointsVisible = true;
    this.handlePointerDown = (event) => {
      if (event.button !== 0) return;
      const axis = this.pickAxis(event.clientX, event.clientY);
      if (axis && this.selectedId) {
        this.consume(event);
        this.beginDrag(event, this.selectedId, axis);
        return;
      }
      const id = this.pick(event.clientX, event.clientY);
      if (id) {
        this.consume(event);
        this.onSelect?.(id);
        this.beginDrag(event, id, null);
        return;
      }
      const boneJointId = this.pickBone(event.clientX, event.clientY);
      if (!boneJointId) return;
      this.consume(event);
      this.onSelect?.(boneJointId);
    };
    this.handlePointerMove = (event) => {
      if (!this.dragJointId) return;
      this.consume(event);
      this.setRay(event.clientX, event.clientY);
      if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) return;
      if (this.dragAxis) {
        const distance = this.dragPoint.clone().sub(this.dragStartPoint).dot(this.dragAxisWorld);
        this.dragTarget.copy(this.dragStartPosition).addScaledVector(this.dragAxisWorld, distance);
      } else {
        this.dragTarget.copy(this.dragPoint).add(this.dragOffset);
      }
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
      this.dragAxis = null;
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
      toneMapped: false,
    });
    this.markers = new THREE.InstancedMesh(
      this.markerGeometry,
      this.markerMaterial,
      this.jointIds.length,
    );
    this.markers.frustumCulled = false;
    this.markers.renderOrder = 30;
    this.jointIds.forEach((id, index) => {
      this.updateMarkerMatrix(id);
      this.markers.setColorAt(index, JOINT_COLOR);
    });
    this.markers.instanceMatrix.needsUpdate = true;
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    this.scene.add(this.markers);

    this.segments = [];
    for (const id of this.jointIds) {
      const parentId = sidecar.joints[id].parent;
      if (!parentId || !this.positions.has(parentId)) continue;
      this.segments.push({ parentId, childId: id });
    }
    this.boneGeometry = new THREE.ConeGeometry(markerRadius * 0.7, 1, 7, 1);
    this.boneMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 0.86,
      toneMapped: false,
    });
    this.boneBodies = new THREE.InstancedMesh(
      this.boneGeometry,
      this.boneMaterial,
      this.segments.length,
    );
    this.boneBodies.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.boneBodies.frustumCulled = false;
    this.boneBodies.renderOrder = 29;
    this.updateBoneBodies();
    this.scene.add(this.boneBodies);

    this.gizmo = new THREE.Group();
    this.gizmo.name = 'Fit joint translation gizmo';
    this.gizmo.visible = false;
    this.gizmoGeometry = new THREE.CylinderGeometry(0.007, 0.007, 0.14, 8);
    this.gizmoTipGeometry = new THREE.ConeGeometry(0.016, 0.04, 10);
    this.gizmoMaterials = [];
    for (const [axis, direction] of Object.entries(AXIS_VECTORS)) {
      const material = new THREE.MeshBasicMaterial({
        color: AXIS_COLORS[axis], depthTest: false, depthWrite: false,
        transparent: true, opacity: 0.96, toneMapped: false,
      });
      this.gizmoMaterials.push(material);
      const orientation = new THREE.Quaternion().setFromUnitVectors(AXIS_UP, direction);
      const shaft = new THREE.Mesh(this.gizmoGeometry, material);
      shaft.position.copy(direction).multiplyScalar(0.07);
      shaft.quaternion.copy(orientation);
      shaft.userData.fitAxis = axis;
      shaft.renderOrder = 34;
      const tip = new THREE.Mesh(this.gizmoTipGeometry, material);
      tip.position.copy(direction).multiplyScalar(0.16);
      tip.quaternion.copy(orientation);
      tip.userData.fitAxis = axis;
      tip.renderOrder = 34;
      this.gizmo.add(shaft, tip);
    }
    this.scene.add(this.gizmo);
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

  pickAxis(clientX, clientY) {
    if (!this.camera || !this.canvas || !this.gizmo.visible) return null;
    this.setRay(clientX, clientY);
    return this.raycaster.intersectObjects(this.gizmo.children, false)[0]?.object?.userData?.fitAxis
      || null;
  }

  pickBone(clientX, clientY) {
    if (!this.camera || !this.canvas || !this.boneBodies.visible) return null;
    this.setRay(clientX, clientY);
    const hit = this.raycaster.intersectObject(this.boneBodies, false)[0];
    return Number.isInteger(hit?.instanceId) ? this.segments[hit.instanceId].childId : null;
  }

  beginDrag(event, id, axis) {
    this.dragJointId = id;
    this.dragAxis = axis;
    const position = this.positions.get(id);
    this.setRay(event.clientX, event.clientY);
    if (axis) {
      this.dragStartPosition.copy(position);
      this.dragAxisWorld.copy(AXIS_VECTORS[axis]).transformDirection(this.armatureNode.matrixWorld);
      this.camera.getWorldDirection(this.dragTarget);
      const normal = this.dragAxisWorld.clone().cross(this.dragTarget).cross(this.dragAxisWorld);
      if (normal.lengthSq() < 1e-8) normal.copy(this.camera.up);
      this.dragPlane.setFromNormalAndCoplanarPoint(normal.normalize(), position);
      if (!this.raycaster.ray.intersectPlane(this.dragPlane, this.dragStartPoint)) {
        this.dragStartPoint.copy(position);
      }
    } else {
      this.camera.getWorldDirection(this.dragTarget);
      this.dragPlane.setFromNormalAndCoplanarPoint(this.dragTarget, position);
      this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint);
      this.dragOffset.subVectors(position, this.dragPoint);
    }
    this.canvas.setPointerCapture?.(event.pointerId);
    this.canvas.style.cursor = 'grabbing';
    this.onDragChange?.(true);
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
    const previousId = this.selectedId;
    this.selectedId = this.jointIndex.has(id) ? id : null;
    if (previousId && this.jointIndex.has(previousId)) {
      this.markers.setColorAt(this.jointIndex.get(previousId), JOINT_COLOR);
      this.updateMarkerMatrix(previousId);
    }
    if (this.selectedId) {
      this.markers.setColorAt(this.jointIndex.get(this.selectedId), SELECTED);
      this.updateMarkerMatrix(this.selectedId);
    }
    this.updateGizmo();
    this.updateBoneColors();
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    this.markers.instanceMatrix.needsUpdate = true;
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
    this.updateMarkerMatrix(id);
    if (id === this.selectedId) this.updateGizmo();
    this.markers.instanceMatrix.needsUpdate = true;
    this.updateBoneBodies();
  }

  updateBoneBodies() {
    this.segments.forEach(({ parentId, childId }, index) => {
      const parent = this.positions.get(parentId);
      const child = this.positions.get(childId);
      DIRECTION.subVectors(child, parent);
      const length = DIRECTION.length();
      MIDPOINT.addVectors(parent, child).multiplyScalar(0.5);
      if (length > 1e-8) ORIENTATION.setFromUnitVectors(AXIS_UP, DIRECTION.normalize());
      else ORIENTATION.identity();
      const selected = childId === this.selectedId;
      SCALE.set(selected ? 1.45 : 1, length, selected ? 1.45 : 1);
      MATRIX.compose(MIDPOINT, ORIENTATION, SCALE);
      this.boneBodies.setMatrixAt(index, MATRIX);
    });
    this.boneBodies.instanceMatrix.needsUpdate = true;
    this.updateBoneColors();
  }

  updateBoneColors() {
    this.segments.forEach(({ childId }, index) => {
      this.boneBodies.setColorAt(index, childId === this.selectedId ? SELECTED : BONE_COLOR);
    });
    if (this.boneBodies.instanceColor) {
      this.boneBodies.instanceColor.needsUpdate = true;
    }
  }

  setVisible(visible) {
    this.setJointsVisible(visible);
    this.setBonesVisible(visible);
  }

  setJointsVisible(visible) {
    this.jointsVisible = visible;
    this.markers.visible = visible;
    this.updateGizmo();
  }

  setBonesVisible(visible) {
    this.boneBodies.visible = visible;
  }

  updateMarkerMatrix(id) {
    const index = this.jointIndex.get(id);
    const scale = id === this.selectedId ? 1.45 : 1;
    SCALE.setScalar(scale);
    MATRIX.compose(this.positions.get(id), ORIENTATION.identity(), SCALE);
    this.markers.setMatrixAt(index, MATRIX);
  }

  updateGizmo() {
    this.gizmo.visible = !!this.selectedId && this.jointsVisible;
    if (this.gizmo.visible) this.gizmo.position.copy(this.positions.get(this.selectedId));
  }

  setXray(xray) {
    this.markerMaterial.depthTest = !xray;
    this.markerMaterial.needsUpdate = true;
    this.boneMaterial.depthTest = !xray;
    this.boneMaterial.needsUpdate = true;
    for (const material of this.gizmoMaterials) {
      material.depthTest = !xray;
      material.needsUpdate = true;
    }
  }

  dispose() {
    this.canvas?.removeEventListener('pointerdown', this.handlePointerDown, true);
    this.canvas?.removeEventListener('pointermove', this.handlePointerMove, true);
    this.canvas?.removeEventListener('pointerup', this.handlePointerUp, true);
    this.canvas?.removeEventListener('pointercancel', this.handlePointerUp, true);
    this.markers.removeFromParent();
    this.boneBodies.removeFromParent();
    this.gizmo.removeFromParent();
    this.markerGeometry.dispose();
    this.markerMaterial.dispose();
    this.boneGeometry.dispose();
    this.boneMaterial.dispose();
    this.gizmoGeometry.dispose();
    this.gizmoTipGeometry.dispose();
    this.gizmoMaterials.forEach((material) => material.dispose());
  }
}

function findArmatureNode(bone) {
  let rootBone = bone;
  while (rootBone.parent?.isBone) rootBone = rootBone.parent;
  return rootBone.parent || rootBone;
}
