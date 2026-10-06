// A skeleton you can actually see: solid bones (Blender-style double pyramids) and joint
// dots, drawn on top of the model. three.js lines are always one pixel wide, which is why
// the stock SkeletonHelper disappears on a busy model.

import * as THREE from 'three';

const COLOUR = 0xffb37a;
const JOINT = 0xfd6d14;

/** A unit bone along +Y from 0 to 1: a pyramid up to a waist at 0.12, then a long one to the tip. */
function boneGeometry() {
  const w = 0.12;
  const vertices = [0, 0, 0, w, w, w, -w, w, w, -w, w, -w, w, w, -w, 0, 1, 0];
  const faces = [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(faces);
  geometry.computeVertexNormals();
  return geometry;
}

export function createBoneDisplay(bones) {
  const group = new THREE.Group();
  group.renderOrder = 999;
  const material = new THREE.MeshBasicMaterial({ color: COLOUR, depthTest: false, transparent: true, opacity: 0.9 });
  const jointMaterial = new THREE.MeshBasicMaterial({ color: JOINT, depthTest: false, transparent: true, opacity: 0.95 });
  const geometry = boneGeometry();
  const jointGeometry = new THREE.SphereGeometry(1, 10, 8);
  const set = new Set(bones);
  const links = bones.filter((bone) => set.has(bone.parent)).map((bone) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = 999;
    group.add(mesh);
    return { from: bone.parent, to: bone, mesh };
  });
  const joints = bones.map((bone) => {
    const mesh = new THREE.Mesh(jointGeometry, jointMaterial);
    mesh.renderOrder = 1000;
    group.add(mesh);
    return { bone, mesh };
  });

  const a = new THREE.Vector3(), b = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
  function update() {
    // size everything from the skeleton itself, so a dwarf and a giant both read well
    let total = 0;
    for (const link of links) {
      link.from.getWorldPosition(a);
      link.to.getWorldPosition(b);
      const length = a.distanceTo(b);
      total += length;
      link.mesh.visible = length > 1e-5;
      if (!link.mesh.visible) continue;
      dir.subVectors(b, a).divideScalar(length);
      link.mesh.position.copy(a);
      link.mesh.quaternion.setFromUnitVectors(up, dir);
      link.mesh.scale.set(length, length, length);
    }
    const jointSize = Math.max(1e-4, (total / Math.max(links.length, 1)) * 0.12);
    for (const joint of joints) {
      joint.bone.getWorldPosition(a);
      joint.mesh.position.copy(a);
      joint.mesh.scale.setScalar(jointSize);
    }
  }

  return {
    group,
    update,
    dispose() {
      group.removeFromParent();
      geometry.dispose(); jointGeometry.dispose(); material.dispose(); jointMaterial.dispose();
    },
  };
}
