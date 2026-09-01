/** Collect the runtime rig data that glTF preserves without depending on Three.js types. */
export function inspectRig(root, animations = []) {
  const bones = [];
  const skeletons = [];
  const skinnedMeshes = [];
  const seenBones = new Set();
  const seenSkeletons = new Set();

  root?.traverse((object) => {
    if (object.isBone && !seenBones.has(object)) {
      seenBones.add(object);
      bones.push(object);
    }
    if (!object.isSkinnedMesh) return;
    skinnedMeshes.push(object);
    if (object.skeleton && !seenSkeletons.has(object.skeleton)) {
      seenSkeletons.add(object.skeleton);
      skeletons.push(object.skeleton);
    }
    for (const bone of object.skeleton?.bones || []) {
      if (seenBones.has(bone)) continue;
      seenBones.add(bone);
      bones.push(bone);
    }
  });

  const bindPose = new Map(bones.map((bone) => [bone, snapshotTransform(bone)]));

  return {
    animations: [...animations],
    bones,
    skeletons,
    skinnedMeshes,
    bindPose,
  };
}

/** Restore every unique skeleton to the bind pose recorded in the GLB. */
export function resetRigPose(rig) {
  for (const skeleton of rig.skeletons) skeleton.pose();
}

/** Return stable hierarchy data plus immutable bind and current local transforms. */
export function describeBone(bone, rig) {
  if (!bone) return null;
  return {
    name: bone.name || '(unnamed bone)',
    parent: bone.parent?.isBone ? (bone.parent.name || '(unnamed bone)') : null,
    children: (bone.children || [])
      .filter((child) => child.isBone)
      .map((child) => child.name || '(unnamed bone)'),
    bind: rig.bindPose.get(bone) || snapshotTransform(bone),
    pose: snapshotTransform(bone),
  };
}

function snapshotTransform(object) {
  return {
    position: readComponents(object.position, ['x', 'y', 'z'], [0, 0, 0]),
    quaternion: readComponents(object.quaternion, ['x', 'y', 'z', 'w'], [0, 0, 0, 1]),
    scale: readComponents(object.scale, ['x', 'y', 'z'], [1, 1, 1]),
  };
}

function readComponents(value, keys, fallback) {
  if (!value) return [...fallback];
  if (typeof value.toArray === 'function') return value.toArray().slice(0, keys.length);
  return keys.map((key, index) => Number(value[key] ?? fallback[index]));
}
