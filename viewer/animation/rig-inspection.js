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

  return {
    animations: [...animations],
    bones,
    skeletons,
    skinnedMeshes,
  };
}

/** Restore every unique skeleton to the bind pose recorded in the GLB. */
export function resetRigPose(rig) {
  for (const skeleton of rig.skeletons) skeleton.pose();
}
