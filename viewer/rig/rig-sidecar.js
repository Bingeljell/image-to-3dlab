const FINGERPRINT = /^sha256:[0-9a-f]{64}$/;
const AXES = new Set(['X', 'Y', 'Z']);

export function parseRigSidecar(input) {
  let source = input;
  if (typeof input === 'string') {
    try { source = JSON.parse(input); }
    catch (error) { throw new Error(`Rig sidecar is not valid JSON: ${error.message}`); }
  }
  requireObject(source, 'root');
  allowKeys(source, [
    'schemaVersion', 'rigProfile', 'assetFingerprint', 'coordinateSpace',
    'mirror', 'joints', 'corrections', 'binding',
  ], 'root');
  if (source.schemaVersion !== 1) fail('schemaVersion must be 1');
  if (!nonEmptyString(source.rigProfile)) fail('rigProfile must be a non-empty string');
  if (!FINGERPRINT.test(source.assetFingerprint || '')) {
    fail('assetFingerprint must be sha256 followed by 64 lowercase hexadecimal characters');
  }
  if (source.coordinateSpace !== 'armature-local') {
    fail('coordinateSpace must be "armature-local"');
  }

  requireObject(source.mirror, 'mirror');
  allowKeys(source.mirror, ['axis', 'origin'], 'mirror');
  if (!AXES.has(source.mirror.axis)) fail('mirror.axis must be X, Y, or Z');
  if (!Number.isFinite(source.mirror.origin)) fail('mirror.origin must be finite');
  requireObject(source.joints, 'joints');
  const jointEntries = Object.entries(source.joints);
  if (!jointEntries.length) fail('joints must contain at least one joint');

  const joints = {};
  for (const [id, joint] of jointEntries) {
    if (!nonEmptyString(id)) fail('joint ids must be non-empty');
    requireObject(joint, `joints.${id}`);
    allowKeys(joint, ['label', 'position', 'sourceBone', 'parent', 'mirrorOf'], `joints.${id}`);
    if (!nonEmptyString(joint.label)) fail(`joints.${id}.label must be non-empty`);
    if (!nonEmptyString(joint.sourceBone)) fail(`joints.${id}.sourceBone must be non-empty`);
    joints[id] = {
      label: joint.label,
      position: vector3(joint.position, `joints.${id}.position`),
      sourceBone: joint.sourceBone,
      parent: nullableString(joint.parent, `joints.${id}.parent`),
      mirrorOf: nullableString(joint.mirrorOf, `joints.${id}.mirrorOf`),
    };
  }
  for (const [id, joint] of Object.entries(joints)) {
    if (joint.parent && !joints[joint.parent]) fail(`joints.${id}.parent references unknown joint`);
    if (joint.mirrorOf && !joints[joint.mirrorOf]) fail(`joints.${id}.mirrorOf references unknown joint`);
  }

  const corrections = {};
  if (source.corrections != null) {
    requireObject(source.corrections, 'corrections');
    for (const [id, correction] of Object.entries(source.corrections)) {
      if (!joints[id]) fail(`corrections.${id} references unknown joint`);
      requireObject(correction, `corrections.${id}`);
      allowKeys(
        correction,
        ['sourcePosition', 'targetPosition', 'delta', 'mirrored'],
        `corrections.${id}`,
      );
      const sourcePosition = vector3(correction.sourcePosition, `corrections.${id}.sourcePosition`);
      const targetPosition = vector3(correction.targetPosition, `corrections.${id}.targetPosition`);
      const delta = vector3(correction.delta, `corrections.${id}.delta`);
      if (correction.mirrored !== true && correction.mirrored !== false) {
        fail(`corrections.${id}.mirrored must be boolean`);
      }
      const expectedDelta = targetPosition.map((value, axis) => value - sourcePosition[axis]);
      if (expectedDelta.some((value, axis) => Math.abs(value - delta[axis]) > 1e-6)) {
        fail(`corrections.${id}.delta does not match targetPosition - sourcePosition`);
      }
      corrections[id] = { sourcePosition, targetPosition, delta, mirrored: correction.mirrored };
    }
  }

  let binding = null;
  if (source.binding != null) {
    requireObject(source.binding, 'binding');
    allowKeys(
      source.binding,
      ['adapter', 'sceneFingerprint', 'metarigObjectId', 'joints'],
      'binding',
    );
    if (!nonEmptyString(source.binding.adapter)) fail('binding.adapter must be non-empty');
    if (!FINGERPRINT.test(source.binding.sceneFingerprint || '')) {
      fail('binding.sceneFingerprint must be sha256 followed by 64 lowercase hexadecimal characters');
    }
    if (!nonEmptyString(source.binding.metarigObjectId)) {
      fail('binding.metarigObjectId must be non-empty');
    }
    requireObject(source.binding.joints, 'binding.joints');
    const bindingJoints = {};
    for (const [id, value] of Object.entries(source.binding.joints)) {
      if (!joints[id]) fail(`binding.joints.${id} references unknown joint`);
      requireObject(value, `binding.joints.${id}`);
      allowKeys(value, ['targets'], `binding.joints.${id}`);
      if (!Array.isArray(value.targets) || !value.targets.length) {
        fail(`binding.joints.${id}.targets must contain at least one target`);
      }
      bindingJoints[id] = {
        targets: value.targets.map((target, index) => {
          const path = `binding.joints.${id}.targets.${index}`;
          requireObject(target, path);
          allowKeys(target, ['boneId', 'boneName', 'endpoint'], path);
          if (!nonEmptyString(target.boneId)) fail(`${path}.boneId must be non-empty`);
          if (!nonEmptyString(target.boneName)) fail(`${path}.boneName must be non-empty`);
          if (target.endpoint !== 'head' && target.endpoint !== 'tail') {
            fail(`${path}.endpoint must be head or tail`);
          }
          return { boneId: target.boneId, boneName: target.boneName, endpoint: target.endpoint };
        }),
      };
    }
    binding = {
      adapter: source.binding.adapter,
      sceneFingerprint: source.binding.sceneFingerprint,
      metarigObjectId: source.binding.metarigObjectId,
      joints: bindingJoints,
    };
  }

  return {
    schemaVersion: 1,
    rigProfile: source.rigProfile,
    assetFingerprint: source.assetFingerprint,
    coordinateSpace: 'armature-local',
    mirror: { axis: source.mirror.axis, origin: Number(source.mirror.origin) },
    joints,
    corrections,
    binding,
  };
}

export async function fingerprintAsset(fileOrBuffer, cryptoProvider = globalThis.crypto) {
  if (!cryptoProvider?.subtle) throw new Error('Web Crypto is unavailable; cannot fingerprint asset');
  const buffer = typeof fileOrBuffer?.arrayBuffer === 'function'
    ? await fileOrBuffer.arrayBuffer()
    : fileOrBuffer;
  if (!(buffer instanceof ArrayBuffer) && !ArrayBuffer.isView(buffer)) {
    throw new Error('Asset fingerprint input must be a File, ArrayBuffer, or typed array');
  }
  const bytes = ArrayBuffer.isView(buffer)
    ? buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
    : buffer;
  const digest = await cryptoProvider.subtle.digest('SHA-256', bytes);
  return 'sha256:' + [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

export function findRigSidecarFile(fileList) {
  const matches = [...fileList].filter((file) => file.name.toLowerCase().endsWith('.rig.json'));
  if (matches.length > 1) throw new Error('Choose only one .rig.json sidecar');
  return matches[0] || null;
}

function vector3(value, path) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(item))) {
    fail(`${path} must contain exactly three finite numbers`);
  }
  return value.map(Number);
}

function nullableString(value, path) {
  if (value == null) return null;
  if (!nonEmptyString(value)) fail(`${path} must be null or a non-empty string`);
  return value;
}

function nonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function requireObject(value, path) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${path} must be an object`);
}

function allowKeys(value, allowed, path) {
  const permitted = new Set(allowed);
  const unknown = Object.keys(value).filter((key) => !permitted.has(key));
  if (unknown.length) fail(`${path} contains unknown field${unknown.length === 1 ? '' : 's'}: ${unknown.join(', ')}`);
}

function fail(message) {
  throw new Error(`Rig sidecar: ${message}`);
}
