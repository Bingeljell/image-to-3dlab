// How the model is drawn: its own paint, grey clay (shape only) or normals (surface direction
// as colour, so dents, flipped faces and lumps show up). No three.js import here: the look
// materials are handed in, so this runs (and is tested) without a browser.

export const LOOKS = ['paint', 'clay', 'normal'];

/** Put every mesh in `mode`. The mesh's own material is kept on first use and put back for
 *  'paint'; a multi-material mesh gets the look in every slot so its groups stay valid. */
export function applyLook(meshes, mode, looks) {
  const look = mode === 'paint' ? null : looks[mode];
  for (const mesh of meshes) {
    if (!('paint' in mesh.userData)) mesh.userData.paint = mesh.material;
    const paint = mesh.userData.paint;
    mesh.material = !look ? paint : Array.isArray(paint) ? paint.map(() => look) : look;
  }
}

/** Each mesh once, from the viewport's per-material list. */
export const meshesOf = (entries) => [...new Set(entries.map((entry) => entry.mesh))];

/** "123,456 faces" from the viewport's stats text (which also carries a verts line). */
export function faceLabel(statsText) {
  const match = /^([\d,.\s]+) faces/.exec(statsText || '');
  return match ? `${match[1].trim()} faces` : '';
}
