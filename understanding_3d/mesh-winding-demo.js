// A classic entry point allows opening the HTML directly from disk. The import map still
// supplies the same Three.js modules from the CDN.
(async () => {
  const [THREE, { OrbitControls }, { CSS2DRenderer, CSS2DObject }] = await Promise.all([
    import('three'),
    import('three/addons/controls/OrbitControls.js'),
    import('three/addons/renderers/CSS2DRenderer.js'),
  ]);

  // ---------- Scene setup ----------
  const container = document.getElementById('canvas-container');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1115);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(3.2, 2.4, 4.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(5, 8, 4);
  scene.add(dir);

  // Grid helper (subtle)
  const grid = new THREE.GridHelper(8, 16, 0x2a2f3a, 0x1c212b);
  grid.position.y = -1.01;
  scene.add(grid);

  // ---------- Cube data ----------
  // Standard unit cube vertices (centered)
  const positions = new Float32Array([
    // 0
    -1, -1,  1,
    // 1
     1, -1,  1,
    // 2
     1,  1,  1,
    // 3
    -1,  1,  1,
    // 4
    -1, -1, -1,
    // 5
     1, -1, -1,
    // 6
     1,  1, -1,
    // 7
    -1,  1, -1
  ]);

  // Indices for a cube (each face two triangles).
  // Front face (z+) is triangles 0 and 1 — we will flip triangle 0 for the demo.
  // Front: 0,1,2 and 0,2,3   (both CCW when looking from +Z)
  let indices = new Uint16Array([
    // Front (z+)
    0, 1, 2,   // triangle 0  ← this is the one we flip
    0, 2, 3,   // triangle 1
    // Right (x+)
    1, 5, 6,
    1, 6, 2,
    // Back (z-)
    5, 4, 7,
    5, 7, 6,
    // Left (x-)
    4, 0, 3,
    4, 3, 7,
    // Top (y+)
    3, 2, 6,
    3, 6, 7,
    // Bottom (y-)
    4, 5, 1,
    4, 1, 0
  ]);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();

  // Materials
  const solidMat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    metalness: 0.15,
    roughness: 0.55,
    side: THREE.DoubleSide,
    flatShading: true
  });

  // Face orientation material (we will switch to this)
  const orientationMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.85
  });

  const mesh = new THREE.Mesh(geometry, solidMat);
  scene.add(mesh);

  // Wireframe
  const wireGeo = new THREE.WireframeGeometry(geometry);
  const wire = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.6 }));
  scene.add(wire);

  // ---------- Vertex spheres + labels ----------
  const vertexGroup = new THREE.Group();
  scene.add(vertexGroup);
  const labelGroup = new THREE.Group();
  scene.add(labelGroup);

  const sphereGeo = new THREE.SphereGeometry(0.07, 16, 16);
  const sphereMat = new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0xf97316, emissiveIntensity: 0.25 });

  const vertexPositions = [];
  for (let i = 0; i < 8; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    vertexPositions.push(new THREE.Vector3(x, y, z));

    const sphere = new THREE.Mesh(sphereGeo, sphereMat);
    sphere.position.set(x, y, z);
    vertexGroup.add(sphere);

    // CSS2D label
    const div = document.createElement('div');
    div.textContent = i;
    div.style.color = '#fff';
    div.style.fontSize = '13px';
    div.style.fontWeight = '700';
    div.style.background = 'rgba(0,0,0,0.65)';
    div.style.padding = '2px 6px';
    div.style.borderRadius = '6px';
    div.style.border = '1px solid #f97316';
    const label = new CSS2DObject(div);
    label.position.set(x, y + 0.18, z);
    labelGroup.add(label);
  }

  // ---------- Face normals (ArrowHelpers) ----------
  const normalGroup = new THREE.Group();
  scene.add(normalGroup);

  function getFaceCenterAndNormal(i0, i1, i2) {
    const v0 = vertexPositions[i0];
    const v1 = vertexPositions[i1];
    const v2 = vertexPositions[i2];
    const center = new THREE.Vector3().add(v0).add(v1).add(v2).divideScalar(3);
    const edge1 = new THREE.Vector3().subVectors(v1, v0);
    const edge2 = new THREE.Vector3().subVectors(v2, v0);
    const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();
    return { center, normal };
  }

  function rebuildNormals() {
    // clear old
    while (normalGroup.children.length) {
      const obj = normalGroup.children[0];
      normalGroup.remove(obj);
      if (obj.dispose) obj.dispose();
    }

    const idx = geometry.index.array;
    for (let i = 0; i < idx.length; i += 3) {
      const i0 = idx[i], i1 = idx[i+1], i2 = idx[i+2];
      const { center, normal } = getFaceCenterAndNormal(i0, i1, i2);
      const arrow = new THREE.ArrowHelper(normal, center, 0.55, 0x22d3ee, 0.12, 0.08);
      normalGroup.add(arrow);
    }
  }
  rebuildNormals();

  // ---------- Face orientation colors ----------
  function updateOrientationColors() {
    const colors = [];
    const idx = geometry.index.array;
    const pos = geometry.attributes.position;

    // We color per-vertex for the two triangles of the front face specially,
    // but for simplicity we recompute a face color based on normal direction relative to +Z for demo.
    // Better: use a simple approach - create vertex colors based on whether the face normal points "out".

    // For educational clarity we only strongly color the front face triangles.
    // All other faces stay neutral blue-ish.
    const frontColor = new THREE.Color(0x3b82f6); // blue = front
    const backColor  = new THREE.Color(0xef4444); // red = back
    const otherColor = new THREE.Color(0x64748b);

    // We need per-vertex colors. Because vertices are shared we will slightly abuse and
    // just set colors for the first 6 indices (front two triangles) based on their current winding.
    // For a clean demo we recompute everything with non-indexed temporarily?
    // Simpler approach for this teaching tool:

    geometry.deleteAttribute('color');
    const colorAttr = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3);
    geometry.setAttribute('color', colorAttr);

    // Default all vertices to otherColor
    for (let i = 0; i < pos.count; i++) {
      colorAttr.setXYZ(i, otherColor.r, otherColor.g, otherColor.b);
    }

    // Check the two front triangles (indices 0-5)
    for (let t = 0; t < 2; t++) {
      const base = t * 3;
      const i0 = idx[base], i1 = idx[base+1], i2 = idx[base+2];
      const { normal } = getFaceCenterAndNormal(i0, i1, i2);
      // Front face should have normal roughly +Z
      const isFront = normal.z > 0.3;
      const col = isFront ? frontColor : backColor;
      colorAttr.setXYZ(i0, col.r, col.g, col.b);
      colorAttr.setXYZ(i1, col.r, col.g, col.b);
      colorAttr.setXYZ(i2, col.r, col.g, col.b);
    }
    colorAttr.needsUpdate = true;
  }

  // ---------- State ----------
  let showNormals = true;
  let showOrientation = false;
  let showWire = true;
  let showLabels = true;
  let flipped = false;

  normalGroup.visible = showNormals;
  labelGroup.visible = showLabels;
  wire.visible = showWire;

  // ---------- UI handlers ----------
  document.getElementById('btn-normals').addEventListener('click', (e) => {
    showNormals = !showNormals;
    normalGroup.visible = showNormals;
    e.target.classList.toggle('active', showNormals);
  });

  document.getElementById('btn-orientation').addEventListener('click', (e) => {
    showOrientation = !showOrientation;
    if (showOrientation) {
      updateOrientationColors();
      mesh.material = orientationMat;
    } else {
      mesh.material = solidMat;
    }
    e.target.classList.toggle('active', showOrientation);
  });

  document.getElementById('btn-wire').addEventListener('click', (e) => {
    showWire = !showWire;
    wire.visible = showWire;
    e.target.classList.toggle('active', showWire);
  });

  document.getElementById('btn-labels').addEventListener('click', (e) => {
    showLabels = !showLabels;
    labelGroup.visible = showLabels;
    e.target.classList.toggle('active', showLabels);
  });

  document.getElementById('btn-flip').addEventListener('click', () => {
    // Flip the first triangle: reverse its three indices
    const idx = geometry.index.array;
    // Currently: 0,1,2  → become 0,2,1
    const a = idx[0], b = idx[1], c = idx[2];
    idx[0] = a;
    idx[1] = c;
    idx[2] = b;
    geometry.index.needsUpdate = true;
    geometry.computeVertexNormals();

    // Update wireframe
    scene.remove(wire);
    const newWireGeo = new THREE.WireframeGeometry(geometry);
    wire.geometry.dispose();
    wire.geometry = newWireGeo;
    scene.add(wire);

    rebuildNormals();
    if (showOrientation) updateOrientationColors();

    flipped = !flipped;
    const status = document.getElementById('status');
    const indexDisplay = document.getElementById('index-display');

    if (flipped) {
      indexDisplay.textContent = '0, 2, 1';
      status.textContent = 'Current: Clockwise (back / inverted normal)';
      status.classList.add('flipped');
    } else {
      indexDisplay.textContent = '0, 1, 2';
      status.textContent = 'Current: Counter-clockwise (front / outward normal)';
      status.classList.remove('flipped');
    }
  });

  // Initial button states
  document.getElementById('btn-normals').classList.add('active');
  document.getElementById('btn-labels').classList.add('active');
  document.getElementById('btn-wire').classList.add('active');

  // ---------- Resize & animate ----------
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
  });

  function animate() {
    window.requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  animate();
})();
