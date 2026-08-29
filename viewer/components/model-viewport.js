import * as THREE from 'three';
import { GLTFLoader } from '../vendor/loaders/GLTFLoader.js';
import { OrbitControls } from '../vendor/controls/OrbitControls.js';
import { IndexedOBJLoader } from '../IndexedOBJLoader.js';
import { RoomEnvironment } from '../vendor/environments/RoomEnvironment.js';

// Pure geometry shared by every viewport. Each renderer still creates its own PMREM
// texture because GPU resources cannot be shared across WebGL contexts.
const roomEnvironment = new RoomEnvironment();

/**
 * Create and begin loading one reusable model viewport.
 *
 * Modes own the surrounding pane chrome and decide what happens after a model loads. The
 * component owns the Three.js scene, camera, renderer, controls, model normalization, and
 * resource lifecycle.
 */
export function createModelViewport({ pane, spec, slotIndex = null, onChange, onLoaded, onError }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  pane.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161a);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(0, 0.3, 3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);
  if (onChange) controls.addEventListener('change', onChange);

  // Studio HDRI environment, matching Blender's Material Preview/LookDev lighting.
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(roomEnvironment, 0.04).texture;
  pmremGenerator.dispose();
  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x30302a, 0.15));

  const view = {
    kind: 'model', spec, slotIndex, pane, renderer, scene, camera, controls,
    root: null, stats: pane.querySelector('.stats'), materials: [], overlay: null,
  };
  loadModel(view, onLoaded, onError);
  return view;
}

export function disposeModelViewport(view) {
  view.controls.dispose();
  view.renderer.dispose();
  view.renderer.forceContextLoss();
}

function loadModel(view, onLoaded, onError) {
  const { spec } = view;
  const loaded = (root) => {
    // Normalize to a unit box centered on the origin so differently scaled exports remain
    // directly comparable and every room can use the same camera framing contract.
    const pivot = new THREE.Group();
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const scale = 1 / Math.max(size.x, size.y, size.z);
    root.position.sub(centre);
    pivot.scale.setScalar(scale);
    pivot.add(root);
    pivot.updateMatrixWorld(true);

    let faces = 0;
    let verts = 0;
    root.traverse((object) => {
      if (!object.isMesh) return;
      const geometry = object.geometry;
      // Shape-stage exports may omit NORMAL. Inspection materials and wireframe need a
      // real attribute even when the original flat-shaded material appears acceptable.
      if (!geometry.attributes.normal) geometry.computeVertexNormals();
      verts += geometry.attributes.position.count;
      faces += geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        view.materials.push({
          mesh: object,
          original: material,
          flat: null,
          normal: null,
          origFlatShading: !!material.flatShading,
        });
      }
    });

    view.root = pivot;
    view.scene.add(pivot);
    view.stats.textContent = `${faces.toLocaleString()} faces\n${verts.toLocaleString()} verts (as stored)`;
    if (onLoaded) onLoaded(view);
  };
  const progress = (event) => {
    if (!event.lengthComputable) return;
    view.stats.textContent = `loading… ${Math.round(event.loaded / event.total * 100)}%`;
  };
  const failed = (error) => {
    view.stats.textContent = 'FAILED';
    if (onError) onError(error);
  };

  if (spec.ext === 'obj') {
    new IndexedOBJLoader().load(spec.url, (root) => {
      // Match o_voxel's native TRELLIS -> GLB coordinate transform for raw OBJ inputs.
      root.rotation.x = -Math.PI / 2;
      loaded(root);
    }, progress, failed);
    return;
  }

  const loader = spec.manager ? new GLTFLoader(spec.manager) : new GLTFLoader();
  loader.load(spec.url, (gltf) => loaded(gltf.scene), progress, failed);
}

