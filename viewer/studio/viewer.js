// The studio's one 3D viewer: the same controls for every asset and every step.
// Wraps the shared model viewport; owns the render loop, axis map, tools and clip bar.

import * as THREE from 'three';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { setCameraView } from '../components/camera-view-controls.js';
import { AnimationPlayer } from '../animation/player.js';
import { projectAxes, AXES } from './gizmo.js';
import { createBoneDisplay } from './bones.js';

const ICON = {
  reset: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  wire: '<svg viewBox="0 0 24 24"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/></svg>',
  bones: '<svg viewBox="0 0 24 24"><circle cx="12" cy="4" r="2"/><path d="M12 6v8M12 9l-5 3M12 9l5 3M12 14l-3 7M12 14l3 7"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4l13 8-13 8z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h3v14H7zM14 5h3v14h-3z"/></svg>',
};

export function createStudioViewer({ stage, logoUrl, onClipChange }) {
  const pane = document.createElement('div');
  pane.className = 'sv-pane';
  pane.innerHTML = '<span class="stats" hidden></span>';
  stage.appendChild(pane);

  const overlay = document.createElement('div');
  overlay.className = 'sv-overlay';
  stage.appendChild(overlay);

  const tools = document.createElement('div');
  tools.className = 'sv-tools';
  tools.innerHTML = `
    <div class="sv-gizmo" aria-label="Look from an axis"><svg viewBox="-56 -56 112 112" aria-hidden="true"><g class="lines"></g></svg></div>
    <div class="sv-toolrow">
      <button class="tool" data-act="reset" data-tip="Reset view: back to the starting camera (Home)" aria-label="Reset view">${ICON.reset}</button>
      <button class="tool" data-act="wire" aria-pressed="false" data-tip="Wireframe: see the triangles" aria-label="Wireframe">${ICON.wire}</button>
      <button class="tool" data-act="bones" aria-pressed="false" data-tip="Show skeleton: the bones the auto-rig placed" aria-label="Show skeleton">${ICON.bones}</button>
    </div>`;
  stage.appendChild(tools);

  const clipBar = document.createElement('div');
  clipBar.className = 'sv-clips';
  clipBar.hidden = true;
  stage.appendChild(clipBar);

  let view = null, player = null, skeleton = null, renderPending = false, loadToken = 0;
  let clips = [], clipIndex = -1;
  const gizmo = tools.querySelector('.sv-gizmo');
  const axisButtons = new Map(AXES.map((end) => {
    const button = document.createElement('button');
    button.className = `${end.axis}${end.key[0] === '-' ? ' neg' : ''} tip-left`;
    button.textContent = end.key[0] === '+' ? end.axis.toUpperCase() : '';
    button.dataset.tip = `Look from: ${end.label}`;
    button.setAttribute('aria-label', `Look from ${end.label}`);
    button.onclick = () => snap(end.view);
    gizmo.appendChild(button);
    return [end.key, button];
  }));

  function requestRender() {
    if (renderPending || !view) return;
    renderPending = true;
    requestAnimationFrame(renderFrame);
  }
  function renderFrame() {
    renderPending = false;
    if (!view) return;
    const moving = view.controls.update();
    skeleton?.update();
    view.renderer.render(view.scene, view.camera);
    drawGizmo();
    if (moving) requestRender();
  }
  function resize() {
    if (!view) return;
    const { clientWidth: w, clientHeight: h } = pane;
    if (!w || !h) return;
    view.renderer.setSize(w, h);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
    requestRender();
  }
  new ResizeObserver(resize).observe(pane);

  function drawGizmo() {
    const enabled = !!view?.root;
    gizmo.classList.toggle('off', !enabled);
    const direction = enabled
      ? view.camera.position.clone().sub(view.controls.target).normalize().toArray()
      : [0.5, 0.35, 0.8];
    const up = enabled ? view.camera.up.toArray() : [0, 1, 0];
    const ends = projectAxes(direction, up);
    const radius = 38;
    gizmo.querySelector('.lines').innerHTML = ends.filter((e) => e.key[0] === '+')
      .map((e) => `<line class="${e.axis}" x1="0" y1="0" x2="${e.x * radius}" y2="${e.y * radius}"/>`).join('');
    // Buttons are made once and only moved: rebuilding them every animation frame swallowed clicks.
    ends.forEach((end, order) => {
      const button = axisButtons.get(end.key);
      button.style.left = `${56 + end.x * radius}px`;
      button.style.top = `${56 + end.y * radius}px`;
      button.style.zIndex = String(order + 1);  // nearer ends on top
      button.disabled = !enabled;
    });
  }

  function snap(name) {
    if (setCameraView(view, name)) requestRender();
  }

  function setWireframe(on) {
    if (!view) return;
    for (const entry of view.materials) entry.original.wireframe = on;
    requestRender();
  }

  function setSkeleton(on) {
    skeleton?.dispose();
    skeleton = null;
    if (on && view?.rig?.bones?.length) {
      skeleton = createBoneDisplay(view.rig.bones);
      view.scene.add(skeleton.group);
      skeleton.update();
    }
    requestRender();
  }

  tools.addEventListener('click', (event) => {
    const button = event.target.closest('[data-act]');
    if (!button) return;
    const act = button.dataset.act;
    if (act === 'reset') return snap('reset');
    const on = button.getAttribute('aria-pressed') !== 'true';
    if (act === 'bones' && on && !view?.rig?.bones?.length) {
      overlayNote('No skeleton yet: rig this model first.');
      return;
    }
    button.setAttribute('aria-pressed', String(on));
    if (act === 'wire') setWireframe(on);
    if (act === 'bones') setSkeleton(on);
  });

  let noteTimer = null;
  function overlayNote(text) {
    const note = document.createElement('div');
    note.className = 'sv-note';
    note.textContent = text;
    stage.appendChild(note);
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => note.remove(), 2400);
  }

  function setOverlay(html) {
    overlay.innerHTML = html;
    overlay.hidden = !html;
  }

  function dispose() {
    player?.dispose?.();
    player = null;
    skeleton?.dispose();
    skeleton = null;
    if (view) {
      disposeModelViewport(view);
      view.renderer.domElement.remove();
      view = null;
    }
  }

  function load(url, { label = '', autoplay = false } = {}) {
    const token = ++loadToken;
    dispose();
    setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>Loading model…</span></div>`);
    const created = createModelViewport({
      pane,
      spec: { url, ext: 'glb', label },
      onChange: requestRender,
      onLoaded: (loaded) => {
        if (token !== loadToken) return;
        setOverlay('');
        setCameraView(loaded, 'reset');
        if (loaded.rig.animations.length) {
          player = new AnimationPlayer(new THREE.AnimationMixer(loaded.modelRoot), {
            onFrame: () => requestRender(),
            onStateChange: (playing) => syncPlayButton(playing),
          });
          player.select(loaded.rig.animations[0]);
          player.action.setLoop(THREE.LoopRepeat, Infinity);
          if (autoplay) player.play();
        }
        // keep wireframe and skeleton on across models and moves
        const pressed = (act) => tools.querySelector(`[data-act="${act}"]`).getAttribute('aria-pressed') === 'true';
        if (pressed('wire')) setWireframe(true);
        if (pressed('bones')) {
          if (loaded.rig.bones.length) setSkeleton(true);
          else tools.querySelector('[data-act="bones"]').setAttribute('aria-pressed', 'false');
        }
        resize();
        drawGizmo();
      },
      onError: (error) => {
        if (token !== loadToken) return;
        setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>This model didn't load: ${String(error?.message || error).slice(0, 140)}</span></div>`);
      },
    });
    view = created;
    resize();
  }

  function showImage(url, caption) {
    ++loadToken;
    dispose();
    setOverlay(`<figure class="sv-picture"><img src="${url}" alt=""><figcaption>${caption}</figcaption></figure>`);
    drawGizmo();
    clipBar.hidden = true;
  }

  function showEmpty(text) {
    ++loadToken;
    dispose();
    setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>${text}</span></div>`);
    drawGizmo();
    clipBar.hidden = true;
  }

  // ------------------------------------------------------------------ clips
  function syncPlayButton(playing) {
    const button = clipBar.querySelector('[data-clip-act="play"]');
    if (button) {
      button.innerHTML = playing ? ICON.pause : ICON.play;
      button.dataset.tip = playing ? 'Pause' : 'Play';
    }
  }

  /** Show a row of moves (or, with `still`, of props). `items` are {name, url}. */
  function setClips(items, { restUrl = null, index = 0, still = false } = {}) {
    clips = items;
    clipBar.hidden = !items.length && !restUrl;
    if (clipBar.hidden) { clipBar.innerHTML = ''; return; }
    clipIndex = items.length ? index : -1;
    clipBar.innerHTML = (still ? '' : `
      <button class="tool" data-clip-act="play" data-tip="Play or pause" aria-label="Play or pause">${ICON.pause}</button>
      <button class="ghost" data-clip-act="rest" data-tip="Reset pose: stop and show the character in its rest pose">Reset pose</button>
      <span class="sep"></span>`) + `
      ${items.map((clip, i) => `<button class="chip" data-clip="${i}" aria-pressed="${i === clipIndex}">${clip.name}</button>`).join('')}`;
    if (clipIndex >= 0) onClipChange?.(items[clipIndex]);
    clipBar.dataset.rest = restUrl || '';
    if (clipIndex >= 0) load(items[clipIndex].url, { label: items[clipIndex].name, autoplay: true });
    else if (restUrl) load(restUrl, { label: 'rest pose' });
  }

  clipBar.addEventListener('click', (event) => {
    const chip = event.target.closest('[data-clip]');
    if (chip) {
      clipIndex = Number(chip.dataset.clip);
      clipBar.querySelectorAll('[data-clip]').forEach((b) => b.setAttribute('aria-pressed', String(b === chip)));
      load(clips[clipIndex].url, { label: clips[clipIndex].name, autoplay: true });
      onClipChange?.(clips[clipIndex]);
      return;
    }
    const act = event.target.closest('[data-clip-act]')?.dataset.clipAct;
    if (act === 'play' && player) player.playing ? player.pause() : player.play();
    if (act === 'rest') {
      player?.pause();
      view?.resetPose();
      requestRender();
      syncPlayButton(false);
    }
  });

  window.addEventListener('keydown', (event) => {
    if (event.target.closest?.('input, textarea, select, [contenteditable]')) return;
    const keys = { 1: 'front', 3: 'right', 7: 'top', Home: 'reset' };
    if (keys[event.key]) snap(keys[event.key]);
  });

  drawGizmo();
  return { load, showImage, showEmpty, setClips, snap, dispose };
}
