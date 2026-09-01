/**
 * Small timeline controller around a Three.js AnimationMixer.
 *
 * The mixer is injected so timing and transport behavior stay independently testable and
 * future animation rooms can reuse it without owning another render loop.
 */
export class AnimationPlayer {
  constructor(mixer, options = {}) {
    this.mixer = mixer;
    this.onFrame = options.onFrame || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
    this.requestFrame = options.requestFrame || ((callback) => requestAnimationFrame(callback));
    this.cancelFrame = options.cancelFrame || ((handle) => cancelAnimationFrame(handle));
    this.action = null;
    this.clip = null;
    this.playing = false;
    this.frameHandle = null;
    this.lastTimestamp = null;
    this.tick = this.tick.bind(this);

    if (this.mixer.addEventListener) {
      this.mixer.addEventListener('finished', () => this.pause());
    }
  }

  select(clip) {
    this.clear();
    this.clip = clip;
    this.action = this.mixer.clipAction(clip);
    this.action.reset().play();
    this.action.paused = true;
    this.mixer.update(0);
    this.onFrame(this.time, this.duration);
    this.onStateChange(this.playing);
    return this.action;
  }

  clear() {
    this.pause(false);
    this.mixer.stopAllAction();
    this.action = null;
    this.clip = null;
    this.onFrame(0, 0);
    this.onStateChange(false);
  }

  play() {
    if (!this.action || this.playing) return false;
    this.action.paused = false;
    this.playing = true;
    this.lastTimestamp = null;
    this.frameHandle = this.requestFrame(this.tick);
    this.onStateChange(true);
    return true;
  }

  pause(notify = true) {
    if (this.frameHandle != null) this.cancelFrame(this.frameHandle);
    this.frameHandle = null;
    this.lastTimestamp = null;
    if (this.action) this.action.paused = true;
    this.playing = false;
    if (notify) this.onStateChange(false);
  }

  toggle() {
    return this.playing ? (this.pause(), false) : this.play();
  }

  seek(seconds) {
    if (!this.action) return;
    const value = Math.max(0, Math.min(this.duration, Number(seconds) || 0));
    this.action.time = value;
    this.mixer.update(0);
    this.onFrame(this.time, this.duration);
  }

  tick(timestamp) {
    if (!this.playing || !this.action) return;
    const delta = this.lastTimestamp == null ? 0 : Math.max(0, (timestamp - this.lastTimestamp) / 1000);
    this.lastTimestamp = timestamp;
    this.mixer.update(delta);
    this.onFrame(this.time, this.duration);
    if (this.playing) this.frameHandle = this.requestFrame(this.tick);
  }

  dispose() {
    this.clear();
    if (this.mixer.uncacheRoot) this.mixer.uncacheRoot(this.mixer.getRoot());
  }

  get time() {
    return this.action ? this.action.time : 0;
  }

  get duration() {
    return this.clip ? this.clip.duration : 0;
  }
}

