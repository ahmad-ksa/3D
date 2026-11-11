// Studio3D model viewer: <s3d-viewer src="model.glb" info='{"quads":123}'></s3d-viewer>
// Modes: colours / clay / wireframe / x-ray, lighting on/off, skeleton, auto-rotate, stats.
import * as THREE from 'three';
import { GLTFLoader } from '/s3d-static/three/GLTFLoader.js';
import { OrbitControls } from '/s3d-static/three/OrbitControls.js';

const CSS = `
:host{display:block;position:relative;width:100%;height:var(--s3d-h,380px);border-radius:10px;overflow:hidden;
      background:radial-gradient(circle at 50% 40%,#3a3d46,#1d1f25);font-family:'Segoe UI',Tahoma,sans-serif}
canvas{display:block;width:100%;height:100%;outline:none}
.bar{position:absolute;top:8px;left:50%;transform:translateX(-50%);display:flex;gap:4px;padding:4px;
     background:rgba(20,22,28,.72);border-radius:10px;backdrop-filter:blur(6px);direction:rtl;flex-wrap:wrap;
     justify-content:center;max-width:96%}
button{all:unset;cursor:pointer;color:#cfd3dc;font-size:12px;padding:4px 7px;border-radius:7px;white-space:nowrap}
button:hover{background:rgba(255,255,255,.08)}
button.on{background:#5b5bd6;color:#fff}
.sep{width:1px;background:rgba(255,255,255,.15);margin:2px 3px}
.stats{position:absolute;bottom:8px;right:10px;color:#e6e8ee;font-size:12px;direction:rtl;
       background:rgba(20,22,28,.65);padding:5px 10px;border-radius:8px;font-variant-numeric:tabular-nums}
.msg{position:absolute;inset:0;pointer-events:none;display:flex;align-items:center;justify-content:center;color:#9aa0ad;font-size:14px}
`;

const fmt = n => n.toLocaleString('en-US');

class S3DViewer extends HTMLElement {
  static get observedAttributes() { return ['src']; }

  connectedCallback() {
    if (this._init) return;
    this._init = true;
    const root = this.attachShadow({ mode: 'open' });
    root.innerHTML = `<style>${CSS}</style>
      <div class="bar">
        <button data-mode="color" class="on" title="بالألوان">🎨 ألوان</button>
        <button data-mode="clay" title="بدون ألوان">⚪ بدون ألوان</button>
        <button data-mode="wire" title="عرض الأضلاع">🕸️ أضلاع</button>
        <button data-mode="xray" title="شفاف">👻 شفاف</button>
        <span class="sep"></span>
        <button data-t="light" class="on" title="الإضاءة">💡 إضاءة</button>
        <button data-t="bones" title="الهيكل العظمي">🦴 هيكل</button>
        <button data-t="spin" title="دوران تلقائي">🔄 دوران</button>
        <button data-t="play" title="تشغيل الحركة" style="display:none">▶️ تشغيل</button>
        <button data-t="reset" title="إعادة الكاميرا">⟲</button>
      </div>
      <div class="stats"></div><div class="msg">…</div>`;
    this.$stats = root.querySelector('.stats');
    this.$msg = root.querySelector('.msg');
    this.mode = 'color'; this.light = true; this.bones = false;

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    root.appendChild(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.lights = new THREE.Group();
    this.lights.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
    const d1 = new THREE.DirectionalLight(0xffffff, 2.2); d1.position.set(3, 5, 4);
    const d2 = new THREE.DirectionalLight(0xbfd4ff, 0.9); d2.position.set(-4, 2, -3);
    this.lights.add(d1, d2);
    this.scene.add(this.lights);
    this.grid = new THREE.GridHelper(4, 20, 0x555a66, 0x3a3e48);
    this.scene.add(this.grid);

    root.querySelectorAll('button[data-mode]').forEach(b => b.onclick = () => {
      root.querySelectorAll('button[data-mode]').forEach(x => x.classList.toggle('on', x === b));
      this.mode = b.dataset.mode; this.applyMode();
    });
    root.querySelectorAll('button[data-t]').forEach(b => b.onclick = () => {
      const t = b.dataset.t;
      if (t === 'reset') return this.frame();
      if (t === 'play') return this.togglePlay(b);
      if (t === 'spin') { this.controls.autoRotate = !this.controls.autoRotate; b.classList.toggle('on', this.controls.autoRotate); }
      if (t === 'light') { this.light = !this.light; b.classList.toggle('on', this.light); this.applyMode(); }
      if (t === 'bones') { this.bones = !this.bones; b.classList.toggle('on', this.bones); if (this.skel) this.skel.visible = this.bones; }
    });

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this);
    this.clock = new THREE.Clock();
    const loop = () => {
      this._raf = requestAnimationFrame(loop);
      const dt = this.clock.getDelta();
      if (this.mixer && this.playing) {
        // follow the audio clock when there is sound, so lips stay in sync
        if (this.audio && !this.audio.paused) this.mixer.setTime(this.audio.currentTime);
        else this.mixer.update(dt);
      }
      this.controls.update(); r.render(this.scene, this.camera);
    };
    loop();
    this.load(this.getAttribute('src'));
  }

  togglePlay(btn) {
    if (!this.mixer) return;
    this.playing = !this.playing;
    btn.classList.toggle('on', this.playing);
    btn.textContent = this.playing ? '⏸️ إيقاف' : '▶️ تشغيل';
    if (this.audio) {
      if (this.playing) { this.audio.currentTime = 0; this.mixer.setTime(0); this.audio.play().catch(() => {}); }
      else this.audio.pause();
    }
  }

  disconnectedCallback() {
    cancelAnimationFrame(this._raf);
    this.audio && this.audio.pause();
    this._ro && this._ro.disconnect();
    this.clear();
    this.renderer && (this.renderer.dispose(), this.renderer.forceContextLoss());
  }

  attributeChangedCallback(name, oldV, newV) { if (this._init && oldV !== newV) this.load(newV); }

  resize() {
    const w = this.clientWidth, h = this.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  clear() {
    if (!this.model) return;
    this.scene.remove(this.model);
    this.skel && this.scene.remove(this.skel);
    this.model.traverse(o => { o.geometry && o.geometry.dispose(); });
    this.model = null; this.skel = null; this.meshes = [];
  }

  load(src) {
    this.clear();
    if (!src) { this.$msg.textContent = 'لا يوجد مجسم بعد'; this.$stats.style.display = 'none'; return; }
    this.$msg.textContent = 'جارٍ التحميل…';
    new GLTFLoader().load(src, g => {
      this.$msg.textContent = '';
      this.model = g.scene; this.meshes = [];
      let tris = 0, verts = 0, skinned = false;
      this.model.traverse(o => {
        if (!o.isMesh) return;
        const geo = o.geometry;
        tris += (geo.index ? geo.index.count : geo.attributes.position.count) / 3;
        verts += geo.attributes.position.count;
        if (o.isSkinnedMesh) skinned = true;
        o.userData.orig = o.material;
        o.userData.edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 1),
          new THREE.LineBasicMaterial({ color: 0x9fb4ff, transparent: true, opacity: 0.55 }));
        o.add(o.userData.edges);
        this.meshes.push(o);
      });
      this.scene.add(this.model);
      const playBtn = this.shadowRoot.querySelector('button[data-t="play"]');
      this.mixer = null; this.playing = false;
      if (g.animations && g.animations.length) {
        this.mixer = new THREE.AnimationMixer(this.model);
        g.animations.forEach(a => this.mixer.clipAction(a).play());
        this.mixer.setTime(0);
        const a = this.getAttribute('audio');
        this.audio = a ? new Audio(a) : null;
        if (this.audio) this.audio.onended = () => { if (this.playing) this.togglePlay(playBtn); };
        playBtn.style.display = '';
      } else { playBtn.style.display = 'none'; this.audio = null; }
      if (skinned) {
        this.skel = new THREE.SkeletonHelper(this.model);
        this.skel.visible = this.bones;
        this.scene.add(this.skel);
      }
      let info = {};
      try { info = JSON.parse(this.getAttribute('info') || '{}'); } catch (e) {}
      const parts = [`المثلثات ${fmt(Math.round(tris))}`, `النقاط ${fmt(verts)}`];
      if (info.quads) parts.unshift(`مربعات Quad ${fmt(info.quads)}`);
      if (skinned) parts.push('🦴 بهيكل عظمي');
      this.$stats.textContent = parts.join('  •  ');
      this.$stats.style.display = '';
      this.applyMode(); this.frame();
    }, undefined, () => { this.$msg.textContent = 'تعذّر فتح المجسم'; });
  }

  applyMode() {
    for (const o of this.meshes || []) {
      const orig = o.userData.orig;
      const base = Array.isArray(orig) ? orig[0] : orig;
      const map = this.mode === 'color' ? base.map : null;
      const color = this.mode === 'color' ? (base.color ? base.color.clone() : new THREE.Color(1, 1, 1))
                                          : new THREE.Color(0xc9ccd4);
      const opts = { map, color, side: THREE.DoubleSide };
      const m = this.light ? new THREE.MeshStandardMaterial({ ...opts, roughness: 0.8, metalness: 0.0 })
                           : new THREE.MeshBasicMaterial(opts);
      if (this.mode === 'xray') { m.transparent = true; m.opacity = 0.28; m.depthWrite = false; }
      if (this.mode === 'wire') { m.polygonOffset = true; m.polygonOffsetFactor = 1; m.polygonOffsetUnits = 1; }
      if (o.material !== orig) o.material.dispose();
      o.material = (this.mode === 'color' && this.light) ? orig : m;
      o.userData.edges.visible = this.mode === 'wire' || this.mode === 'xray';
    }
    this.lights.visible = this.light;
  }

  frame() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
    const r = Math.max(size.x, size.y, size.z) || 1;
    this.grid.scale.setScalar(r / 2);
    this.grid.position.set(c.x, box.min.y, c.z);
    this.controls.target.copy(c);
    this.controls.target.y -= r * 0.04;
    this.camera.position.set(c.x + r * 1.05, c.y + r * 0.5, c.z + r * 1.9);
    this.camera.near = r / 200; this.camera.far = r * 50; this.camera.updateProjectionMatrix();
    this.controls.update();
  }
}

customElements.define('s3d-viewer', S3DViewer);
