// PersistWorld project page: interactive occupancy viewer.
// One WebGL canvas, three viewports (ground truth / PersistWorld / SparseWorld-TC) sharing one orbit camera.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const NX = 200, NY = 200, NZ = 16, VOX = 0.4, FREE = 17;
const ORIGIN = [-40, -40, -1];
// Soft Occ3D palette of the paper figures (class ids 0..16).
const PALETTE = [
  [70, 70, 70], [255, 120, 50], [255, 150, 170], [235, 200, 0], [0, 130, 235], [0, 200, 210],
  [200, 160, 0], [230, 0, 0], [250, 210, 90], [135, 60, 0], [150, 40, 220], [226, 214, 230],
  [206, 201, 196], [202, 181, 213], [198, 224, 166], [128, 136, 168], [52, 150, 62],
].map(([r, g, b]) => new THREE.Color(r / 255, g / 255, b / 255));
export const CLASS_NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle',
  'pedestrian', 'traffic cone', 'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain',
  'manmade', 'vegetation'];
export const PALETTE_CSS = PALETTE.map(c => '#' + c.getHexString());

// ------------------------------------------------------------------ data
class SceneData {
  constructor(base) { this.base = base; this.cache = new Map(); }
  async load() {
    const [idx, buf] = await Promise.all([
      fetch(this.base + '/index.json').then(r => r.json()),
      fetch(this.base + '/pack.bin').then(r => r.arrayBuffer()),
    ]);
    this.index = idx; this.buf = buf; return this;
  }
  has(key) { return key in this.index.entries; }
  async grid(key) {
    if (this.cache.has(key)) return this.cache.get(key);
    const e = this.index.entries[key];
    if (!e) return null;
    const p = (async () => {
      const ds = new DecompressionStream('deflate');
      const stream = new Blob([new Uint8Array(this.buf, e[0], e[1])]).stream().pipeThrough(ds);
      return new Uint8Array(await new Response(stream).arrayBuffer());
    })();
    this.cache.set(key, p);
    return p;
  }
}

// ------------------------------------------------------------------ one voxel layer
function makeLayer(max = 90000) {
  const geo = new THREE.BoxGeometry(VOX * 0.94, VOX * 0.94, VOX * 0.94);
  const mat = new THREE.MeshLambertMaterial({ vertexColors: false });
  const mesh = new THREE.InstancedMesh(geo, mat, max);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3);
  mesh.count = 0; mesh.frustumCulled = false;
  return mesh;
}
const _m = new THREE.Matrix4();
// Occ3D grid (x forward, y left, z up)  ->  three.js (x right, y up, -z forward)
// Error view colours: correct occupied voxel, missed (ground truth occupied, predicted free), false (predicted, free in truth)
export const ERR_COLORS = { hit: '#c3cbdb', miss: '#ff3b5c', false: '#ffb020' };
const ERR = Object.fromEntries(Object.entries(ERR_COLORS).map(([k, v]) => [k, new THREE.Color(v)]));

// mode 'sem': semantic colours. mode 'err' (needs ref = ground truth): hits grey, misses red, false voxels amber.
function fillLayer(mesh, grid, ref = null, mode = 'sem') {
  let n = 0;
  const err = mode === 'err' && ref && grid !== ref;
  if (grid) {
    for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) {
      const o = (x * NY + y) * NZ;
      for (let z = 0; z < NZ; z++) {
        const c = grid[o + z], g = ref ? ref[o + z] : FREE;
        let col;
        if (err) {
          if (c !== FREE) col = g !== FREE ? ERR.hit : ERR.false;
          else if (g !== FREE) col = ERR.miss;
          else continue;
        } else {
          if (c === FREE) continue;
          col = PALETTE[c];
        }
        if (n >= mesh.instanceMatrix.count) continue;
        const wx = ORIGIN[0] + (x + 0.5) * VOX, wy = ORIGIN[1] + (y + 0.5) * VOX, wz = ORIGIN[2] + (z + 0.5) * VOX;
        _m.makeTranslation(-wy, wz, -wx);
        mesh.setMatrixAt(n, _m);
        const k = 0.82 + 0.18 * Math.min(1, z / 10);   // slight height shading, for depth
        mesh.instanceColor.setXYZ(n, col.r * k, col.g * k, col.b * k);
        n++;
      }
    }
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
}

// Geometric IoU of occupied voxels (both grids already cut to the same camera mask).
export function occIoU(pred, gt) {
  if (!pred || !gt) return null;
  let i = 0, u = 0;
  for (let j = 0; j < pred.length; j++) {
    const p = pred[j] !== FREE, g = gt[j] !== FREE;
    if (p || g) { u++; if (p && g) i++; }
  }
  return u ? 100 * i / u : null;
}

function egoCar() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.9, 1.4, 4.4), new THREE.MeshLambertMaterial({ color: 0xf4f6fb }));
  body.position.set(0, 0.25, 0);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.7, 2.2), new THREE.MeshLambertMaterial({ color: 0x2b3245 }));
  cab.position.set(0, 1.2, 0.2);
  g.add(body, cab);
  return g;
}

// ------------------------------------------------------------------ viewer
export class OccViewer {
  constructor(canvas, panels, opts = {}) {
    this.canvas = canvas; this.panels = panels;           // [{key, el}] DOM boxes marking each viewport
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: !!opts.preserve });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.5, 400);
    this.camera.position.set(0, 21, 31); // behind and above the ego (chase view of the paper figures)
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, -2, -13); this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49; this.controls.minDistance = 8; this.controls.maxDistance = 120;
    this.dirty = true; this.controls.addEventListener('change', () => { this.dirty = true; });
    this.scenes = panels.map(() => {
      const s = new THREE.Scene();
      s.add(new THREE.HemisphereLight(0xffffff, 0x55607a, 1.05));
      const d = new THREE.DirectionalLight(0xffffff, 2.1); d.position.set(-25, 40, 18); s.add(d);
      const f = new THREE.DirectionalLight(0xbcd0ff, 0.5); f.position.set(30, 15, -20); s.add(f);
      const layer = makeLayer(); s.add(layer); s.add(egoCar());
      return { scene: s, layer };
    });
    this.onResize(); window.addEventListener('resize', () => this.onResize());
    if (!opts.manual) this.renderer.setAnimationLoop(() => this.draw());
  }
  onResize() {
    const r = this.canvas.getBoundingClientRect();
    this.renderer.setSize(r.width, r.height, false); this.dirty = true;
  }
  // grids[0] is the ground truth; mode 'err' colours the other panels by their errors against it
  setGrids(grids, mode = 'sem') { grids.forEach((g, i) => fillLayer(this.scenes[i].layer, g, grids[0], i ? mode : 'sem')); this.dirty = true; }
  draw() {
    this.controls.update();
    // render only when something changed (camera, data, size); keeps idle pages and slow GPUs cool
    const cr0 = this.canvas.getBoundingClientRect();
    if (cr0.bottom < 0 || cr0.top > innerHeight) return;
    if (!this.dirty) return; this.dirty = false;
    this.render();
  }
  render() {
    const cr = this.canvas.getBoundingClientRect();
    const r = this.renderer; r.setScissorTest(true);
    r.setClearColor(0x000000, 0); r.clear();
    this.panels.forEach((p, i) => {
      const b = p.el.getBoundingClientRect();
      const x = b.left - cr.left, y = cr.bottom - b.bottom, w = b.width, h = b.height;
      if (w <= 0 || h <= 0) return;
      r.setViewport(x, y, w, h); r.setScissor(x, y, w, h);
      r.setClearColor(0x161e33, 1); r.clear();
      this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
      r.render(this.scenes[i].scene, this.camera);
    });
  }
  resetView() { this.camera.position.set(0, 21, 31); this.controls.target.set(0, -2, -13); this.dirty = true; }
}

export async function loadScene(name, root = 'data/') { return new SceneData(root + name).load(); }
