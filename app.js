import { OccViewer, loadScene, occIoU, CLASS_NAMES, PALETTE_CSS, ERR_COLORS } from './viewer.js';

// k0: the last input frame shown first (keyframe index)
const SCENES = [
  { id: 'scene-0795', k0: 14, title: 'Busy intersection', note: 'crossing an intersection past a stopped bus, cones, and pedestrians on the sidewalk' },
  { id: 'scene-0799', k0: 30, title: 'Crossing', note: 'a motorcycle overtakes while the car crosses an intersection' },
  { id: 'scene-0796', k0: 12, title: 'Scooters and buses', note: 'scooters, buses, and pedestrians around an intersection' },
  { id: 'scene-0921', k0: 26, title: 'Left turn', note: 'turning left past a bus, a truck, and pedestrians' },
  { id: 'scene-0564', k0: 10, title: 'Construction', note: 'turning right past a construction zone and a parked car' },
  { id: 'scene-1069', k0: 10, title: 'At night', note: 'night traffic with a truck and congestion' },
];
const $ = s => document.querySelector(s);

// ---------------------------------------------------------------- legend
const LEGEND_SEM = CLASS_NAMES.map((n, i) => `<span class="lg"><i style="background:${PALETTE_CSS[i]}"></i>${n}</span>`).join('');
const LEGEND_ERR = `<span class="lg"><i style="background:${ERR_COLORS.hit}"></i>correct: occupied in both</span>` +
  `<span class="lg"><i style="background:${ERR_COLORS.miss}"></i>missed: occupied, predicted free</span>` +
  `<span class="lg"><i style="background:${ERR_COLORS.false}"></i>false: predicted, free in truth</span>`;

// ---------------------------------------------------------------- viewer
const panels = ['#p-gt', '#p-ours', '#p-tc'].map(s => ({ el: $(s) }));
const viewer = new OccViewer($('#occ-canvas'), panels);
const state = { data: null, scene: null, a: 0, h: 1, playing: false, mode: 'err' };
function setMode(m) {
  state.mode = m;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  $('#legend').innerHTML = m === 'err' ? LEGEND_ERR : LEGEND_SEM;
  $('#legend').classList.toggle('err', m === 'err');
  update();
}
document.querySelectorAll('.mode-btn').forEach(b => b.onclick = () => setMode(b.dataset.m));

const sceneBar = $('#scene-bar');
sceneBar.innerHTML = SCENES.map((s, i) =>
  `<button class="tab" data-i="${i}"><img src="data/${s.id}/cam/10.jpg" alt="" loading="lazy">${s.title}</button>`).join('');
sceneBar.querySelectorAll('.tab').forEach(b => b.onclick = () => selectScene(+b.dataset.i));

async function selectScene(i) {
  stop();
  sceneBar.querySelectorAll('.tab').forEach((b, j) => b.classList.toggle('on', i === j));
  $('#viewer-status').textContent = 'loading…';
  state.data = await loadScene(SCENES[i].id); state.scene = SCENES[i];
  $('#scene-note').textContent = SCENES[i].title + ': ' + SCENES[i].note + '.';
  const A = state.data.index.anchors;
  const slider = $('#anchor'); slider.max = A.length - 1;
  slider.value = Math.max(0, A.indexOf(SCENES[i].k0));
  state.a = +slider.value; state.h = 3; $('#horizon').value = 3;
  await update();
  $('#viewer-status').textContent = '';
}

function maxH(k) {
  let h = 0; while (state.data.has(`ours/${k}/${h + 1}`)) h++; return h;
}

async function update() {
  const d = state.data; if (!d) return;
  const k = d.index.anchors[state.a];
  const hmax = maxH(k); if (state.h > hmax) state.h = hmax;
  $('#horizon').max = 8; $('#horizon').value = state.h;
  const tgt = k + 2 * state.h;
  const [gt, ours, tc] = await Promise.all([d.grid(`gt/${tgt}`), d.grid(`ours/${k}/${state.h}`), d.grid(`tc/${k}/${state.h}`)]);
  viewer.setGrids([gt, ours, tc], state.mode);
  const fmt = v => v == null ? '' : ` · IoU ${v.toFixed(1)}`;
  $('#iou-ours').textContent = fmt(occIoU(ours, gt)); $('#iou-tc').textContent = fmt(occIoU(tc, gt));
  $('#p-tc').classList.toggle('na', !tc);
  $('#h-label').textContent = state.h === 0 ? 'current frame' : `+${state.h} s`;
  $('#t-label').textContent = `t₀ = ${(k / 2).toFixed(1)} s`;
  $('#cam-in').src = `data/${state.scene.id}/cam/${k}.jpg`;
  $('#cam-out').src = `data/${state.scene.id}/cam/${tgt}.jpg`;
  $('#cam-out-label').textContent = state.h === 0 ? 'Same frame, horizon 0' : `Actual front camera at +${state.h} s, never an input`;
  document.querySelectorAll('.hz-tick').forEach(e => e.classList.toggle('on', +e.dataset.h === state.h));
  document.querySelectorAll('.hz-tick').forEach(e => e.classList.toggle('dis', +e.dataset.h > hmax));
}

$('#anchor').oninput = e => { state.a = +e.target.value; update(); };
$('#horizon').oninput = e => { state.h = +e.target.value; update(); };
$('#hz-ticks').innerHTML = Array.from({ length: 9 }, (_, h) => `<span class="hz-tick" data-h="${h}">${h === 0 ? 'now' : h + 's'}</span>`).join('');
document.querySelectorAll('.hz-tick').forEach(e => e.onclick = () => { state.h = +e.dataset.h; update(); });
$('#reset-view').onclick = () => viewer.resetView();

// play: sweep the horizon 0 -> 8 s, then move to the next anchor
let timer = null;
function stop() { state.playing = false; clearTimeout(timer); $('#play').textContent = '▶ Play'; }
async function tick() {
  if (!state.playing) return;
  const k = state.data.index.anchors[state.a];
  if (state.h < maxH(k)) state.h++;
  else { state.a = (state.a + 1) % state.data.index.anchors.length; state.h = 0; $('#anchor').value = state.a; }
  await update();
  timer = setTimeout(tick, state.h === 0 ? 1100 : 700);
}
$('#play').onclick = () => {
  if (state.playing) return stop();
  state.playing = true; $('#play').textContent = '❚❚ Pause'; tick();
};

setMode('err');
selectScene(0).then(() => { if (matchMedia('(prefers-reduced-motion: no-preference)').matches) $('#play').click(); });

// ---------------------------------------------------------------- video: show the section once the file exists
fetch('video/persistworld.mp4', { method: 'HEAD' })
  .then(r => { if (r.ok) $('#video').hidden = $('#video-btn').hidden = false; }).catch(() => {});

// ---------------------------------------------------------------- bibtex copy
$('#copy-bib').onclick = () => {
  navigator.clipboard.writeText($('#bibtex').innerText);
  $('#copy-bib').textContent = 'Copied'; setTimeout(() => $('#copy-bib').textContent = 'Copy', 1500);
};
