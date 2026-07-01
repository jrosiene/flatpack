// flatpack seam editor.
//
// Modes:
//   orbit  - camera only
//   seam   - click vertices; each click extends the current seam along the
//            shortest surface path from the previous click (server Dijkstra)
//   notch  - click a vertex to toggle a match notch
//   grain  - click two vertices to set the selected panel's grainline
//
// A scripting hook is exposed as window.flatpack for tests and power users.

import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
  mode: "orbit",
  meshName: "",
  positions: null,      // Float64Array view of vertex coords (n*3)
  faces: null,          // Int32Array (t*3)
  vertexFaces: null,    // Map vertex -> [face indices]
  seams: [],            // committed: {name, legs: [[v,...], ...]}
  currentLegs: [],      // legs of the seam being drawn
  notches: new Set(),   // vertex ids
  labels: null,         // per-face panel label from last split preview
  nPanels: 0,
  panelProps: new Map(),// label -> {name, fabric, stretch_axis_deg, grain}
  selectedPanel: null,  // label
  grainPending: null,   // first grain vertex clicked
  diag: 1,              // mesh bbox diagonal, for sizing markers
};

const PALETTE = [0x4e79a7, 0xf28e2b, 0x59a14f, 0xe15759, 0xb07aa1, 0x76b7b2,
                 0xedc948, 0xff9da7, 0x9c755f, 0xbab0ac, 0x86bcb6, 0xd37295];
const BASE_COLOR = new THREE.Color(0xb8bcc2);

// ---------------------------------------------------------------------------
// three.js scaffolding
// ---------------------------------------------------------------------------

const viewport = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x24262b);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1e6);
const controls = new OrbitControls(camera, renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3d44, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(1, 2, 1.5);
scene.add(sun);

let displayMesh = null;              // non-indexed, per-face colourable
const overlay = new THREE.Group();   // seams, markers, notches, grain arrows
scene.add(overlay);
const cursor = marker(0xffe066);     // hover highlight
cursor.visible = false;
scene.add(cursor);

function marker(color, scale = 1) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(1, 12, 8),
    new THREE.MeshBasicMaterial({ color, depthTest: false })
  );
  m.renderOrder = 10;
  m.userData.baseScale = scale;
  return m;
}

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// ---------------------------------------------------------------------------
// mesh loading
// ---------------------------------------------------------------------------

async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function loadMesh() {
  const data = await api("/api/mesh");
  state.meshName = data.name;
  state.positions = new Float64Array(data.vertices);
  state.faces = new Int32Array(data.faces);

  state.vertexFaces = new Map();
  for (let f = 0; f < state.faces.length / 3; f++) {
    for (let k = 0; k < 3; k++) {
      const v = state.faces[3 * f + k];
      if (!state.vertexFaces.has(v)) state.vertexFaces.set(v, []);
      state.vertexFaces.get(v).push(f);
    }
  }

  // Non-indexed geometry so each face can be coloured independently;
  // face i of the geometry is faces[3i..3i+2] of the original mesh.
  const t = state.faces.length / 3;
  const pos = new Float32Array(t * 9);
  const col = new Float32Array(t * 9);
  for (let i = 0; i < t * 3; i++) {
    const v = state.faces[i];
    pos.set([state.positions[3 * v], state.positions[3 * v + 1], state.positions[3 * v + 2]], 3 * i);
    BASE_COLOR.toArray(col, 3 * i);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();

  displayMesh = new THREE.Mesh(
    geo,
    new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })
  );
  scene.add(displayMesh);
  scene.add(new THREE.LineSegments(
    new THREE.WireframeGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x50545a, transparent: true, opacity: 0.15 })
  ));

  geo.computeBoundingSphere();
  const bs = geo.boundingSphere;
  state.diag = bs.radius * 2;
  camera.position.copy(bs.center).add(new THREE.Vector3(0, -bs.radius * 1.2, bs.radius * 1.8));
  camera.near = bs.radius / 100;
  camera.far = bs.radius * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(bs.center);

  document.getElementById("mesh-info").textContent =
    `${data.name}: ${state.positions.length / 3} vertices, ${t} faces`;
  resize();
}

function vertexPos(v) {
  return new THREE.Vector3(
    state.positions[3 * v], state.positions[3 * v + 1], state.positions[3 * v + 2]);
}

// ---------------------------------------------------------------------------
// picking
// ---------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();

function pickVertex(clientX, clientY) {
  if (!displayMesh) return null;
  const r = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((clientX - r.left) / r.width) * 2 - 1,
    -((clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = raycaster.intersectObject(displayMesh)[0];
  if (!hit) return null;
  // Nearest corner of the hit face, in original mesh indexing.
  let best = null, bestD = Infinity;
  for (let k = 0; k < 3; k++) {
    const v = state.faces[3 * hit.faceIndex + k];
    const d = vertexPos(v).distanceTo(hit.point);
    if (d < bestD) { bestD = d; best = v; }
  }
  return { vertex: best, face: hit.faceIndex };
}

let downAt = null;
renderer.domElement.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", e => {
  if (!downAt) return;
  const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
  downAt = null;
  if (moved > 5 || state.mode === "orbit") return;
  const pick = pickVertex(e.clientX, e.clientY);
  if (pick) handleClick(pick).catch(err => setStatus(err.message, true));
});
renderer.domElement.addEventListener("pointermove", e => {
  if (state.mode === "orbit") { cursor.visible = false; return; }
  const pick = pickVertex(e.clientX, e.clientY);
  cursor.visible = !!pick;
  if (pick) placeMarker(cursor, pick.vertex, 0.008);
});

function placeMarker(m, v, size) {
  m.position.copy(vertexPos(v));
  m.scale.setScalar(state.diag * size);
}

// ---------------------------------------------------------------------------
// click handling per mode
// ---------------------------------------------------------------------------

async function handleClick({ vertex, face }) {
  if (state.mode === "seam") return addSeamVertex(vertex);
  if (state.mode === "notch") return toggleNotch(vertex);
  if (state.mode === "grain") return grainClick(vertex);
}

async function addSeamVertex(v) {
  const last = lastSeamVertex();
  if (last === null) {
    state.currentLegs.push([v]);
  } else {
    if (v === last) return;
    const { path } = await api("/api/path", { start: last, end: v });
    state.currentLegs.push(path.slice(1));
  }
  invalidateSplit();
  redrawOverlay();
  updateButtons();
  setStatus(`seam: ${currentSeamPath().length} vertices`);
}

function lastSeamVertex() {
  const legs = state.currentLegs;
  return legs.length ? legs[legs.length - 1][legs[legs.length - 1].length - 1] : null;
}

function currentSeamPath() { return state.currentLegs.flat(); }

function finishSeam() {
  const path = currentSeamPath();
  if (path.length < 2) return;
  state.seams.push({ name: `seam_${state.seams.length + 1}`, legs: state.currentLegs });
  state.currentLegs = [];
  invalidateSplit();
  renderSeamList();
  redrawOverlay();
  updateButtons();
}

function undoLeg() {
  state.currentLegs.pop();
  redrawOverlay();
  updateButtons();
}

function toggleNotch(v) {
  state.notches.has(v) ? state.notches.delete(v) : state.notches.add(v);
  redrawOverlay();
  setStatus(`${state.notches.size} notch(es)`);
}

function grainClick(v) {
  if (state.selectedPanel === null) { setStatus("select a panel first", true); return; }
  if (state.grainPending === null) {
    state.grainPending = v;
    setStatus("grainline: click the second vertex");
  } else {
    props(state.selectedPanel).grain = [state.grainPending, v];
    state.grainPending = null;
    setStatus("grainline set");
    redrawOverlay();
  }
}

// ---------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------

function props(label) {
  if (!state.panelProps.has(label)) {
    state.panelProps.set(label, {
      name: `panel_${label}`, fabric: "rigid", stretch_axis_deg: 0, grain: null,
    });
  }
  return state.panelProps.get(label);
}

function invalidateSplit() {
  state.labels = null;
  state.nPanels = 0;
  state.selectedPanel = null;
  renderPanelList();
  if (displayMesh) paintFaces();
}

async function previewSplit() {
  const seams = allSeamPaths();
  const data = await api("/api/split", { seams });
  state.labels = data.labels;
  state.nPanels = data.n_panels;
  paintFaces();
  renderPanelList();
  setStatus(`${data.n_panels} panel(s)`);
}

function allSeamPaths() {
  const paths = state.seams.map(s => s.legs.flat());
  if (state.currentLegs.length) paths.push(currentSeamPath());
  return paths.filter(p => p.length >= 2);
}

function paintFaces() {
  const col = displayMesh.geometry.getAttribute("color");
  const c = new THREE.Color();
  for (let f = 0; f < col.count / 3; f++) {
    if (state.labels) c.setHex(PALETTE[state.labels[f] % PALETTE.length]);
    else c.copy(BASE_COLOR);
    if (state.labels && state.selectedPanel === state.labels[f]) c.offsetHSL(0, 0, 0.15);
    for (let k = 0; k < 3; k++) c.toArray(col.array, 9 * f + 3 * k);
  }
  col.needsUpdate = true;
}

function selectPanel(label) {
  state.selectedPanel = label;
  paintFaces();
  renderPanelList();
  const p = props(label);
  document.getElementById("panel-props").classList.remove("hidden");
  document.getElementById("panel-name").value = p.name;
  document.getElementById("panel-fabric").value = p.fabric;
  document.getElementById("panel-axis").value = p.stretch_axis_deg;
  document.getElementById("panel-extra").textContent =
    p.grain ? `grain: ${p.grain[0]} → ${p.grain[1]}` : "no grainline (use Grainline mode)";
}

// ---------------------------------------------------------------------------
// spec building + generate
// ---------------------------------------------------------------------------

function buildSpec() {
  if (!state.labels) throw new Error("run Preview split first");
  const panels = [];
  for (let label = 0; label < state.nPanels; label++) {
    const p = props(label);
    const panel = {
      name: p.name,
      anchor_face: state.labels.indexOf(label),
      fabric: p.fabric,
      stretch_axis_deg: Number(p.stretch_axis_deg) || 0,
      notches: [...state.notches].filter(v =>
        (state.vertexFaces.get(v) || []).some(f => state.labels[f] === label)),
    };
    if (p.grain) panel.grain = p.grain;
    panels.push(panel);
  }
  return {
    units: "mm",
    seam_allowance: Number(document.getElementById("allowance").value) || 10,
    seams: state.seams.map(s => ({ name: s.name, path: s.legs.flat() })),
    panels,
  };
}

async function generate() {
  setStatus("generating…");
  if (!state.labels) await previewSplit();
  const data = await api("/api/generate", buildSpec());
  const lines = Object.entries(data.report).map(([name, r]) => {
    const d = r.distortion, f = r.fabric_fit;
    return `${name} (${f.fabric}): area ${d.area_ratio_worst_low.toFixed(3)}–` +
      `${d.area_ratio_worst_high.toFixed(3)}, ${(f.fraction_ok * 100).toFixed(0)}% ok, ` +
      `${f.triangles_needing_dart} dart / ${f.triangles_needing_relief} relief`;
  });
  const rep = document.getElementById("report");
  rep.textContent = lines.join("\n");
  rep.classList.remove("hidden");
  document.getElementById("downloads").innerHTML = data.files
    .map(f => `<a href="/files/${f}" download>${f}</a>`).join(" ");
  setStatus(`generated ${data.panels.length} panel(s)`);
  const img = document.getElementById("svg-img");
  img.src = `/files/pattern.svg?t=${Date.now()}`;
  document.getElementById("svg-preview").classList.remove("hidden");
}

async function saveSpec() {
  if (!state.labels) await previewSplit();
  const { saved } = await api("/api/save", buildSpec());
  setStatus(`saved ${saved}`);
}

// ---------------------------------------------------------------------------
// overlay drawing
// ---------------------------------------------------------------------------

function polyline(path, color) {
  const pts = path.map(vertexPos);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, depthTest: false })
  );
  line.renderOrder = 5;
  return line;
}

function redrawOverlay() {
  overlay.clear();
  for (const seam of state.seams) overlay.add(polyline(seam.legs.flat(), 0xff5252));
  const current = currentSeamPath();
  if (current.length) {
    if (current.length >= 2) overlay.add(polyline(current, 0xffb300));
    for (const leg of state.currentLegs) {
      const m = marker(0xffb300);
      placeMarker(m, leg[leg.length - 1], 0.006);
      overlay.add(m);
    }
  }
  for (const v of state.notches) {
    const m = marker(0x40c4ff);
    placeMarker(m, v, 0.007);
    overlay.add(m);
  }
  for (const [, p] of state.panelProps) {
    if (p.grain) overlay.add(polyline(p.grain, 0x69f0ae));
  }
}

// ---------------------------------------------------------------------------
// sidebar UI
// ---------------------------------------------------------------------------

const HINTS = {
  orbit: "drag to rotate, right-drag to pan, wheel to zoom",
  seam: "click vertices; seam follows the surface between clicks",
  notch: "click a vertex to toggle a match notch",
  grain: "select a panel, then click two vertices",
};

function setMode(mode) {
  state.mode = mode;
  controls.enableRotate = mode === "orbit";
  for (const b of document.querySelectorAll("button.mode"))
    b.classList.toggle("active", b.id === `mode-${mode}`);
  document.getElementById("mode-hint").textContent = HINTS[mode];
}

function renderSeamList() {
  const ul = document.getElementById("seam-list");
  ul.innerHTML = "";
  state.seams.forEach((seam, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${seam.name} (${seam.legs.flat().length} verts)</span>`;
    const del = document.createElement("button");
    del.textContent = "×";
    del.onclick = () => {
      state.seams.splice(i, 1);
      invalidateSplit(); renderSeamList(); redrawOverlay();
    };
    li.appendChild(del);
    ul.appendChild(li);
  });
}

function renderPanelList() {
  const ul = document.getElementById("panel-list");
  ul.innerHTML = "";
  document.getElementById("panel-props").classList.toggle("hidden", state.selectedPanel === null);
  for (let label = 0; label < state.nPanels; label++) {
    const li = document.createElement("li");
    li.classList.toggle("selected", label === state.selectedPanel);
    const color = "#" + new THREE.Color(PALETTE[label % PALETTE.length]).getHexString();
    li.innerHTML = `<span><span class="swatch" style="background:${color}"></span> ${props(label).name}</span>`;
    li.onclick = () => selectPanel(label);
    ul.appendChild(li);
  }
}

function updateButtons() {
  document.getElementById("finish-seam").disabled = currentSeamPath().length < 2;
  document.getElementById("undo-leg").disabled = state.currentLegs.length === 0;
}

function setStatus(text, isError = false) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.style.color = isError ? "#ff8a80" : "";
}

for (const mode of ["orbit", "seam", "notch", "grain"])
  document.getElementById(`mode-${mode}`).onclick = () => setMode(mode);
document.getElementById("finish-seam").onclick = finishSeam;
document.getElementById("undo-leg").onclick = undoLeg;
document.getElementById("preview-split").onclick = () => previewSplit().catch(e => setStatus(e.message, true));
document.getElementById("generate").onclick = () => generate().catch(e => setStatus(e.message, true));
document.getElementById("save-spec").onclick = () => saveSpec().catch(e => setStatus(e.message, true));
document.getElementById("close-preview").onclick = () =>
  document.getElementById("svg-preview").classList.add("hidden");
document.getElementById("panel-name").oninput = e => {
  if (state.selectedPanel !== null) { props(state.selectedPanel).name = e.target.value; renderPanelList(); }
};
document.getElementById("panel-fabric").onchange = e => {
  if (state.selectedPanel !== null) props(state.selectedPanel).fabric = e.target.value;
};
document.getElementById("panel-axis").oninput = e => {
  if (state.selectedPanel !== null) props(state.selectedPanel).stretch_axis_deg = e.target.value;
};

setMode("orbit");
loadMesh().catch(e => setStatus(e.message, true));

// Scripting/testing hook: everything the mouse can do, callable from code.
window.flatpack = {
  state, addSeamVertex, finishSeam, previewSplit, generate, saveSpec,
  selectPanel, toggleNotch, setMode, buildSpec,
  ready: () => !!displayMesh,
};
