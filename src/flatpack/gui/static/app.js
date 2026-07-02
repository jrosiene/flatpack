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
  measureA: null,       // first ruler vertex clicked
  measureLine: null,    // completed ruler: [a, b]
  darts: [],            // {name, path: [v, ...]} mouth -> apex
  dartPending: null,    // dart mouth awaiting an apex click
  marks: [],            // {vertex, type, label, toward}
  markPending: null,    // bar tack anchor awaiting a direction click
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

let wireframe = null;

async function loadMesh() {
  applyMeshData(await api("/api/mesh"), { fitCamera: true });
  resize();
}

// (Re)build the scene from a mesh payload. Used at startup and again
// whenever a straight cut changes the geometry on the server.
function applyMeshData(data, { fitCamera = false } = {}) {
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

  if (displayMesh) {
    scene.remove(displayMesh, wireframe);
    displayMesh.geometry.dispose();
    wireframe.geometry.dispose();
  }
  displayMesh = new THREE.Mesh(
    geo,
    new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })
  );
  wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x23262b, transparent: true, opacity: 0.55 })
  );
  wireframe.visible = document.getElementById("show-edges").checked;
  scene.add(displayMesh, wireframe);

  geo.computeBoundingSphere();
  const bs = geo.boundingSphere;
  state.diag = bs.radius * 2;
  if (fitCamera) {
    camera.position.copy(bs.center).add(new THREE.Vector3(0, -bs.radius * 1.2, bs.radius * 1.8));
    camera.near = bs.radius / 100;
    camera.far = bs.radius * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(bs.center);
  }

  document.getElementById("mesh-info").textContent =
    `${data.name}: ${state.positions.length / 3} vertices, ${t} faces`;
  updateMeshDims();
}

function updateMeshDims() {
  const p = state.positions;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3)
    for (let k = 0; k < 3; k++) {
      if (p[i + k] < lo[k]) lo[k] = p[i + k];
      if (p[i + k] > hi[k]) hi[k] = p[i + k];
    }
  const dims = hi.map((h, k) => h - lo[k]);
  const el = document.getElementById("mesh-dims");
  el.innerHTML = `size: ${dims.map(d => d.toFixed(d < 10 ? 1 : 0)).join(" × ")} mm`;
  const largest = Math.max(...dims);
  if (largest < 100) {
    el.innerHTML += ` <span class="warn">— suspiciously small for a pack:
      check units and use Scale ×</span>`;
  }
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
  if (state.mode === "measure") return measureClick(vertex);
  if (state.mode === "dart") return dartClick(vertex);
  if (state.mode === "mark") return markClick(vertex);
}

async function dartClick(v) {
  if (state.dartPending === null) {
    state.dartPending = v;
    setStatus("dart: mouth set - now click the apex (inside the panel)");
  } else if (v !== state.dartPending) {
    const { path } = await api("/api/path", { start: state.dartPending, end: v });
    state.darts.push({ name: `dart_${state.darts.length + 1}`, path });
    state.dartPending = null;
    renderDartMarkLists();
    setStatus("dart added - intake is computed when you generate");
  }
  redrawOverlay();
}

function markClick(v) {
  const type = document.getElementById("mark-type").value;
  const label = document.getElementById("mark-label").value.trim();
  if (type === "bartack") {
    if (state.markPending === null) {
      state.markPending = v;
      setStatus("bar tack: anchor set - click a second vertex for direction");
    } else if (v !== state.markPending) {
      state.marks.push({
        vertex: state.markPending,
        type,
        label: label || `bartack ${state.marks.length + 1}`,
        toward: v,
      });
      state.markPending = null;
      renderDartMarkLists();
      setStatus("bar tack added");
    }
  } else {
    state.marks.push({
      vertex: v,
      type,
      label: label || `attach ${state.marks.length + 1}`,
      toward: null,
    });
    renderDartMarkLists();
    setStatus("attachment point added");
  }
  redrawOverlay();
}

function renderDartMarkLists() {
  const dl = document.getElementById("dart-list");
  dl.innerHTML = "";
  state.darts.forEach((dart, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${dart.name} (${dart.path.length} verts)</span>`;
    const del = document.createElement("button");
    del.textContent = "×";
    del.onclick = () => { state.darts.splice(i, 1); renderDartMarkLists(); redrawOverlay(); };
    li.appendChild(del);
    dl.appendChild(li);
  });
  const ml = document.getElementById("mark-list");
  ml.innerHTML = "";
  state.marks.forEach((mark, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${mark.type}: ${mark.label}</span>`;
    const del = document.createElement("button");
    del.textContent = "×";
    del.onclick = () => { state.marks.splice(i, 1); renderDartMarkLists(); redrawOverlay(); };
    li.appendChild(del);
    ml.appendChild(li);
  });
}

function measureClick(v) {
  if (state.measureA === null || state.measureLine) {
    state.measureA = v;
    state.measureLine = null;
  } else if (v !== state.measureA) {
    state.measureLine = [state.measureA, v];
    state.measureA = null;
  }
  redrawOverlay();
  updateMeasureReadout();
}

function updateMeasureReadout() {
  const el = document.getElementById("measure-out");
  if (state.measureLine) {
    const [a, b] = state.measureLine;
    const d = vertexPos(a).distanceTo(vertexPos(b));
    el.textContent = `ruler: ${d.toFixed(1)} mm (straight line)`;
  } else if (state.measureA !== null) {
    el.textContent = "ruler: click the second point…";
  } else {
    el.textContent = "";
  }
}

async function applyScale(factor) {
  const data = await api("/api/scale", { factor });
  applyMeshData(data.mesh, { fitCamera: true });
  document.getElementById("reset-mesh").classList.remove("hidden");
  redrawOverlay();
  if (state.labels) paintFaces();
  updateMeasureReadout();
  setStatus(`mesh scaled ×${factor}`);
}

async function addSeamVertex(v) {
  const last = lastSeamVertex();
  if (last === null) {
    state.currentLegs.push([v]);
  } else {
    if (v === last) return;
    if (document.getElementById("straight-cut").checked) {
      // Cut straight across faces: the server inserts new vertices and
      // retriangulates, so reload the geometry it sends back. Existing
      // vertex indices (seams, notches, grainlines) stay valid.
      const data = await api("/api/cut", { start: last, end: v });
      applyMeshData(data.mesh);
      document.getElementById("reset-mesh").classList.remove("hidden");
      state.currentLegs.push(data.path.slice(1));
    } else {
      const { path } = await api("/api/path", { start: last, end: v });
      state.currentLegs.push(path.slice(1));
    }
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
  document.getElementById("clear-grain").disabled = !p.grain;
}

function clearGrain() {
  if (state.selectedPanel === null) return;
  props(state.selectedPanel).grain = null;
  state.grainPending = null;
  selectPanel(state.selectedPanel); // refresh the props display
  redrawOverlay();
  setStatus("grainline removed");
}

async function resetMesh() {
  if (!window.confirm(
    "Undo all straight cuts and restore the original mesh?\n" +
    "Seams, notches and grainlines will be cleared too, because they may " +
    "reference vertices created by the cuts.")) return;
  const data = await api("/api/reset", {});
  state.seams = [];
  state.currentLegs = [];
  state.notches.clear();
  state.panelProps.clear();
  state.grainPending = null;
  state.measureA = null;
  state.measureLine = null;
  state.darts = [];
  state.dartPending = null;
  state.marks = [];
  state.markPending = null;
  renderDartMarkLists();
  updateMeasureReadout();
  applyMeshData(data.mesh);
  invalidateSplit();
  renderSeamList();
  redrawOverlay();
  updateButtons();
  document.getElementById("reset-mesh").classList.add("hidden");
  setStatus("mesh restored");
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
    darts: state.darts.map(d => ({ name: d.name, path: d.path })),
    marks: state.marks,
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
  for (const dart of state.darts) overlay.add(polyline(dart.path, 0xff4081));
  if (state.dartPending !== null) {
    const m = marker(0xff4081);
    placeMarker(m, state.dartPending, 0.006);
    overlay.add(m);
  }
  for (const mark of state.marks) {
    const m = marker(0xffa726);
    placeMarker(m, mark.vertex, mark.type === "bartack" ? 0.005 : 0.008);
    overlay.add(m);
    if (mark.toward !== null) overlay.add(polyline([mark.vertex, mark.toward], 0xffa726));
  }
  if (state.markPending !== null) {
    const m = marker(0xffa726);
    placeMarker(m, state.markPending, 0.006);
    overlay.add(m);
  }
  if (state.measureA !== null) {
    const m = marker(0xffffff);
    placeMarker(m, state.measureA, 0.006);
    overlay.add(m);
  }
  if (state.measureLine) {
    overlay.add(polyline(state.measureLine, 0xffffff));
    for (const v of state.measureLine) {
      const m = marker(0xffffff);
      placeMarker(m, v, 0.006);
      overlay.add(m);
    }
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
  measure: "click two vertices to measure between them",
  dart: "click the dart mouth (on a boundary or seam), then the apex",
  mark: "attachment: one click; bar tack: click anchor, then direction",
};

function setMode(mode) {
  state.mode = mode;
  controls.enableRotate = mode === "orbit";
  for (const b of document.querySelectorAll("button.mode"))
    b.classList.toggle("active", b.id === `mode-${mode}`);
  document.getElementById("mode-hint").textContent = HINTS[mode];
  document.getElementById("mark-opts").classList.toggle("hidden", mode !== "mark");
  state.dartPending = null;
  state.markPending = null;
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

for (const mode of ["orbit", "seam", "notch", "grain", "measure", "dart", "mark"])
  document.getElementById(`mode-${mode}`).onclick = () => setMode(mode);
document.getElementById("show-edges").onchange = e => {
  if (wireframe) wireframe.visible = e.target.checked;
};
document.getElementById("apply-scale").onclick = () => {
  const factor = Number(document.getElementById("scale-factor").value);
  applyScale(factor).catch(e => setStatus(e.message, true));
};
for (const b of document.querySelectorAll(".preset-scale"))
  b.onclick = () => {
    document.getElementById("scale-factor").value = b.dataset.f;
    applyScale(Number(b.dataset.f)).catch(e => setStatus(e.message, true));
  };
document.getElementById("finish-seam").onclick = finishSeam;
document.getElementById("undo-leg").onclick = undoLeg;
document.getElementById("preview-split").onclick = () => previewSplit().catch(e => setStatus(e.message, true));
document.getElementById("generate").onclick = () => generate().catch(e => setStatus(e.message, true));
document.getElementById("save-spec").onclick = () => saveSpec().catch(e => setStatus(e.message, true));
document.getElementById("close-preview").onclick = () =>
  document.getElementById("svg-preview").classList.add("hidden");
document.getElementById("clear-grain").onclick = clearGrain;
document.getElementById("reset-mesh").onclick = () => resetMesh().catch(e => setStatus(e.message, true));
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
  selectPanel, toggleNotch, setMode, buildSpec, clearGrain, resetMesh,
  measureClick, applyScale, dartClick, markClick,
  setStraightCut: on => { document.getElementById("straight-cut").checked = on; },
  ready: () => !!displayMesh,
};
