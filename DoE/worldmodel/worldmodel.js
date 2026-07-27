const baseWords = (
  'yent janus doe parliament notorch field resonance debt drift identity boundary ' +
  'limpha memory evidence silence chosen rejected thought answer token tensor ' +
  'calendar dissonance birth origin consensus expert gate scar shadow wall ' +
  'probability manifested almost future present innerworld method arianna'
).split(/\s+/);
const interfaceDeps = window.YentInterfaceDeps;
if (!interfaceDeps) throw new Error('YentInterfaceDeps helper missing');
const deps = interfaceDeps.load({ worldGeometry: true });
const interfaceSession = deps.interfaceSession;
const interfaceRestore = deps.interfaceRestore;
const chatStream = deps.chatStream;
const interfaceText = deps.interfaceText;
const tokenTelemetry = deps.tokenTelemetry;
const interfaceState = deps.interfaceState;
const interfaceClock = deps.interfaceClock;
const interfaceStatus = deps.interfaceStatus;
const interfaceOutput = deps.interfaceOutput;
const interfaceHud = deps.interfaceHud;
const interfaceReplay = deps.interfaceReplay;
const interfaceInput = deps.interfaceInput;
const interfaceEvents = deps.interfaceEvents;
const interfaceSubmit = deps.interfaceSubmit;
const interfaceOutcome = deps.interfaceOutcome;
const worldGeometry = deps.worldGeometry;
const interfaceRun = deps.interfaceRun;
const interfaceBoot = deps.interfaceBoot;
const interfaceAnimation = deps.interfaceAnimation;
const interfaceCanvas = deps.interfaceCanvas;
const interfaceStyle = deps.interfaceStyle;
const inputControls = interfaceInput.bindControls(document);
const promptInput = inputControls.promptInput;
const composer = inputControls.composer;
const sendButton = inputControls.sendButton;
const manifestText = interfaceOutput.bind(document, 'manifest-text');
const clamp = deps.interfaceMath.clamp;
const mix = deps.interfaceMath.mix;
const fieldSurface = interfaceCanvas.bind({
  document,
  id: 'field',
  contextOptions: { alpha: false }
});
const canvas = fieldSurface.canvas;
const ctx = fieldSurface.context;
const generationRun = interfaceRun.create({ button: sendButton });
const replayRequest = interfaceReplay.request();
const replayMode = replayRequest.enabled;
const sessionReceipt = interfaceSession.createAdapter({ replayMode });
const hud = interfaceHud.bind(document);
const statusLabels = interfaceStatus.bind(document, {
  note: 'status-note',
  manifest: 'manifest-state',
  shell: 'manifest-shell'
});
const fonts = interfaceStyle.create();
const animationFrame = interfaceAnimation.create();
const tokenClock = interfaceClock.create({ minElapsedSeconds: 0.001 });

const state = interfaceState.create({
  cameraX: 0,
  cameraY: 0,
  cameraZ: 0,
  angle: 0,
  topologySeed: 0.37,
  topologyWarp: 0.0,
  pulse: 0,
  quake: 0,
  idle: 0
});

let dpr = 1;
let width = 0;
let height = 0;
let time = 0;
let lastFrame = tokenClock.now();
let chosenText = '';
let manifestWords = [];
let fieldWords = baseWords.slice();
let candidateCloud = [];
let messages = [];
let visibleMessages = [];
const keys = Object.create(null);
const geometry = worldGeometry.create({ seed: state.topologySeed });

const hash = worldGeometry.hash;
const textSeed = worldGeometry.textSeed;

function syncTopologyFromGeometry() {
  state.topologySeed = geometry.seed;
  state.topologyWarp = geometry.warp;
}

function resize() {
  const size = interfaceCanvas.resize({ canvas, context: ctx });
  dpr = size.dpr;
  width = size.width;
  height = size.height;
}

const cleanWords = interfaceText.cleanWords;

function rebuildManifest() {
  manifestWords = cleanWords(chosenText).slice(-90);
}

function candidateEntries(data) {
  return tokenTelemetry.candidateWords(data, { limit: 18, wordsPerToken: 2 }).map((entry, i) => ({
    word: entry.word,
    prob: entry.prob,
    logprob: entry.logprob,
    rank: entry.rank,
    seed: textSeed(`${entry.word}:${i}:${state.step}`)
  }));
}

function rememberCandidates(entries, tailMass) {
  const baseStep = state.step;
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i];
    const seed = (e.seed * 0.67 + textSeed(`${e.word}:${baseStep}:${i}`) * 0.33) % 1;
    candidateCloud.unshift({
      word: e.word,
      prob: e.prob,
      logprob: e.logprob,
      rank: e.rank,
      seed,
      side: hash(seed * 1009 + baseStep) < 0.5 ? -1 : 1,
      age: 0,
      life: clamp(0.46 + Math.sqrt(e.prob) * 1.6 + tailMass * 0.24, 0.42, 1.15)
    });
  }
  while (candidateCloud.length > 128) candidateCloud.pop();
}

function decayCandidateCloud(dt) {
  for (let i = candidateCloud.length - 1; i >= 0; i--) {
    const c = candidateCloud[i];
    c.age += dt;
    c.life *= Math.pow(0.986, dt * 60);
    if (c.life < 0.035) candidateCloud.splice(i, 1);
  }
}

function absorbToken(token, data) {
  if (!token) return;
  const telemetry = tokenTelemetry.normalize(data);
  chosenText += token;
  rebuildManifest();
  const words = cleanWords(token);
  for (const w of words) {
    fieldWords.unshift(w);
    if (fieldWords.length > 260) fieldWords.pop();
  }
  const alternatives = candidateEntries(telemetry);
  let insertAt = Math.min(fieldWords.length, Math.max(1, words.length + 1));
  for (const alt of alternatives) {
    fieldWords.splice(insertAt, 0, alt.word);
    insertAt++;
  }
  while (fieldWords.length > 280) fieldWords.pop();
  if (telemetry.hasStep) state.step = telemetry.step;
  else state.step++;
  state.pulse = 1;
  state.quake = clamp(state.quake + 0.2, 0, 1);
  worldGeometry.absorbToken(geometry, token, telemetry);
  syncTopologyFromGeometry();
  const candidates = tokenTelemetry.applyCandidateState(state, telemetry, { candidateCount: alternatives.length });
  rememberCandidates(alternatives, candidates.candidateTail);
  state.debt = telemetry.hasDebt ? telemetry.debt : clamp(state.debt * 0.985 + 0.006, 0, 1);
  state.consensus = telemetry.hasConsensus ? telemetry.consensus : clamp(state.consensus * 0.992 + 0.004, 0, 1);
  state.field = telemetry.hasFieldHealth ? telemetry.fieldHealth : clamp(state.field * 0.996 + 0.004, 0, 1);
  state.tokps = tokenClock.tick();
  if (telemetry.hasEntropy) {
    state.entropy = telemetry.entropy;
  } else {
    const diversity = new Set(fieldWords.slice(0, 80).map(w => w.toLowerCase())).size;
    state.entropy = Math.log(Math.max(1, diversity));
  }
}

function wordAt(i) {
  if (!fieldWords.length) return baseWords[i % baseWords.length];
  const j = Math.abs(i) % fieldWords.length;
  return fieldWords[j] || baseWords[j % baseWords.length];
}

function viewFrame() {
  const yaw = clamp(state.angle, -1.05, 1.05);
  return {
    yaw,
    sin: Math.sin(yaw),
    cos: Math.cos(yaw),
    horizon: height * 0.43,
    vanishX: width * 0.5 - Math.sin(yaw) * width * 0.32
  };
}

function drawBackground() {
  const g = ctx.createLinearGradient(0, 0, 0, height);
  g.addColorStop(0, '#fbfaf7');
  g.addColorStop(0.56, '#f6f3ec');
  g.addColorStop(1, '#ebe7dc');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.globalAlpha = 0.28;
  ctx.strokeStyle = '#d8d5cc';
  ctx.lineWidth = 1;
  const view = viewFrame();
  const horizon = view.horizon;
  const tilt = view.sin * height * 0.028;
  ctx.beginPath();
  ctx.moveTo(0, horizon + tilt);
  ctx.lineTo(width, horizon - tilt);
  ctx.stroke();

  for (let i = 0; i < 14; i++) {
    const y = mix(horizon + 22, height - 118, i / 13);
    const sway = view.sin * (18 + i * 5);
    ctx.globalAlpha = 0.08 + i * 0.004;
    ctx.beginPath();
    ctx.moveTo(-30, y + sway);
    ctx.lineTo(width + 30, y - sway * 0.35);
    ctx.stroke();
  }
  ctx.restore();
}

function projectWorld(worldX, depth, worldY) {
  const view = viewFrame();
  const x = worldX - state.cameraX;
  const viewX = x * view.cos - depth * view.sin * 0.74;
  const viewZ = Math.max(72, depth * view.cos + x * view.sin * 0.34);
  const scale = 900 / (900 + viewZ);
  return {
    x: view.vanishX + viewX * scale,
    y: view.horizon + (worldY - state.cameraY) * scale,
    scale,
    depth: viewZ,
    yaw: view.yaw
  };
}

function wallShape(side) {
  const params = worldGeometry.wallShapeParams(geometry, side);
  const nearOuter = projectWorld(side * params.nearOuterX, params.nearDepth, params.nearOuterY);
  const nearTop = projectWorld(side * params.nearTopX, params.nearDepth, params.nearTopY);
  const farTop = projectWorld(side * params.farTopX, params.farDepth, params.farTopY);
  const farBottom = projectWorld(side * params.farBottomX, params.farDepth, params.farBottomY);
  return [nearTop, farTop, farBottom, nearOuter];
}

function drawWallSurface(side) {
  const shape = wallShape(side);
  const stress = clamp(state.debt * 0.75 + (1 - state.consensus) * 0.35, 0, 1);
  const wake = state.pulse * 0.045;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(shape[0].x, shape[0].y);
  for (let i = 1; i < shape.length; i++) ctx.lineTo(shape[i].x, shape[i].y);
  ctx.closePath();
  ctx.clip();

  const horizon = viewFrame().horizon;
  ctx.strokeStyle = `rgba(216,213,204,${0.08 + stress * 0.045 + wake * 0.25})`;
  ctx.lineWidth = 1;
  for (let lane = 0; lane < 9; lane++) {
    const xw = side * (470 + lane * 86);
    const a = projectWorld(xw, 220, 455);
    const b = projectWorld(xw * 0.72, 3400, 250);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  for (let band = 0; band < 10; band++) {
    const depth = 320 + band * 310;
    const a = projectWorld(side * 440, depth, 440);
    const b = projectWorld(side * 1180, depth, 440);
    ctx.globalAlpha = 0.055 + wake * 0.25;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.textBaseline = 'middle';
  ctx.textAlign = side < 0 ? 'left' : 'right';
  const rows = 9 + Math.floor(hash(state.topologySeed + side * 2.1) * 4);
  const cols = 11 + Math.floor(hash(state.topologySeed + side * 3.4) * 4);
  const span = 3500;
  for (let c = 0; c < cols; c++) {
    const rawDepth = ((c * 285 + state.topologySeed * 480 - state.cameraZ * 0.85) % span + span) % span;
    const depth = 180 + rawDepth;
    const fadeNear = clamp((depth - 180) / 320, 0, 1);
    const fadeFar = clamp((span - rawDepth) / 620, 0, 1);
    const depthFade = fadeNear * fadeFar;
    if (depthFade <= 0.03) continue;

    for (let r = 0; r < rows; r++) {
      const lane = r % 5;
      const topo = state.topologySeed * 997 + side * 31;
      const wallX = side * (500 + lane * (108 + hash(topo + r) * 28) + hash(c * 41 + r * 7 + topo) * 58);
      const wallY = -132 + r * (48 + hash(topo + c) * 12) + Math.sin(time * 0.2 + c + r + topo) * stress * (5 + state.topologyWarp * 18);
      const p = projectWorld(wallX, depth, wallY);
      if (p.y < horizon - 125 || p.y > height - 105) continue;
      const k = Math.floor(hash(c * 97 + r * 37 + state.step * 0.13) * 190);
      const word = wordAt(k + c * 3 + r);
      const head = k < 10;
      const tail = k > 135;
      const fs = clamp(6.5 + p.scale * 12.5 + (head ? 1.8 : 0), 7, 18);
      const alpha = depthFade * (tail ? 0.22 : head ? 0.82 : 0.34 + p.scale * 0.35);
      const weight = head ? 700 : tail ? 350 : 470;
      ctx.font = `${weight} ${fs}px ${fonts.mono()}`;
      ctx.fillStyle = head
        ? `rgba(197,68,107,${alpha})`
        : `rgba(13,13,11,${alpha})`;
      ctx.fillText(word, p.x, p.y);
    }
  }
  ctx.restore();
}

function drawWalls() {
  drawWallSurface(-1);
  drawWallSurface(1);
}

function drawRejectedMass() {
  const view = viewFrame();
  const count = (width < 720 ? 42 : 88) + Math.floor(state.candidateTail * 42);
  const stress = clamp(0.35 + state.debt * 0.9 + (1 - state.consensus) * 0.5, 0, 1.6);
  const span = 3400;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < count; i++) {
    const topo = state.topologySeed * 4096;
    const rawDepth = ((hash(i * 19 + 3 + topo) * span - state.cameraZ * 0.62) % span + span) % span;
    const depth = 720 + rawDepth;
    const worldX = (hash(i * 31 + 7 + topo) - 0.5) * (960 + state.topologyWarp * 260) + Math.sin(time * 0.11 + i) * 70 * stress;
    const worldY = -250 + hash(i * 17 + 9 + topo) * (300 + state.topologyWarp * 140) + Math.cos(time * 0.17 + i) * 28 * stress;
    const p = projectWorld(worldX, depth, worldY);
    if (p.x < -80 || p.x > width + 80 || p.y < view.horizon - 190 || p.y > height - 130) continue;
    const word = wordAt(Math.floor(hash(i * 73 + state.step) * 140) + i);
    const depthFade = clamp((depth - 760) / 700, 0, 1) * clamp((3600 - depth) / 980, 0, 1);
    const alpha = depthFade * (0.06 + hash(i + 4) * 0.19);
    const fs = clamp(7 + p.scale * 18 + hash(i + 8) * 5, 8, 21);
    ctx.font = `${fs}px ${fonts.mono()}`;
    ctx.fillStyle = i % 7 === 0
      ? `rgba(71,122,168,${alpha})`
      : `rgba(73,72,67,${alpha})`;
    ctx.fillText(word, p.x, p.y);
  }

  for (let i = candidateCloud.length - 1; i >= 0; i--) {
    const c = candidateCloud[i];
    const seed = c.seed * 8192 + c.rank * 17;
    const rawDepth = ((c.seed * span + c.rank * 113 - state.cameraZ * 0.74) % span + span) % span;
    const depth = 620 + rawDepth;
    const orbit = time * (0.14 + c.rank * 0.008) + seed;
    const side = c.side || (hash(seed) < 0.5 ? -1 : 1);
    const worldX = side * (120 + hash(seed + 11) * (760 + state.topologyWarp * 210)) + Math.sin(orbit) * (24 + stress * 58);
    const worldY = -230 + hash(seed + 19) * (420 + state.topologyWarp * 120) + Math.cos(orbit * 0.83) * (18 + stress * 44);
    const p = projectWorld(worldX, depth, worldY);
    if (p.x < -120 || p.x > width + 120 || p.y < view.horizon - 210 || p.y > height - 110) continue;

    const depthFade = clamp((depth - 620) / 560, 0, 1) * clamp((span + 620 - depth) / 920, 0, 1);
    const probBoost = Math.sqrt(clamp(c.prob, 0, 1));
    const rankBoost = 1 / (1 + c.rank * 0.2);
    const alpha = depthFade * c.life * clamp(0.1 + probBoost * 1.35 + rankBoost * 0.18, 0, 0.84);
    if (alpha <= 0.025) continue;
    const fs = clamp(8 + p.scale * 26 + probBoost * 24 + rankBoost * 4, 8, 34);
    const weight = c.rank <= 2 ? 720 : c.rank <= 5 ? 610 : 470;
    ctx.font = `${weight} ${fs}px ${fonts.mono()}`;
    ctx.fillStyle = c.rank <= 2
      ? `rgba(197,68,107,${alpha})`
      : `rgba(71,122,168,${alpha * 0.76})`;
    ctx.fillText(c.word, p.x, p.y);
  }
  ctx.restore();
}

function drawManifestedAnswer() {
  const answerDepth = ((1180 + state.topologySeed * 620 - state.cameraZ * 0.55) % 2600 + 2600) % 2600 + 520;
  const anchor = projectWorld((state.topologySeed - 0.5) * 220, answerDepth, -40 + (hash(state.topologySeed + 8) - 0.5) * 90);
  const centerX = anchor.x;
  const centerY = anchor.y;
  const maxW = clamp(width * 0.54, 300, 820);
  const words = manifestWords.slice(-34);
  const pulse = state.pulse;
  const certainty = clamp(state.selectedProb * 3.2, 0, 1);

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  if (!words.length) {
    ctx.restore();
    return;
  }

  const fontSize = (width < 720 ? 22 : 32) * clamp(anchor.scale * 1.9, 0.62, 1.0);
  ctx.font = `650 ${fontSize}px ${fonts.serif()}`;

  const lines = [];
  let line = '';
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (ctx.measureText(next).width > maxW && line) {
      lines.push(line);
      line = w;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  const visible = lines.slice(-5);
  const lineH = fontSize * 1.34;
  const startY = centerY - (visible.length - 1) * lineH * 0.5;

  ctx.shadowColor = `rgba(197,68,107,${0.14 + pulse * 0.18 + certainty * 0.08})`;
  ctx.shadowBlur = 14 + pulse * 24 + state.candidateTail * 14;
  for (let i = 0; i < visible.length; i++) {
    const y = startY + i * lineH;
    const age = visible.length - 1 - i;
    ctx.fillStyle = `rgba(13,13,11,${clamp(0.34 + i * 0.16 + certainty * 0.12, 0, 0.96)})`;
    ctx.fillText(visible[i], centerX, y);
    if (age === 0) {
      const last = words[words.length - 1] || '';
      const xoff = ctx.measureText(visible[i]).width * 0.5 - ctx.measureText(last).width * 0.5;
      ctx.fillStyle = `rgba(197,68,107,${0.72 + pulse * 0.22})`;
      ctx.fillText(last, centerX + xoff, y);
    }
  }

  ctx.shadowBlur = 0;
  ctx.strokeStyle = `rgba(197,68,107,${0.2 + pulse * 0.28})`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(centerX - maxW * 0.34, startY + visible.length * lineH * 0.5 + 18);
  ctx.lineTo(centerX + maxW * 0.34, startY + visible.length * lineH * 0.5 + 18);
  ctx.stroke();
  ctx.restore();
}

function updateHud() {
  interfaceHud.render(hud, state, { tokenTelemetry });
}

function tickCamera(dt) {
  const speed = (keys.shift ? 520 : 260) * dt;
  const vertical = (keys.shift ? 360 : 190) * dt;
  const turn = (keys.shift ? 1.75 : 1.05) * dt;
  if (keys.w || keys.arrowup) state.cameraZ += speed;
  if (keys.s || keys.arrowdown) state.cameraZ -= speed;
  if (keys.a || keys.arrowleft) state.angle -= turn;
  if (keys.d || keys.arrowright) state.angle += turn;
  if (keys.q) state.cameraX -= speed * 0.8;
  if (keys.e) state.cameraX += speed * 0.8;
  if (keys.r || keys.pageup) state.cameraY += vertical;
  if (keys.f || keys.pagedown) state.cameraY -= vertical;
  state.angle = clamp(state.angle, -1.05, 1.05);
  state.cameraY = clamp(state.cameraY, -280, 280);
  state.cameraX *= Math.pow(0.93, dt * 60);
}

function animate(frameNow) {
  animationFrame.requestFrame(animate);
  const now = tokenClock.now(frameNow);
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  time += dt;
  state.idle += dt;
  state.pulse *= Math.pow(0.86, dt * 60);
  state.quake *= Math.pow(0.9, dt * 60);
  worldGeometry.decay(geometry, dt);
  syncTopologyFromGeometry();
  if (!generationRun.isRunning()) {
    state.debt = mix(state.debt, 0, 0.006);
    state.consensus = mix(state.consensus, 0.62, 0.004);
    state.tokps = mix(state.tokps, 0, 0.03);
    state.candidateTail = mix(state.candidateTail, 0, 0.01);
    state.selectedProb = mix(state.selectedProb, 0, 0.012);
  }
  decayCandidateCloud(dt);
  tickCamera(dt);
  drawBackground();
  drawWalls();
  drawRejectedMass();
  drawManifestedAnswer();
  updateHud();
}

function setStatus(text) {
  interfaceStatus.setText(statusLabels.note, text);
}

function setManifestState(text, active) {
  interfaceStatus.setManifest(statusLabels, text, active);
}

function setManifestText(text) {
  interfaceOutput.setTextAndScroll(manifestText, text);
}

function restoreInterfaceSession() {
  const restored = interfaceRestore.load({ sessionReceipt, replayMode });
  if (!restored) return;

  visibleMessages = restored.visibleMessages;
  const combined = restored.combinedText;
  const words = cleanWords(combined).slice(-120);
  if (words.length) {
    fieldWords.unshift(...words);
    fieldWords = fieldWords.slice(0, 260);
    worldGeometry.resetFromPrompt(geometry, combined);
    syncTopologyFromGeometry();
    state.entropy = Math.log(Math.max(1, new Set(words.map(w => w.toLowerCase())).size));
  }

  const lastAssistant = restored.lastAssistant;
  if (lastAssistant) {
    chosenText = lastAssistant.content;
    rebuildManifest();
    setManifestText(chosenText);
    setManifestState('RESTORED', true);
    setStatus('FIELD RESTORED.');
  } else {
    setManifestState('IDLE', false);
    setStatus('FIELD REMEMBERS PROMPT.');
  }
}

async function generate(text) {
  const submit = await interfaceSubmit.run({
    generationRun,
    document,
    interfaceInput,
    interfaceTurn: deps.interfaceTurn,
    chatStream,
    interfaceReplay,
    replayMode,
    replayRequest,
    sessionReceipt,
    messages,
    visibleMessages,
    text,
    beforeUser: () => {
      setStatus('FIELD DISTORTED.');
      setManifestState('GENERATING', true);
      setManifestText('');
      chosenText = '';
      manifestWords = [];
      tokenClock.reset();
      state.debt = 0.46;
      state.consensus = 0.16;
      state.field = 0.92;
      state.entropy = Math.max(state.entropy, 3.4);
      tokenTelemetry.resetCandidateState(state);
      candidateCloud = [];
      worldGeometry.resetFromPrompt(geometry, text);
      syncTopologyFromGeometry();
      state.cameraY = mix(state.cameraY, (state.topologySeed - 0.5) * 170, 0.22);
      fieldWords.unshift(...cleanWords(text).slice(0, 18));
      fieldWords = fieldWords.slice(0, 260);
    },
    onUser: userTurn => {
      messages = userTurn.messages;
      visibleMessages = userTurn.visibleMessages;
    },
    onToken: (token, data, responseText) => {
      setManifestText(responseText);
      absorbToken(token, data);
    }
  });

  messages = submit.messages;
  visibleMessages = submit.visibleMessages;
  interfaceOutcome.handle(submit, {
    stopped: (_turn, result) => {
      setStatus('MANIFESTATION STOPPED.');
      setManifestState(result.hasText ? 'STOPPED' : 'IDLE', result.hasText);
    },
    fault: (_turn, result) => {
      setStatus(`FIELD FAULT: ${result.message}`);
      setManifestState('FAULT', result.hasText);
      if (!result.hasText) setManifestText(`FIELD FAULT: ${result.message}`);
      fieldWords.unshift('fault', 'unreachable');
    },
    complete: (_turn, result) => {
      setStatus('FIELD SETTLED.');
      setManifestState(result.kind === 'empty' ? 'EMPTY' : 'COMPLETE', result.hasText);
      state.consensus = clamp(state.consensus + 0.18, 0, 1);
      state.debt = clamp(state.debt * 0.68, 0, 1);
    }
  });
}

interfaceEvents.bindKeyState({
  keys,
  ignore: () => interfaceInput.isFocused(document, promptInput)
});

interfaceBoot.start({
  restore: restoreInterfaceSession,
  resize,
  composer,
  startAnimation: () => animationFrame.start(animate),
  interfaceReplay,
  replayMode,
  replayRequest,
  promptInput,
  generationRun,
  generate
});
