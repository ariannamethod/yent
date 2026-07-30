const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-=+*#@%&';
const seedWords = 'Yent DoE Janus parliament notorch prophecy debt consensus memory limpha identity boundary'.split(' ');
const interfaceDeps = globalThis.YentInterfaceDeps;
if (!interfaceDeps) throw new Error('YentInterfaceDeps helper missing');
const deps = interfaceDeps.load({ transcript: true });
const interfaceSession = deps.interfaceSession;
const interfaceRestore = deps.interfaceRestore;
const chatStream = deps.chatStream;
const interfaceText = deps.interfaceText;
const tokenTelemetry = deps.tokenTelemetry;
const interfaceState = deps.interfaceState;
const interfaceClock = deps.interfaceClock;
const interfaceStatus = deps.interfaceStatus;
const interfaceOutput = deps.interfaceOutput;
const interfaceTranscript = deps.interfaceTranscript;
const interfaceHud = deps.interfaceHud;
const interfaceReplay = deps.interfaceReplay;
const interfaceInput = deps.interfaceInput;
const interfaceEvents = deps.interfaceEvents;
const interfaceSubmit = deps.interfaceSubmit;
const interfaceOutcome = deps.interfaceOutcome;
const interfaceRun = deps.interfaceRun;
const interfaceBoot = deps.interfaceBoot;
const interfaceAnimation = deps.interfaceAnimation;
const interfaceCanvas = deps.interfaceCanvas;
const interfaceStyle = deps.interfaceStyle;
const inputControls = interfaceInput.bindControls();
const composer = inputControls.composer;
const promptInput = inputControls.promptInput;
const sendButton = inputControls.sendButton;
const transcript = interfaceOutput.bind({ id: 'transcript' });
const clamp = deps.interfaceMath.clamp;
const mix = deps.interfaceMath.mix;
const fieldSurface = interfaceCanvas.bind({
  id: 'field',
  contextOptions: { alpha: false }
});
const field = fieldSurface.canvas;
const ctx = fieldSurface.context;
const traceSurface = interfaceCanvas.bind({ id: 'trace' });
const trace = traceSurface.canvas;
const tctx = traceSurface.context;
const maskSurface = interfaceCanvas.createScratch({
  contextOptions: { willReadFrequently: true }
});
const mask = maskSurface.canvas;
const mctx = maskSurface.context;
const generationRun = interfaceRun.create({ button: sendButton });
const replayRequest = interfaceReplay.request();
const replayMode = replayRequest.enabled;
const sessionReceipt = interfaceSession.createAdapter({ replayMode });
const hud = interfaceHud.bind();
const statusLabels = interfaceStatus.bind({ ids: { run: 'run-state' } });
const fonts = interfaceStyle.create();
const animationFrame = interfaceAnimation.create();
const tokenClock = interfaceClock.create();
const state = interfaceState.create({
  velocity: 1.2,
  sidePulse: 0.0
});

let width = 0;
let height = 0;
let dpr = 1;
let particles = [];
let bursts = [];
let traceHistory = [];
let tokenTape = seedWords.join('');
let latentTape = seedWords.join('');
let messages = [];
let visibleMessages = [];
let mouseX = -9999;
let mouseY = -9999;
let smoothX = 0;
let smoothY = 0;
let time = 0;

function candidateTapeText(data) {
  return tokenTelemetry.candidateText(data, {
    limit: 10,
    wordsPerToken: 1,
    sanitizer: interfaceText.tokenTapeText
  });
}

function sceneMetrics() {
  const desktop = width >= 980;
  const scale = desktop ? clamp(width / 1500, 1.04, 1.34) : (width < 620 ? 0.78 : 0.95);
  return {
    x: desktop ? clamp(width * 0.36, 330, width * 0.43) : width * 0.5,
    y: desktop ? height * 0.48 : height * 0.43,
    scale,
    sep: scale * (desktop ? 118 : 82)
  };
}

function drawJanusMask(anchorX, anchorY, scale) {
  mctx.save();
  mctx.translate(anchorX, anchorY);
  mctx.scale(scale, scale);

  const grad = mctx.createRadialGradient(0, -24, 12, 0, 4, 152);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.58, 'rgba(170,170,170,0.95)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  mctx.fillStyle = grad;

  mctx.beginPath();
  mctx.moveTo(0, -154);
  mctx.bezierCurveTo(86, -150, 120, -67, 112, 12);
  mctx.bezierCurveTo(106, 86, 58, 153, 0, 164);
  mctx.bezierCurveTo(-58, 153, -106, 86, -112, 12);
  mctx.bezierCurveTo(-120, -67, -86, -150, 0, -154);
  mctx.fill();

  mctx.fillStyle = 'rgba(255,255,255,0.33)';
  mctx.beginPath();
  mctx.moveTo(0, -142);
  mctx.bezierCurveTo(18, -92, 14, -38, 4, 5);
  mctx.bezierCurveTo(18, 28, 16, 54, 0, 78);
  mctx.bezierCurveTo(-16, 54, -18, 28, -4, 5);
  mctx.bezierCurveTo(-14, -38, -18, -92, 0, -142);
  mctx.fill();

  mctx.globalCompositeOperation = 'destination-out';
  mctx.fillStyle = 'rgba(0,0,0,0.78)';
  mctx.beginPath();
  mctx.ellipse(-43, -32, 28, 12, 0.16, 0, Math.PI * 2);
  mctx.ellipse(43, -32, 28, 12, -0.16, 0, Math.PI * 2);
  mctx.fill();
  mctx.beginPath();
  mctx.ellipse(0, 55, 43, 10, 0, 0, Math.PI * 2);
  mctx.fill();
  mctx.beginPath();
  mctx.moveTo(-4, -138);
  mctx.bezierCurveTo(10, -84, -12, -32, 2, 11);
  mctx.bezierCurveTo(-10, 38, 11, 65, -2, 103);
  mctx.lineTo(9, 103);
  mctx.bezierCurveTo(-3, 66, 16, 37, 6, 9);
  mctx.bezierCurveTo(18, -33, 0, -86, 12, -138);
  mctx.closePath();
  mctx.fill();

  mctx.globalCompositeOperation = 'source-over';
  mctx.fillStyle = 'rgba(255,255,255,0.9)';
  mctx.beginPath();
  mctx.ellipse(-43, -33, 7, 4, 0.16, 0, Math.PI * 2);
  mctx.ellipse(43, -33, 7, 4, -0.16, 0, Math.PI * 2);
  mctx.fill();
  mctx.restore();
}

class Particle {
  constructor(localX, localY, side, brightness, eye) {
    this.localX = localX;
    this.localY = localY;
    this.side = side;
    this.brightness = brightness;
    this.eye = eye;
    const scene = sceneMetrics();
    this.x = scene.x + side * (120 + Math.random() * 260);
    this.y = height * 0.5 + (Math.random() - 0.5) * height * 0.8;
    this.vx = (Math.random() - 0.5) * 5;
    this.vy = (Math.random() - 0.5) * 5;
    this.phase = Math.random() * Math.PI * 2;
    this.mass = 0.6 + Math.random() * 1.6;
    this.charIndex = Math.floor(Math.random() * tokenTape.length);
    this.restlessness = 0.45 + Math.random() * 1.25;
    this.driftA = Math.random() * Math.PI * 2;
    this.driftB = Math.random() * Math.PI * 2;
    this.jitterX = (Math.random() - 0.5) * 20;
    this.jitterY = (Math.random() - 0.5) * 20;
  }

  target(now) {
    const scene = sceneMetrics();
    const torn = 1 - state.consensus;
    const splitSep = mix(0, scene.sep, torn);
    const anchorX = scene.x + this.side * splitSep;
    const anchorY = scene.y;
    const seamPull = this.side * torn * clamp(1 - Math.abs(this.localX) / (scene.scale * 115), 0, 1) * scene.scale * 16;
    const unrest = 0.32 + state.debt * 0.78 + torn * 0.56 + state.candidateTail * 0.35;
    const localRadius = Math.hypot(this.localX, this.localY);
    const edge = clamp(localRadius / (scene.scale * 155), 0, 1);
    const drift = Math.sin(now * 0.7 + this.phase) * (3 + state.debt * 14) * (0.45 + torn);
    const skitterX =
      Math.sin(now * (0.95 + this.restlessness * 0.32) + this.driftA) * this.jitterX * unrest +
      Math.cos(now * (1.47 + this.restlessness * 0.21) + this.driftB) * scene.scale * (2 + edge * 8) * unrest;
    const skitterY =
      Math.cos(now * (0.86 + this.restlessness * 0.28) + this.driftB) * this.jitterY * unrest +
      Math.sin(now * (1.31 + this.restlessness * 0.24) + this.driftA) * scene.scale * (2 + edge * 7) * unrest;
    return {
      x: anchorX + this.localX + seamPull + this.side * drift + skitterX,
      y: anchorY + this.localY + Math.cos(now * 0.5 + this.phase) * (3 + state.debt * 11) + skitterY
    };
  }

  update(now) {
    const t = this.target(now);
    const chaos = clamp(state.debt * 1.15 + (1 - state.consensus) * 0.45, 0, 1);
    const spring = 0.009 + state.consensus * 0.038 + (1 - chaos) * 0.014;

    this.vx += (t.x - this.x) * spring;
    this.vy += (t.y - this.y) * spring;

    const scene = sceneMetrics();
    const torn = 1 - state.consensus;
    const halfX = scene.x + this.side * mix(0, scene.sep, torn);
    const halfY = scene.y;
    const hx = this.x - halfX;
    const hy = this.y - halfY;
    const hd = Math.max(22, Math.hypot(hx, hy));
    const vortex = (0.010 + torn * 0.018 + state.debt * 0.015) * this.restlessness;
    this.vx += (-hy / hd) * vortex * scene.scale * 34 * this.side;
    this.vy += (hx / hd) * vortex * scene.scale * 34 * this.side;

    const staticNoise = 0.75 + chaos * 2.1 + torn * 0.9;
    this.vx += (Math.random() - 0.5) * state.velocity * staticNoise;
    this.vy += (Math.random() - 0.5) * state.velocity * staticNoise;

    const dx = this.x - mouseX;
    const dy = this.y - mouseY;
    const dist = Math.max(1, Math.hypot(dx, dy));
    const repel = 108 + state.debt * 70;
    if (dist < repel && mouseX > -9000) {
      const f = Math.pow((repel - dist) / repel, 2);
      this.vx += (dx / dist) * f * 1.7 / this.mass;
      this.vy += (dy / dist) * f * 1.7 / this.mass;
    }

    for (const b of bursts) {
      const bx = this.x - b.x;
      const by = this.y - b.y;
      const bd = Math.max(1, Math.hypot(bx, by));
      const ring = Math.abs(bd - b.radius);
      if (ring < 30) {
        const f = b.power * b.life * (1 - ring / 30);
        this.vx += (bx / bd) * f / this.mass;
        this.vy += (by / bd) * f / this.mass;
      }
    }

    const speed = Math.hypot(this.vx, this.vy);
    if (speed > 24) {
      this.vx = this.vx / speed * 24;
      this.vy = this.vy / speed * 24;
    }
    this.vx *= 0.86;
    this.vy *= 0.86;
    this.x += this.vx;
    this.y += this.vy;
  }

  draw() {
    const b = this.brightness / 255;
    const mergeGlow = state.consensus;
    const left = { r: 226, g: 100, b: 87 };
    const right = { r: 85, g: 167, b: 216 };
    const core = this.side < 0 ? left : right;
    let r = mix(core.r, 235, mergeGlow * 0.55);
    let g = mix(core.g, 230, mergeGlow * 0.45);
    let bl = mix(core.b, 210, mergeGlow * 0.35);

    if (state.debt > 0.68) {
      const wound = (state.debt - 0.68) / 0.32;
      r = mix(r, 255, wound * 0.35);
      g = mix(g, 190, wound * 0.2);
      bl = mix(bl, 110, wound * 0.35);
    }

    const alpha = clamp(b * (0.22 + state.field * 0.55 + state.consensus * 0.32), 0.04, 0.96);
    const sourceTape = this.side < 0 && latentTape.length > seedWords.join('').length ? latentTape : tokenTape;
    const ch = this.eye ? 'Y' : sourceTape[(this.charIndex + Math.floor(time * (5 + state.candidateTail * 5))) % sourceTape.length] || charset[this.charIndex % charset.length];

    ctx.fillStyle = this.eye
      ? `rgba(255,255,255,${clamp(alpha + 0.32, 0, 1)})`
      : `rgba(${r | 0},${g | 0},${bl | 0},${alpha.toFixed(3)})`;
    ctx.fillText(ch, this.x, this.y);
  }
}

function buildParticles() {
  mask.width = width;
  mask.height = height;
  mctx.setTransform(1, 0, 0, 1, 0, 0);
  mctx.clearRect(0, 0, width, height);

  const scene = sceneMetrics();
  const cy = scene.y;
  drawJanusMask(scene.x, cy, scene.scale);

  const data = mctx.getImageData(0, 0, width, height).data;
  particles = [];
  const step = width < 760 ? 12 : 9;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const i = (y * width + x) * 4;
      const bright = data[i];
      if (bright > 20 && Math.random() > 0.1) {
        const side = x < scene.x ? -1 : 1;
        particles.push(new Particle(x - scene.x, y - cy, side, bright, false));
      }
    }
  }

  for (let side of [-1, 1]) {
    for (let i = 0; i < 42; i++) {
      const lx = side * scene.scale * 43 + (Math.random() - 0.5) * scene.scale * 15;
      const ly = -scene.scale * 33 + (Math.random() - 0.5) * scene.scale * 8;
      particles.push(new Particle(lx, ly, side, 255, true));
    }
  }
}

function resize() {
  const size = interfaceCanvas.resize({
    surfaces: [
      { canvas: field, context: ctx },
      { canvas: trace, context: tctx, height: 22 }
    ]
  });
  dpr = size.dpr;
  width = size.width;
  height = size.height;
  const scene = sceneMetrics();
  smoothX = scene.x;
  smoothY = scene.y;
  buildParticles();
}

function addTurn(role, text) {
  return interfaceTranscript.appendTurn({
    container: transcript,
    interfaceOutput,
    role,
    text,
    labels: { user: 'OLEG', assistant: 'YENT' }
  });
}

function setStatus(text) {
  interfaceStatus.setText({ target: statusLabels.run, text });
}

function restoreInterfaceSession() {
  const restored = interfaceRestore.load({ sessionReceipt, replayMode });
  if (!restored) return;

  visibleMessages = restored.visibleMessages;
  interfaceTranscript.clear({ container: transcript, interfaceOutput });
  for (const msg of visibleMessages) addTurn(msg.role, msg.content);

  const tapeText = interfaceText.tokenTapeText(restored.combinedText);
  if (tapeText) tokenTape = (seedWords.join('') + ' ' + tapeText).slice(-900);
}

function pushBurst(power) {
  const scene = sceneMetrics();
  bursts.push({
    x: scene.x + (Math.random() - 0.5) * width * 0.12,
    y: scene.y + (Math.random() - 0.5) * height * 0.12,
    radius: 0,
    power,
    life: 1.0
  });
}

function absorbToken(token, data) {
  if (!token) return;
  const telemetry = tokenTelemetry.normalize(data);
  tokenTape = interfaceText.appendTape({ tape: tokenTape, text: token, limit: 900 });
  const latent = candidateTapeText(telemetry);
  if (latent) latentTape = (latentTape + latent + ' ').slice(-900);
  state.tokps = tokenClock.tick();

  if (telemetry.hasExperts) state.experts = telemetry.experts;
  if (telemetry.hasDebt) state.debt = telemetry.debt;
  else state.debt = clamp(state.debt * 0.985 + (/[?!]/.test(token) ? 0.035 : 0.004), 0, 1);

  if (telemetry.hasConsensus) state.consensus = telemetry.consensus;
  else state.consensus = clamp(state.consensus + 0.006 + (/[.!?]/.test(token) ? 0.026 : 0), 0, 1);

  if (telemetry.hasFieldHealth) state.field = telemetry.fieldHealth;
  else state.field = clamp(1.0 - state.debt * 0.38 + state.consensus * 0.08, 0, 1);

  const candidates = tokenTelemetry.applyCandidateState(state, telemetry, { hasCandidates: latent.length > 0 });
  state.sidePulse = clamp(
    state.sidePulse + candidates.candidateTail * 0.2 +
      (candidates.hasSelectedProb ? (1 - candidates.selectedProb) * 0.05 : 0),
    0,
    1
  );
  state.velocity = 0.75 + state.debt * 2.7 + (1 - state.consensus) * 1.2 + candidates.candidateTail * 0.8;
  if (/[.!?]/.test(token)) pushBurst(0.55);
}

function drawTrace() {
  const w = trace.width / dpr;
  const h = trace.height / dpr;
  tctx.clearRect(0, 0, w, h);
  if (traceHistory.length < 2) return;

  function line(key, color, scale, offset) {
    tctx.beginPath();
    tctx.strokeStyle = color;
    tctx.lineWidth = 1.5;
    const dx = w / Math.max(1, traceHistory.length - 1);
    traceHistory.forEach((p, i) => {
      const x = i * dx;
      const y = offset - p[key] * scale;
      if (i === 0) tctx.moveTo(x, y);
      else tctx.lineTo(x, y);
    });
    tctx.stroke();
  }

  line('debt', 'rgba(226,100,87,0.78)', 16, 20);
  line('consensus', 'rgba(85,167,216,0.9)', 17, 19);
  line('field', 'rgba(119,188,145,0.62)', 12, 16);
}

function updateHud() {
  interfaceHud.render({ hud, state, tokenTelemetry });
}

function drawFieldHaze(scene) {
  const torn = 1 - state.consensus;
  const pulse = Math.sin(time * 0.7) * 0.5 + 0.5;
  const leftX = scene.x - scene.sep * (0.64 + torn * 0.42) + Math.sin(time * 0.37) * 18;
  const rightX = scene.x + scene.sep * (0.64 + torn * 0.42) + Math.cos(time * 0.31) * 18;
  const topY = scene.y - scene.scale * 68 + Math.sin(time * 0.23) * 14;
  const lowY = scene.y + scene.scale * 72 + Math.cos(time * 0.29) * 18;

  let g = ctx.createRadialGradient(leftX, topY, 0, leftX, topY, Math.max(width, height) * 0.42);
  g.addColorStop(0, `rgba(79,54,48,${0.18 + torn * 0.05})`);
  g.addColorStop(0.42, 'rgba(21,24,31,0.15)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  g = ctx.createRadialGradient(rightX, lowY, 0, rightX, lowY, Math.max(width, height) * 0.45);
  g.addColorStop(0, `rgba(42,67,83,${0.17 + pulse * 0.04})`);
  g.addColorStop(0.48, 'rgba(18,24,31,0.13)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  g = ctx.createLinearGradient(scene.x - scene.sep, scene.y - scene.scale * 160, scene.x + scene.sep, scene.y + scene.scale * 160);
  g.addColorStop(0, 'rgba(226,100,87,0.035)');
  g.addColorStop(0.49, 'rgba(255,255,255,0)');
  g.addColorStop(0.51, 'rgba(255,255,255,0)');
  g.addColorStop(1, 'rgba(85,167,216,0.04)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.globalAlpha = 0.045;
  ctx.fillStyle = '#ffffff';
  const n = width < 760 ? 32 : 58;
  for (let i = 0; i < n; i++) {
    const x = (Math.sin(time * (0.15 + i * 0.002) + i * 12.9898) * 0.5 + 0.5) * width;
    const y = (Math.cos(time * (0.18 + i * 0.003) + i * 78.233) * 0.5 + 0.5) * height;
    ctx.fillRect(x, y, 1, 1);
  }
  ctx.restore();
}

function animate() {
  animationFrame.requestFrame({ callback: animate });
  time += 0.016;
  state.sidePulse *= 0.96;
  if (!generationRun.isRunning()) {
    state.candidateTail = mix(state.candidateTail, 0, 0.018);
    state.selectedProb = mix(state.selectedProb, 0, 0.018);
  }
  smoothX += (mouseX - smoothX) * 0.08;
  smoothY += (mouseY - smoothY) * 0.08;

  ctx.fillStyle = 'rgba(5,6,7,0.32)';
  ctx.fillRect(0, 0, width, height);

  const scene = sceneMetrics();
  drawFieldHaze(scene);

  for (let i = bursts.length - 1; i >= 0; i--) {
    const b = bursts[i];
    b.radius += 12 + state.debt * 5;
    b.life -= 0.026;
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(213,185,111,${Math.max(0, b.life * 0.42)})`;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (b.life <= 0) bursts.splice(i, 1);
  }

  ctx.font = `${width < 620 ? 10 : 11}px ${fonts.mono()}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const p of particles) {
    p.update(time);
    p.draw();
  }

  traceHistory.push({ debt: state.debt, consensus: state.consensus, field: state.field });
  if (traceHistory.length > 160) traceHistory.shift();
  drawTrace();
  updateHud();
}

async function generate(text) {
  let assistantBody = null;

  const submit = await interfaceSubmit.run({
    generationRun,
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
      setStatus('GENERATING');
      tokenClock.reset();
      state.debt = 0.42;
      state.consensus = 0.12;
      state.field = 0.86;
      state.velocity = 2.4;
      tokenTelemetry.resetCandidateState(state);
      state.sidePulse = 0.5;
      latentTape = seedWords.join('');
      pushBurst(2.4);
    },
    onUser: userTurn => {
      messages = userTurn.messages;
      visibleMessages = userTurn.visibleMessages;
      addTurn('user', text);
      assistantBody = addTurn('assistant', '');
    },
    onToken: (token, data, responseText) => {
      interfaceOutput.setTextAndScroll({
        target: assistantBody,
        text: responseText,
        scrollTarget: transcript
      });
      absorbToken(token, data);
    }
  });

  messages = submit.messages;
  visibleMessages = submit.visibleMessages;
  interfaceOutcome.handle({
    submit,
    handlers: {
      stopped: (_turn, result) => {
        setStatus(result.hasText ? 'STOPPED' : 'IDLE');
      },
      fault: (turn, result) => {
        setStatus('FAULT');
        interfaceOutput.setText({
          target: assistantBody,
          text: result.hasText
            ? `${turn.text}\n\n[stream fault: ${result.message}]`
            : `parliament unreachable: ${result.message}`
        });
      },
      complete: (_turn, result) => {
        setStatus(result.kind === 'empty' ? 'EMPTY' : 'COMPLETE');
        state.consensus = clamp(state.consensus + 0.16, 0, 1);
        state.debt = clamp(state.debt * 0.72, 0, 1);
      }
    }
  });
}

interfaceEvents.bindPointer({
  onMove: point => {
    mouseX = point.x;
    mouseY = point.y;
  },
  onLeave: () => {
    mouseX = -9999;
    mouseY = -9999;
  },
  onDown: point => {
    bursts.push({ x: point.x, y: point.y, radius: 0, power: 1.3, life: 1.0 });
  }
});

interfaceBoot.start({
  restore: restoreInterfaceSession,
  resize,
  composer,
  startAnimation: () => animationFrame.start({ callback: animate }),
  interfaceReplay,
  replayMode,
  replayRequest,
  promptInput,
  generationRun,
  generate
});
