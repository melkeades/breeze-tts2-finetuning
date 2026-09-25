const $ = (id) => document.getElementById(id);

const ui = {
  form: $("speechForm"), text: $("text"), instruction: $("instruction"), refAudio: $("refAudio"),
  refText: $("refText"), cfgScale: $("cfgScale"), cfgValue: $("cfgValue"), seed: $("seed"),
  generate: $("generateButton"), adapter: $("adapterSelect"), reload: $("reloadButton"),
  adapterState: $("adapterState"), adapterDescription: $("adapterDescription"), reloadProgress: $("reloadProgress"),
  reloadStage: $("reloadStage"), reloadEta: $("reloadEta"), reloadBar: $("reloadBar"),
  streamCard: $("streamCard"), streamLabel: $("streamLabel"), streamStatus: $("streamStatus"), streamClock: $("streamClock"),
  ttfa: $("ttfaValue"), rtf: $("rtfValue"), duration: $("durationValue"), render: $("renderValue"),
  rtfNote: $("rtfNote"), chunkNote: $("chunkNote"), silenceNote: $("silenceNote"),
  waveform: $("waveform"), player: $("audioPlayer"), download: $("downloadButton"), history: $("historyTable"),
};

let currentAbort = null;
let resultUrl = null;
let resultBlob = null;
let audioContext = null;
let nextPlayAt = 0;
let history = [];
let clockTimer = null;
let adapterCatalog = [];

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.className = "toast", 3600);
}

function formatClock(ms) {
  const total = Math.max(0, ms) / 1000;
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  return `${minutes}:${(total % 60).toFixed(1).padStart(4, "0")}`;
}

function setRuntime(status, fast = null) {
  $("runtimeStatus").textContent = status;
  $("statusDot").className = `status-dot ${status === "Ready" ? "ready" : status === "Offline" ? "error" : ""}`;
  if (fast !== null) $("fastPath").textContent = fast ? "Fast path · warmed" : "Eager path";
}

async function refreshHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.status || "Loading");
    setRuntime("Ready", data.fast_path);
  } catch (error) {
    setRuntime(error.message === "loading" ? "Loading weights" : "Offline");
  }
}

async function loadAdapters() {
  try {
    const response = await fetch("/v1/adapters", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load adapter catalog");
    const data = await response.json();
    adapterCatalog = data.adapters;
    ui.adapter.innerHTML = "";
    for (const adapter of adapterCatalog) {
      const option = document.createElement("option");
      option.value = adapter.id;
      option.textContent = adapter.name;
      option.selected = adapter.active;
      option.disabled = adapter.description.startsWith("Invalid adapter:");
      ui.adapter.append(option);
    }
    ui.adapter.disabled = false;
    ui.adapterState.textContent = "Weights ready";
    updateAdapterDescription();
  } catch (error) {
    ui.adapterState.textContent = "Catalog unavailable";
    toast(error.message, true);
  }
}

function updateAdapterDescription() {
  const selected = adapterCatalog.find((item) => item.id === ui.adapter.value);
  ui.adapterDescription.textContent = selected?.description || "Select a compatible LoRA adapter.";
  ui.reload.disabled = !selected || selected.active;
}

async function reloadWeights() {
  ui.reload.disabled = true;
  ui.generate.disabled = true;
  ui.adapter.disabled = true;
  ui.reloadProgress.classList.remove("hidden");
  ui.reloadBar.style.width = "1%";
  setRuntime("Loading weights");
  try {
    const response = await fetch("/v1/adapters/reload", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ adapter_id: ui.adapter.value }),
    });
    const start = await response.json();
    if (!response.ok) throw new Error(start.detail || "Could not start weight reload");
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const poll = await fetch(`/v1/adapters/reload/${start.id}`, { cache: "no-store" });
      const job = await poll.json();
      if (!poll.ok) throw new Error(job.detail || "Lost reload progress");
      ui.reloadStage.textContent = job.stage;
      ui.reloadBar.style.width = `${Math.max(1, job.progress * 100)}%`;
      ui.reloadEta.textContent = job.status === "ready" ? `${job.elapsed_seconds.toFixed(1)}s total` : job.eta_seconds == null ? "Estimating…" : `~${Math.ceil(job.eta_seconds)}s remaining`;
      if (job.status === "failed") throw new Error(job.error || "Weight reload failed");
      if (job.status === "ready") break;
    }
    toast("Adapter weights are ready.");
    await Promise.all([loadAdapters(), refreshHealth()]);
  } catch (error) {
    toast(error.message, true);
    await refreshHealth();
  } finally {
    ui.adapter.disabled = false;
    ui.generate.disabled = false;
    updateAdapterDescription();
  }
}

function concatBytes(chunks) {
  const size = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const output = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.byteLength; }
  return output;
}

function pcmBytesToFloat(bytes) {
  const length = Math.floor(bytes.byteLength / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, length * 2);
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
  return samples;
}

function percentile(values, percentileValue) {
  const sorted = new Float32Array(values);
  sorted.sort();
  if (!sorted.length) return 0;
  const rank = (sorted.length - 1) * percentileValue / 100;
  const lower = Math.floor(rank), upper = Math.ceil(rank);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
}

function firstDenseOnset(active, frameMs, windowMs, activeRatio, minActiveMs) {
  const window = Math.max(1, Math.round(windowMs / frameMs));
  const minFrames = Math.max(1, Math.round(minActiveMs / frameMs));
  for (let center = 0; center < active.length; center++) {
    const left = Math.max(0, center - Math.floor(window / 2));
    const right = Math.min(active.length, center + Math.ceil(window / 2));
    let count = 0;
    for (let i = left; i < right; i++) if (active[i]) count++;
    if (count / window < activeRatio) continue;
    let run = 0;
    for (let i = left; i < right; i++) {
      run = active[i] ? run + 1 : 0;
      if (run >= minFrames) return i - run + 1;
    }
  }
  return null;
}

function detectVoiceOnset(samples, sampleRate) {
  if (!samples.length) throw new Error("The model returned empty audio");
  const centered = new Float32Array(samples.length);
  const median = percentile(samples, 50);
  for (let i = 0; i < samples.length; i++) centered[i] = samples[i] - median;
  const frameMs = 2;
  const frame = Math.max(1, Math.round(sampleRate * frameMs / 1000));
  const frameCount = Math.max(1, Math.floor(centered.length / frame));
  const envelopes = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    let energy = 0;
    const start = f * frame;
    const end = Math.min(start + frame, centered.length);
    for (let i = start; i < end; i++) energy += centered[i] * centered[i];
    envelopes[f] = Math.sqrt(energy / Math.max(1, end - start));
  }
  const speechLevel = percentile(envelopes, 95);
  const highThreshold = Math.max(0.0005, speechLevel * 0.030);
  const lowThreshold = Math.max(0.0001, speechLevel * 0.005);
  const high = Array.from(envelopes, (value) => value >= highThreshold);
  const low = Array.from(envelopes, (value) => value >= lowThreshold);
  const highOnset = firstDenseOnset(high, frameMs, 10, .60, 4);
  const lowOnset = firstDenseOnset(low, frameMs, 20, .80, 10);
  if (highOnset === null && lowOnset === null) throw new Error("No audible voice onset was detected");
  const candidates = [highOnset, lowOnset].filter((value) => value !== null);
  if (highOnset !== null) {
    let start = highOnset, gap = 0;
    const maxGap = Math.round(12 / frameMs);
    for (let i = highOnset; i >= 0; i--) {
      if (envelopes[i] >= lowThreshold) { start = i; gap = 0; }
      else if (++gap > maxGap) break;
    }
    candidates.push(start);
  }
  const onsetSample = Math.min(...candidates) * frame;
  return { onsetSample, onsetMs: onsetSample * 1000 / sampleRate };
}

function wavBlob(pcmBytes, sampleRate) {
  const buffer = new ArrayBuffer(44 + pcmBytes.byteLength);
  const view = new DataView(buffer);
  const write = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + pcmBytes.byteLength, true); write(8, "WAVE"); write(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, "data");
  view.setUint32(40, pcmBytes.byteLength, true); new Uint8Array(buffer, 44).set(pcmBytes);
  return new Blob([buffer], { type: "audio/wav" });
}

async function playChunk(bytes, sampleRate) {
  if (!$("livePlayback").checked || bytes.byteLength < 2) return;
  audioContext ||= new AudioContext();
  if (audioContext.state === "suspended") await audioContext.resume();
  const samples = pcmBytesToFloat(bytes);
  if (!samples.length) return;
  const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer; source.connect(audioContext.destination);
  nextPlayAt = Math.max(nextPlayAt, audioContext.currentTime + .025);
  source.start(nextPlayAt); nextPlayAt += buffer.duration;
}

function drawWaveform(samples) {
  const canvas = ui.waveform, ctx = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height, middle = height / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,.045)"; ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 48) { ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, height - 20); ctx.stroke(); }
  if (!samples?.length) {
    ctx.strokeStyle = "rgba(144,154,168,.28)"; ctx.beginPath(); ctx.moveTo(0, middle); ctx.lineTo(width, middle); ctx.stroke(); return;
  }
  const gradient = ctx.createLinearGradient(0, 0, width, 0); gradient.addColorStop(0, "#66f6c3"); gradient.addColorStop(1, "#5bd9f7");
  ctx.strokeStyle = gradient; ctx.lineWidth = 1.6; ctx.beginPath();
  const block = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x++) {
    let peak = 0; const start = x * block;
    for (let i = start; i < Math.min(start + block, samples.length); i++) peak = Math.max(peak, Math.abs(samples[i]));
    const amplitude = Math.max(1, peak * (height * .42));
    ctx.moveTo(x, middle - amplitude); ctx.lineTo(x, middle + amplitude);
  }
  ctx.stroke();
}

function resetMetrics() {
  for (const node of [ui.ttfa, ui.rtf, ui.duration, ui.render]) node.textContent = "—";
  ui.rtfNote.textContent = "Render time ÷ audio length"; ui.chunkNote.textContent = "Awaiting a run"; ui.silenceNote.textContent = "Includes network delivery";
}

function renderHistory() {
  if (!history.length) { ui.history.innerHTML = '<div class="empty-history">Completed generations will appear here.</div>'; return; }
  ui.history.innerHTML = history.map((run) => `<div class="history-row"><span class="run-copy">${escapeHtml(run.text)}</span><span><b>${run.ttfa.toFixed(0)}</b> ms TTFA</span><span><b>${run.rtf.toFixed(2)}</b> RTF</span><span>${run.duration.toFixed(1)}s audio</span><span>${escapeHtml(run.adapter)}</span></div>`).join("");
}

function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value; return node.innerHTML; }

async function generateSpeech(event) {
  event.preventDefault();
  if (currentAbort) { currentAbort.abort(); return; }
  const file = ui.refAudio.files[0];
  if (Boolean(file) !== Boolean(ui.refText.value.trim())) { toast("Reference audio and its exact transcript must be provided together.", true); return; }
  if (!ui.text.value.trim()) return;

  currentAbort = new AbortController();
  const started = performance.now();
  const chunks = [], arrivals = [];
  let cumulativeBytes = 0, readCount = 0, firstChunkMs = null, carry = null;
  ui.generate.querySelector("span").textContent = "Stop generation";
  ui.generate.querySelector("i").textContent = "■";
  ui.streamCard.classList.add("streaming"); ui.streamLabel.textContent = "LIVE"; ui.streamStatus.textContent = "Waiting for the first audible frame…";
  resetMetrics(); drawWaveform(null); ui.player.removeAttribute("src"); ui.download.disabled = true;
  clearInterval(clockTimer); clockTimer = setInterval(() => ui.streamClock.textContent = formatClock(performance.now() - started), 100);
  if (resultUrl) URL.revokeObjectURL(resultUrl); resultUrl = null; resultBlob = null;
  nextPlayAt = audioContext?.currentTime || 0;

  try {
    const form = new FormData();
    form.append("text", ui.text.value.trim()); form.append("instruction", ui.instruction.value.trim());
    form.append("cfg_scale", ui.cfgScale.value); form.append("seed", ui.seed.value || "42");
    if (file) { form.append("ref_audio", file); form.append("ref_text", ui.refText.value.trim()); }
    const response = await fetch("/v1/audio/speech", { method: "POST", body: form, signal: currentAbort.signal });
    if (!response.ok) { const detail = await response.json().catch(() => ({})); throw new Error(detail.detail || `Generation failed (${response.status})`); }
    const sampleRate = Number(response.headers.get("X-Sample-Rate") || 24000);
    const activeAdapter = response.headers.get("X-Adapter-Id") || ui.adapter.value;
    const reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      const atMs = performance.now() - started;
      if (firstChunkMs === null) firstChunkMs = atMs;
      chunks.push(value); cumulativeBytes += value.byteLength; readCount++; arrivals.push({ bytes: cumulativeBytes, atMs });
      ui.streamStatus.textContent = `${(cumulativeBytes / 2 / sampleRate).toFixed(1)}s of audio received`;
      let playable = value;
      if (carry !== null) { const joined = new Uint8Array(value.byteLength + 1); joined[0] = carry; joined.set(value, 1); playable = joined; carry = null; }
      if (playable.byteLength % 2) { carry = playable[playable.byteLength - 1]; playable = playable.subarray(0, playable.byteLength - 1); }
      await playChunk(playable, sampleRate);
    }
    const finished = performance.now();
    const pcm = concatBytes(chunks);
    const samples = pcmBytesToFloat(pcm);
    const duration = samples.length / sampleRate;
    if (!duration || firstChunkMs === null) throw new Error("The model completed without audio");
    const onset = detectVoiceOnset(samples, sampleRate);
    const requiredBytes = (onset.onsetSample + 1) * 2;
    const available = arrivals.find((mark) => mark.bytes >= requiredBytes)?.atMs ?? firstChunkMs + onset.onsetMs;
    const ttfa = Math.max(firstChunkMs + onset.onsetMs, available);
    const elapsed = (finished - started) / 1000;
    const rtf = elapsed / duration;

    ui.ttfa.textContent = ttfa.toFixed(0); ui.rtf.textContent = rtf.toFixed(2); ui.duration.textContent = duration.toFixed(1); ui.render.textContent = elapsed.toFixed(1);
    ui.rtfNote.textContent = rtf < 1 ? `${(1 / rtf).toFixed(1)}× faster than real time` : `${rtf.toFixed(1)}× slower than real time`;
    ui.chunkNote.textContent = `${readCount} browser chunks · ${sampleRate / 1000} kHz mono`;
    ui.silenceNote.textContent = `${onset.onsetMs.toFixed(0)}ms detected initial silence`;
    ui.streamLabel.textContent = "READY"; ui.streamStatus.textContent = "Generation complete — replay or download below."; ui.streamClock.textContent = formatClock(finished - started);
    resultBlob = wavBlob(pcm, sampleRate); resultUrl = URL.createObjectURL(resultBlob); ui.player.src = resultUrl; ui.download.disabled = false;
    drawWaveform(samples);
    const adapterName = adapterCatalog.find((item) => item.id === activeAdapter)?.name || activeAdapter;
    history.unshift({ text: ui.text.value.trim(), ttfa, rtf, duration, adapter: adapterName }); history = history.slice(0, 6); renderHistory();
  } catch (error) {
    if (error.name === "AbortError") { ui.streamStatus.textContent = "Generation stopped."; toast("Generation stopped."); }
    else { ui.streamStatus.textContent = error.message; toast(error.message, true); }
    ui.streamLabel.textContent = "IDLE";
  } finally {
    clearInterval(clockTimer); currentAbort = null; ui.streamCard.classList.remove("streaming");
    ui.generate.querySelector("span").textContent = "Generate speech"; ui.generate.querySelector("i").textContent = "→";
  }
}

ui.form.addEventListener("submit", generateSpeech);
ui.text.addEventListener("input", () => $("charCount").textContent = ui.text.value.length);
ui.cfgScale.addEventListener("input", () => ui.cfgValue.textContent = Number(ui.cfgScale.value).toFixed(1));
ui.refAudio.addEventListener("change", () => $("fileName").textContent = ui.refAudio.files[0]?.name || "Choose a clean voice sample");
ui.adapter.addEventListener("change", updateAdapterDescription);
ui.reload.addEventListener("click", reloadWeights);
ui.download.addEventListener("click", () => { if (!resultBlob) return; const anchor = document.createElement("a"); anchor.href = resultUrl; anchor.download = `breeze-${Date.now()}.wav`; anchor.click(); });
$("clearButton").addEventListener("click", () => { resetMetrics(); drawWaveform(null); ui.player.removeAttribute("src"); ui.streamLabel.textContent = "IDLE"; ui.streamStatus.textContent = "Ready for a new performance."; ui.streamClock.textContent = "00:00.0"; ui.download.disabled = true; });
$("clearHistory").addEventListener("click", () => { history = []; renderHistory(); });

$("charCount").textContent = ui.text.value.length;
drawWaveform(null);
Promise.all([refreshHealth(), loadAdapters()]);
