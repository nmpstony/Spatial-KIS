const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const regionDefs = [
  ['all', 'all (Toàn khung hình)'], ['left', 'left (Bên trái)'], ['center', 'center (Chính giữa)'],
  ['right', 'right (Bên phải)'], ['top', 'top (Phía trên)'], ['bottom', 'bottom (Phía dưới)'],
  ['middle_row', 'middle_row (Hàng giữa)'], ['center_col', 'center_col (Cột giữa)'],
  ['top_left', 'top_left (Góc trên trái)'], ['top_center', 'top_center (Giữa trên)'],
  ['top_right', 'top_right (Góc trên phải)'], ['middle_left', 'middle_left (Giữa trái)'],
  ['middle_right', 'middle_right (Giữa phải)'], ['bottom_left', 'bottom_left (Góc dưới trái)'],
  ['bottom_center', 'bottom_center (Giữa dưới)'], ['bottom_right', 'bottom_right (Góc dưới phải)']
];
const REGION_NAMES = Object.fromEntries(regionDefs.map(([id, label]) => [id, label.split(' (')[0]]));

const scenesContainer = $('#scenes-container');
const results = $('#results');
const statusEl = $('#status');
const inspect = $('#inspect');
const player = $('#player');
const details = $('#details');
const dresStatus = $('#dres-status');
const csvQueryInput = $('#csv-query-input');
const asrQueryInput = $('#asr-query-input');
const csvResultsBar = $('#csv-results-bar');
const evidenceSection = $('#ocr-results-section');
const candidateBadge = $('#ocr-candidate-badge');
const resultCount = $('#result-count');

let currentInspectItem = null;
let segmentPlayback = false;
let lastVisualResults = [];
let lastVisualLatencyMs = 0;
let activeCombinedFilter = null;
let combineMode = localStorage.getItem('spatialKis.filterCombineMode') === 'or' ? 'or' : 'and';
let gridDensity = localStorage.getItem('spatialKis.gridDensity') === 'compact' ? 'compact' : 'comfortable';
let hideSubmitted = localStorage.getItem('spatialKis.hideSubmitted') === 'true';
let evidenceShowAll = false;
let visualSearchController = null;
let visualGeneration = 0;
let ocrAbortController = null;
let asrAbortController = null;
let neighborsAbortController = null;
let filterGeneration = 0;
const modalityCaches = { ocr: new Map(), asr: new Map() };
const submittedKeys = new Set();
const copiedKeys = new Set();
const inFlightSubmissions = new Set();

function cleanVideoStem(videoId) {
  if (!videoId) return '';
  return String(videoId).split('/').pop().split('\\').pop().replace(/\.mp4$/i, '');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

function formatTime(seconds, tenths = false) {
  const value = Math.max(0, Number(seconds) || 0);
  const mins = Math.floor(value / 60);
  const secs = value - mins * 60;
  return `${String(mins).padStart(2, '0')}:${secs.toFixed(tenths ? 1 : 2).padStart(tenths ? 4 : 5, '0')}`;
}

function itemMs(item) {
  return Math.max(0, Math.round(item.timestamp_ms != null ? Number(item.timestamp_ms) : Number(item.timestamp || 0) * 1000));
}

function imageUrl(path) {
  if (!path) return '';
  return `/media/keyframe/${String(path).replace(/^images[\\/]/, '').split('/').map(encodeURIComponent).join('/')}`;
}

function getVisualCandidateVideos() {
  return [...new Set(lastVisualResults.map((item) => (item.video_stem || cleanVideoStem(item.video_id)).toUpperCase()).filter(Boolean))];
}

function visualSignature() {
  return getVisualCandidateVideos().slice().sort().join(',');
}

function cacheKey(signature, query) {
  return `${signature}::${query.trim()}`;
}

function cacheSet(source, key, value) {
  const cache = modalityCaches[source];
  if (cache.has(key)) cache.delete(key);
  cache.set(key, value);
  while (cache.size > 30) cache.delete(cache.keys().next().value);
}

function showToast(message, tone = 'info') {
  const toast = $('#action-toast');
  const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (toast) {
    toast.textContent = `${stamp} · ${message}`;
    toast.dataset.tone = tone;
    toast.hidden = false;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { toast.hidden = true; }, 4500);
  }
  if (dresStatus) dresStatus.textContent = `${stamp} · ${message}`;
}

// Sidebar state and shortcuts.
const isMobile = () => window.matchMedia('(max-width: 960px)').matches;
function setSidebar(open) {
  if (isMobile()) {
    document.body.classList.toggle('mobile-sidebar-open', open);
    const scrim = $('#sidebar-scrim');
    if (scrim) scrim.hidden = !open;
  } else {
    document.body.classList.toggle('sidebar-open', open);
    document.body.classList.toggle('sidebar-collapsed', !open);
    localStorage.setItem('spatialKis.sidebarCollapsed', String(!open));
  }
  const expanded = isMobile() ? document.body.classList.contains('mobile-sidebar-open') : !document.body.classList.contains('sidebar-collapsed');
  ['#sidebar-toggle', '#sidebar-handle'].forEach((selector) => {
    const button = $(selector);
    if (!button) return;
    button.setAttribute('aria-expanded', String(expanded));
    button.setAttribute('aria-label', expanded ? 'Collapse filters' : 'Open filters');
  });
}

function toggleSidebar() {
  const open = isMobile() ? document.body.classList.contains('mobile-sidebar-open') : !document.body.classList.contains('sidebar-collapsed');
  setSidebar(!open);
}

let mobileLayout = isMobile();
if (mobileLayout) setSidebar(false);
else setSidebar(localStorage.getItem('spatialKis.sidebarCollapsed') !== 'true');
$('#sidebar-toggle')?.addEventListener('click', toggleSidebar);
$('#sidebar-handle')?.addEventListener('click', toggleSidebar);
$('#sidebar-scrim')?.addEventListener('click', () => setSidebar(false));
window.addEventListener('resize', () => {
  const nextMobileLayout = isMobile();
  if (nextMobileLayout === mobileLayout) return;
  mobileLayout = nextMobileLayout;
  if (mobileLayout) setSidebar(false);
  else setSidebar(localStorage.getItem('spatialKis.sidebarCollapsed') !== 'true');
});

// Backend and index status/upload.
async function checkServerHealth() {
  const badge = $('#backend-badge');
  if (!badge) return;
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    badge.textContent = `● Server ready · ${data.backend || 'active'}`;
    badge.className = 'status-pill ready';
  } catch (_) {
    badge.textContent = '● Server offline';
    badge.className = 'status-pill error';
  }
}

async function refreshIndexStatus(kind) {
  const badge = kind === 'ocr' ? $('#csv-meta-badge') : $('#asr-meta-badge');
  if (!badge) return;
  try {
    const response = await fetch(kind === 'ocr' ? '/api/csv/status' : '/api/asr/status');
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    badge.textContent = data.ready || data.total_rows > 0
      ? `${data.total_rows || data.loaded_rows || 0} rows · ${data.unique_videos || 0} videos · ${(data.files || []).length} files`
      : 'No index loaded';
    badge.className = `index-meta ${data.ready || data.total_rows > 0 ? 'ok' : ''}`;
    badge.title = (data.files || []).map((file) => file.name || String(file)).join(', ');
  } catch (error) {
    badge.textContent = `Error: ${error.message}`;
    badge.className = 'index-meta error';
  }
}

function setupUpload(kind) {
  const input = kind === 'ocr' ? $('#csv-file-input') : $('#asr-file-input');
  const button = kind === 'ocr' ? $('#csv-upload-btn') : $('#asr-upload-btn');
  const badge = kind === 'ocr' ? $('#csv-meta-badge') : $('#asr-meta-badge');
  const prefix = kind === 'ocr' ? 'ocr_' : 'asr_';
  if (!input || !button || !badge) return;
  button.addEventListener('click', () => input.click());
  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().startsWith(prefix) || !file.name.toLowerCase().endsWith('.csv')) {
      badge.textContent = `Upload failed: filename must match ${prefix}*.csv`;
      badge.className = 'index-meta error';
      input.value = '';
      return;
    }
    badge.textContent = `Uploading ${file.name}…`;
    badge.className = 'index-meta';
    try {
      const endpoint = kind === 'ocr' ? '/api/csv/upload' : '/api/asr/upload';
      const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: file.name, content: await file.text() }) });
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`${response.status} ${body.slice(0, 120)}`);
      }
      await refreshIndexStatus(kind);
      showToast(`${kind.toUpperCase()} index uploaded: ${file.name}`, 'success');
    } catch (error) {
      badge.textContent = `Upload failed: ${error.message}`;
      badge.className = 'index-meta error';
    } finally {
      input.value = '';
    }
  });
}

checkServerHealth();
refreshIndexStatus('ocr');
refreshIndexStatus('asr');
setupUpload('ocr');
setupUpload('asr');

// DRES state, authentication, persistence, and submissions.
const dresUrlInput = $('#dres-url');
const dresUserInput = $('#dres-user');
const dresPassInput = $('#dres-pass');
const dresSessionInput = $('#dres-session');
const dresEvalSelect = $('#dres-eval-select');
const dresEvalIdInput = $('#dres-eval-id');
const qaAnswerInput = $('#qa-answer');

function submissionStorageKey() {
  return `spatialKis.submitted.${encodeURIComponent(dresSessionInput?.value.trim() || 'offline')}.${encodeURIComponent(dresEvalIdInput?.value.trim() || 'none')}`;
}

function loadSubmittedMarkers() {
  submittedKeys.clear();
  try {
    const stored = JSON.parse(localStorage.getItem(submissionStorageKey()) || '[]');
    stored.forEach((key) => submittedKeys.add(String(key)));
  } catch (_) {}
  renderVisualResults();
}

function persistSubmittedMarkers() {
  localStorage.setItem(submissionStorageKey(), JSON.stringify([...submittedKeys].slice(-1000)));
}

function saveDresStorage() {
  if (dresUrlInput) localStorage.setItem('dres_url', dresUrlInput.value.trim());
  if (dresUserInput) localStorage.setItem('dres_user', dresUserInput.value.trim());
  if (dresSessionInput) localStorage.setItem('dres_session', dresSessionInput.value.trim());
  if (dresEvalIdInput) localStorage.setItem('dres_eval_id', dresEvalIdInput.value.trim());
  updateDresConnection();
}

function updateDresConnection() {
  const connected = Boolean(dresSessionInput?.value.trim() && dresEvalIdInput?.value.trim());
  const pill = $('#dres-connection-pill');
  if (pill) {
    pill.textContent = connected ? 'Connected' : 'Offline';
    pill.className = `connection-pill ${connected ? 'connected' : 'offline'}`;
  }
}

function loadDresStorage() {
  const values = { '#dres-url': 'dres_url', '#dres-user': 'dres_user', '#dres-session': 'dres_session', '#dres-eval-id': 'dres_eval_id' };
  Object.entries(values).forEach(([selector, key]) => { const el = $(selector); const value = localStorage.getItem(key); if (el && value) el.value = value; });
  updateDresConnection();
  loadSubmittedMarkers();
}

function populateEvalDropdown(evaluations) {
  if (!dresEvalSelect || !Array.isArray(evaluations)) return;
  dresEvalSelect.replaceChildren(new Option('Select evaluation', ''));
  evaluations.forEach((evaluation) => dresEvalSelect.add(new Option(`${evaluation.name || evaluation.id}${evaluation.status ? ` · ${evaluation.status}` : ''}`, evaluation.id)));
  const active = evaluations.find((evaluation) => evaluation.status === 'ACTIVE') || evaluations[0];
  if (active?.id) {
    dresEvalSelect.value = active.id;
    dresEvalIdInput.value = active.id;
    saveDresStorage();
    loadSubmittedMarkers();
  }
}

[dresUrlInput, dresUserInput, dresSessionInput, dresEvalIdInput].forEach((element) => element?.addEventListener('change', () => { saveDresStorage(); loadSubmittedMarkers(); }));
dresEvalSelect?.addEventListener('change', () => {
  if (dresEvalSelect.value) dresEvalIdInput.value = dresEvalSelect.value;
  saveDresStorage();
  loadSubmittedMarkers();
});

$('#dres-setup-toggle')?.addEventListener('click', () => {
  const setup = $('#dres-setup');
  if (!setup) return;
  setup.hidden = !setup.hidden;
  $('#dres-setup-toggle').setAttribute('aria-expanded', String(!setup.hidden));
});

function updateModePill() {
  const isQa = Boolean(qaAnswerInput?.value.trim());
  const pill = $('#dres-mode-pill');
  if (pill) { pill.textContent = isQa ? 'Q&A' : 'KIS'; pill.classList.toggle('qa', isQa); }
  updateLiveTelemetry();
}
qaAnswerInput?.addEventListener('input', updateModePill);
$('#qa-clear-btn')?.addEventListener('click', () => { qaAnswerInput.value = ''; updateModePill(); qaAnswerInput.focus(); });

$('#dres-login-btn')?.addEventListener('click', async () => {
  const username = dresUserInput?.value.trim();
  const password = dresPassInput?.value || '';
  const base_url = dresUrlInput?.value.trim() || 'https://eventretrieval.one';
  if (!username || !password) { showToast('DRES login requires username and password', 'error'); return; }
  const button = $('#dres-login-btn');
  button.disabled = true;
  dresStatus.textContent = 'Logging in to DRES…';
  try {
    const response = await fetch('/api/dres/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ base_url, username, password }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    dresSessionInput.value = data.sessionId || data.session_id || '';
    populateEvalDropdown(data.evaluations || []);
    saveDresStorage();
    loadSubmittedMarkers();
    showToast(`DRES connected · session ${dresSessionInput.value.slice(0, 10)}…`, 'success');
  } catch (error) { showToast(`DRES login failed: ${error.message}`, 'error'); }
  finally { button.disabled = false; }
});

$('#dres-refresh-eval')?.addEventListener('click', async () => {
  const base_url = dresUrlInput?.value.trim() || 'https://eventretrieval.one';
  const session_id = dresSessionInput?.value.trim();
  if (!session_id) { showToast('Enter a DRES session before refreshing evaluations', 'error'); return; }
  try {
    const response = await fetch(`/api/dres/evaluations?base_url=${encodeURIComponent(base_url)}&session_id=${encodeURIComponent(session_id)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    populateEvalDropdown(data.evaluations || data || []);
    showToast(`Loaded ${(data.evaluations || data || []).length} evaluations`, 'success');
  } catch (error) { showToast(`Evaluation refresh failed: ${error.message}`, 'error'); }
});

function markerKey(mode, videoStem, ms, qaAnswer = '') {
  return `${mode}:${videoStem.toUpperCase()}:${ms}:${mode === 'qa' ? qaAnswer : ''}`;
}

function itemIsSubmitted(item) {
  const stem = cleanVideoStem(item.video_id).toUpperCase();
  const ms = itemMs(item);
  return [...submittedKeys].some((key) => { const parts = key.split(':'); return parts[1] === stem && Number(parts[2]) === ms; });
}

function setSubmissionBusy(stem, ms, busy) {
  const id = `${stem.toUpperCase()}|${ms}`;
  $$('[data-submission]').filter((button) => button.dataset.submission === id).forEach((button) => { button.disabled = busy; button.classList.toggle('loading', busy); });
}

async function submitToDres(videoId, timeMs, mode = 'kis', cardEl = null) {
  const stem = cleanVideoStem(videoId);
  const ms = Math.max(0, Math.round(Number(timeMs) || 0));
  const qa_answer = qaAnswerInput?.value.trim() || '';
  if (mode === 'qa' && !qa_answer) { showToast('Enter a Q&A answer before submitting', 'error'); qaAnswerInput?.focus(); return; }
  const key = markerKey(mode, stem, ms, qa_answer);
  if (submittedKeys.has(key) && !window.confirm(`Already submitted ${stem} @ ${ms} ms. Submit it again?`)) return;
  if (inFlightSubmissions.has(key)) return;
  const session_id = dresSessionInput?.value.trim();
  const evaluation_id = dresEvalIdInput?.value.trim();
  const payloadLabel = mode === 'qa' ? `Q&A ${qa_answer}-${stem}-${ms}` : `${stem} @ ${ms} ms`;
  if (!session_id || !evaluation_id) {
    try { await copyPayload(stem, ms, mode); showToast(`Copied (DRES offline): ${payloadLabel}`, 'warning'); } catch (_) {}
    return;
  }
  inFlightSubmissions.add(key);
  setSubmissionBusy(stem, ms, true);
  try {
    const response = await fetch('/api/dres/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ base_url: (dresUrlInput?.value.trim() || 'https://eventretrieval.one').replace(/\/$/, ''), session_id, evaluation_id, mode, video_id: stem, start_ms: ms, end_ms: ms, qa_answer }) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data) || `HTTP ${response.status}`);
    submittedKeys.add(key);
    persistSubmittedMarkers();
    cardEl?.classList.add('submitted');
    showToast(`Submitted ${payloadLabel}`, 'success');
    renderVisualResults();
  } catch (error) { showToast(`Submit failed (${payloadLabel}): ${error.message}`, 'error'); }
  finally { inFlightSubmissions.delete(key); setSubmissionBusy(stem, ms, false); }
}

async function copyPayload(stem, ms, mode = null) {
  const qa = qaAnswerInput?.value.trim() || '';
  const useQa = mode === 'qa' || (!mode && qa);
  const text = useQa ? `${qa}-${stem}-${ms}` : `${stem}\t${ms}`;
  let copied = false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (_) {}
  if (!copied) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0;';
    document.body.append(textarea);
    try {
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
      copied = document.execCommand('copy');
    } catch (_) {
      copied = false;
    } finally {
      textarea.remove();
    }
  }
  if (!copied) {
    showToast('Copy failed: clipboard access is unavailable', 'warning');
    throw new Error('Clipboard copy failed');
  }
  copiedKeys.add(`${stem.toUpperCase()}|${ms}`);
  showToast(`Copied ${useQa ? `Q&A ${text}` : `${stem} @ ${ms} ms`}`, 'success');
  return text;
}

loadDresStorage();
updateModePill();

// Scene builder.
function addConditionToScene(list, value = { text: '', region: 'all', weight: 1, expanded: '' }) {
  if (!list) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'condition-wrapper';
  wrapper.innerHTML = `<div class="condition"><input class="query-text" aria-label="Mô tả" placeholder="Mô tả tiếng Việt hoặc English…" value="${escapeHtml(value.text)}"><select class="region-select">${regionDefs.map(([id, label]) => `<option value="${id}"${id === value.region ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select><input class="weight-input" aria-label="Trọng số" type="number" min="0.1" max="10" step="0.1" value="${Number(value.weight) || 1}"><button class="expand-btn" type="button" title="Dịch sang tiếng Anh">Translate</button><button class="delete-btn" type="button" title="Xóa điều kiện">×</button></div><div class="expanded-row"><span class="expanded-label">QUERY EN</span><input class="expanded-input" placeholder="Auto-translated or manually entered English query…" value="${escapeHtml(value.expanded)}"></div>`;
  const query = $('.query-text', wrapper);
  const expanded = $('.expanded-input', wrapper);
  const expandButton = $('.expand-btn', wrapper);
  $('.delete-btn', wrapper).addEventListener('click', () => {
    if ($$('.condition-wrapper', list).length > 1) wrapper.remove();
    else { query.value = ''; expanded.value = ''; }
  });
  expandButton.addEventListener('click', async () => {
    if (!query.value.trim()) return;
    expandButton.disabled = true;
    try {
      const response = await fetch('/api/expand', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: query.value.trim(), region: 'all', weight: 1, stage: 1 }) });
      const data = await response.json();
      if (response.ok) { expanded.value = data.variants?.at(-1) || query.value.trim(); expanded.dataset.auto = 'true'; }
    } finally { expandButton.disabled = false; }
  });
  query.addEventListener('input', () => { if (expanded.dataset.auto === 'true') expanded.value = ''; });
  expanded.addEventListener('input', () => { expanded.dataset.auto = 'manual'; });
  [query, expanded].forEach((input) => input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); runSearch(); } }));
  list.append(wrapper);
  query.focus();
}

function refreshSceneLabels() {
  $$('.scene-box', scenesContainer).forEach((box, index) => {
    const stage = index + 1;
    box.dataset.stage = String(stage);
    box.classList.toggle('scene-subsequent', stage > 1);
    $('.scene-title', box).textContent = stage === 1 ? 'Scene 1 · conditions in the same frame' : `Scene ${stage} · follows scene ${stage - 1}`;
    const remove = $('.remove-scene-btn', box);
    if (remove) remove.hidden = stage === 1;
  });
}

function addSceneBox() {
  const box = document.createElement('div');
  box.className = 'scene-box';
  box.innerHTML = '<div class="scene-header"><span class="scene-title"></span><div class="scene-actions"><button type="button" class="add-cond-btn">+ Condition</button><button type="button" class="danger-btn remove-scene-btn">Remove scene</button></div></div><div class="scene-conditions"></div>';
  const list = $('.scene-conditions', box);
  $('.add-cond-btn', box).addEventListener('click', () => addConditionToScene(list));
  $('.remove-scene-btn', box).addEventListener('click', () => { box.remove(); refreshSceneLabels(); });
  scenesContainer?.append(box);
  refreshSceneLabels();
  addConditionToScene(list);
}

function getConditions() {
  const conditions = [];
  $$('.scene-box', scenesContainer).forEach((box, index) => {
    $$('.condition-wrapper', box).forEach((wrapper) => {
      const text = $('.query-text', wrapper)?.value.trim();
      const expanded = $('.expanded-input', wrapper)?.value.trim();
      if (!text && !expanded) return;
      conditions.push({ text: text || expanded, region: $('.region-select', wrapper)?.value || 'all', weight: Number($('.weight-input', wrapper)?.value) || 1, stage: Math.min(index + 1, 10), variants: expanded ? [expanded] : null });
    });
  });
  return conditions;
}

addSceneBox();
$('#add-scene')?.addEventListener('click', addSceneBox);

// Scope presets.
function updateScopeSummary() {
  const enabled = $('#use-video-filter')?.checked;
  const value = $('#video-filter-input')?.value.trim();
  const summary = $('#scope-summary');
  if (summary) summary.textContent = enabled && value ? value : 'ALL';
}
$$('[data-scope]').forEach((button) => button.addEventListener('click', () => {
  const value = button.dataset.scope || '';
  $('#video-filter-input').value = value;
  $('#use-video-filter').checked = Boolean(value);
  updateScopeSummary();
  if (getConditions().length) runSearch();
}));
$('#video-filter-input')?.addEventListener('input', updateScopeSummary);
$('#use-video-filter')?.addEventListener('change', updateScopeSummary);
updateScopeSummary();

// Combined OCR + ASR filtering.
function normalizeModalityResult(data, source) {
  const videoIds = [...new Set((data.video_ids || []).map((id) => String(id).toUpperCase()))];
  const matches = (data.matches || []).map((match) => ({ ...match, source }));
  const byVideo = {};
  matches.forEach((match) => { const stem = String(match.video_id || '').toUpperCase(); if (stem && !byVideo[stem]) byVideo[stem] = match; });
  return { ...data, source, videoIds, videoSet: new Set(videoIds), matches, byVideo };
}

function combineSets(baseVideos, ocrResult, asrResult, mode) {
  const base = new Set(baseVideos.map((id) => id.toUpperCase()));
  let combined;
  if (ocrResult && asrResult) combined = mode === 'and'
    ? new Set([...ocrResult.videoSet].filter((id) => asrResult.videoSet.has(id)))
    : new Set([...ocrResult.videoSet, ...asrResult.videoSet]);
  else combined = new Set((ocrResult || asrResult)?.videoSet || []);
  return base.size ? new Set([...combined].filter((id) => base.has(id))) : combined;
}

function commitCombined(signature, ocrResult, asrResult, direct = false) {
  const videos = getVisualCandidateVideos();
  const videoSet = combineSets(videos, ocrResult, asrResult, combineMode);
  activeCombinedFilter = { signature, mode: combineMode, direct, ocr: ocrResult, asr: asrResult, videoSet };
  evidenceShowAll = false;
  renderEvidence();
  renderVisualResults();
  updateFilterSummary();
}

function recombineFromCache() {
  const signature = visualSignature();
  const ocrQuery = csvQueryInput?.value.trim() || '';
  const asrQuery = asrQueryInput?.value.trim() || '';
  const ocrResult = ocrQuery ? modalityCaches.ocr.get(cacheKey(signature, ocrQuery)) : null;
  const asrResult = asrQuery ? modalityCaches.asr.get(cacheKey(signature, asrQuery)) : null;
  if ((!ocrQuery || ocrResult) && (!asrQuery || asrResult) && (ocrQuery || asrQuery)) commitCombined(signature, ocrResult, asrResult, !signature);
  else updateFilterSummary();
}

async function fetchModality(source, query, videos, signal) {
  const response = await fetch(source === 'ocr' ? '/api/csv/search' : '/api/asr/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, limit: 160, video_filter: videos.length ? videos : null }), signal });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  return normalizeModalityResult(data, source);
}

async function runCombinedFilter() {
  const ocrQuery = csvQueryInput?.value.trim() || '';
  const asrQuery = asrQueryInput?.value.trim() || '';
  if (!ocrQuery && !asrQuery) { resetCombinedFilter(); return; }
  const videos = getVisualCandidateVideos();
  const signature = videos.slice().sort().join(',');
  const generation = ++filterGeneration;
  ocrAbortController?.abort();
  asrAbortController?.abort();
  ocrAbortController = new AbortController();
  asrAbortController = new AbortController();
  const applyButton = $('#filter-apply-btn');
  if (applyButton) { applyButton.disabled = true; applyButton.textContent = 'Filtering…'; }
  candidateBadge.textContent = videos.length ? `Searching ${videos.length} visual candidate videos…` : 'Searching the full OCR / ASR corpus…';

  const tasks = [];
  const labels = [];
  for (const [source, query, controller] of [['ocr', ocrQuery, ocrAbortController], ['asr', asrQuery, asrAbortController]]) {
    if (!query) continue;
    labels.push(source);
    const key = cacheKey(signature, query);
    const cached = modalityCaches[source].get(key);
    tasks.push(cached ? Promise.resolve(cached) : fetchModality(source, query, videos, controller.signal).then((result) => { cacheSet(source, key, result); return result; }));
  }
  const settled = await Promise.allSettled(tasks);
  if (generation !== filterGeneration || visualSignature() !== signature) return;
  const failed = settled.map((result, index) => result.status === 'rejected' && result.reason?.name !== 'AbortError' ? `${labels[index].toUpperCase()}: ${result.reason?.message || 'request failed'}` : null).filter(Boolean);
  if (failed.length) {
    candidateBadge.textContent = `Filter not changed · ${failed.join(' · ')}`;
    candidateBadge.dataset.state = 'error';
    if (csvResultsBar) { csvResultsBar.hidden = false; csvResultsBar.innerHTML = `<div class="evidence-summary"><span>⚠ ${escapeHtml(failed.join(' · '))}</span><span>Last valid filter retained</span></div>`; }
  } else {
    let ocrResult = null;
    let asrResult = null;
    settled.forEach((result, index) => { if (labels[index] === 'ocr') ocrResult = result.value; else asrResult = result.value; });
    commitCombined(signature, ocrResult, asrResult, !videos.length);
  }
  if (applyButton) { applyButton.disabled = false; applyButton.textContent = 'Apply filters'; }
}

function resetCombinedFilter() {
  ++filterGeneration;
  ocrAbortController?.abort();
  asrAbortController?.abort();
  activeCombinedFilter = null;
  const applyButton = $('#filter-apply-btn');
  if (applyButton) { applyButton.disabled = false; applyButton.textContent = 'Apply filters'; }
  if (csvQueryInput) csvQueryInput.value = '';
  if (asrQueryInput) asrQueryInput.value = '';
  if (csvResultsBar) { csvResultsBar.hidden = true; csvResultsBar.replaceChildren(); }
  if (evidenceSection) { evidenceSection.hidden = true; evidenceSection.replaceChildren(); }
  renderVisualResults();
  updateFilterSummary();
}

function updateFilterSummary() {
  if (!candidateBadge) return;
  const visualVideos = getVisualCandidateVideos();
  if (!activeCombinedFilter) {
    candidateBadge.textContent = visualVideos.length ? `${visualVideos.length} visual videos · filters disabled` : 'No visual search yet · direct corpus search available';
    candidateBadge.dataset.state = '';
    return;
  }
  const ocrCount = activeCombinedFilter.ocr?.videoIds.length;
  const asrCount = activeCombinedFilter.asr?.videoIds.length;
  if (activeCombinedFilter.direct) {
    candidateBadge.textContent = `Direct corpus · ${activeCombinedFilter.videoSet.size} videos (OCR: ${ocrCount ?? 'off'} · ASR: ${asrCount ?? 'off'} · ${activeCombinedFilter.mode.toUpperCase()})`;
  } else {
    candidateBadge.textContent = `Showing ${activeCombinedFilter.videoSet.size} / ${visualVideos.length} visual videos (OCR: ${ocrCount ?? 'off'} · ASR: ${asrCount ?? 'off'} · ${activeCombinedFilter.mode.toUpperCase()})`;
  }
  candidateBadge.dataset.state = 'ok';
}

function evidenceChip(match, index, source) {
  const start = Number(match.timestamp || 0);
  const end = Number(match.end_timestamp ?? start);
  const interval = source === 'asr' ? `[${formatTime(start, true)}–${formatTime(end, true)}]` : match.frame_id != null ? `#${match.frame_id}` : formatTime(start, true);
  return `<button type="button" class="snippet-chip" data-source="${source}" data-index="${index}"><strong><span class="modality-badge ${source}">${source.toUpperCase()}</span> ${escapeHtml(match.video_id)}</strong><span class="interval">${escapeHtml(interval)} · ${escapeHtml(match.match_type || 'match')}</span><span>${escapeHtml(match.text || '')}</span></button>`;
}

function renderEvidence() {
  if (!activeCombinedFilter || (!activeCombinedFilter.ocr && !activeCombinedFilter.asr)) return;
  const intersectionOnly = Boolean(activeCombinedFilter.ocr && activeCombinedFilter.asr && activeCombinedFilter.mode === 'and');
  const narrowMatches = (matches) => intersectionOnly
    ? matches.filter((match) => activeCombinedFilter.videoSet.has(String(match.video_id || '').toUpperCase()))
    : matches;
  const ocrMatches = narrowMatches(activeCombinedFilter.ocr?.matches || []);
  const asrMatches = narrowMatches(activeCombinedFilter.asr?.matches || []);
  const evidenceCount = ocrMatches.length + asrMatches.length;
  const evidenceLabel = intersectionOnly ? `Matched evidence (${evidenceCount} in intersection)` : (activeCombinedFilter.direct ? 'Direct corpus evidence' : 'Evidence within visual candidates');
  csvResultsBar.hidden = false;
  csvResultsBar.innerHTML = `<div class="evidence-summary"><span>${evidenceLabel}</span><b>${activeCombinedFilter.videoSet.size} combined videos · ${activeCombinedFilter.mode.toUpperCase()}</b></div><div class="evidence-groups">${activeCombinedFilter.ocr ? `<div class="evidence-group"><h4><span class="modality-badge ocr">OCR</span> ${ocrMatches.length}${intersectionOnly ? ' in intersection' : ''} visual-text matches</h4><div class="snippet-list">${ocrMatches.slice(0, 24).map((match, index) => evidenceChip(match, index, 'ocr')).join('') || '<span class="field-hint">No OCR matches</span>'}</div></div>` : ''}${activeCombinedFilter.asr ? `<div class="evidence-group"><h4><span class="modality-badge asr">ASR</span> ${asrMatches.length}${intersectionOnly ? ' in intersection' : ''} spoken-transcript matches</h4><div class="snippet-list">${asrMatches.slice(0, 24).map((match, index) => evidenceChip(match, index, 'asr')).join('') || '<span class="field-hint">No ASR matches</span>'}</div></div>` : ''}</div>`;
  $$('.snippet-chip', csvResultsBar).forEach((chip) => chip.addEventListener('click', () => {
    const matches = chip.dataset.source === 'ocr' ? ocrMatches : asrMatches;
    const match = matches[Number(chip.dataset.index)];
    if (match) inspectResult(matchToItem(match));
  }));
  renderEvidenceCards();
}

function matchToItem(match) {
  const source = match.source || (match.end_timestamp != null ? 'asr' : 'ocr');
  const segment_start = Number(match.timestamp || 0);
  const segment_end = match.end_timestamp != null ? Number(match.end_timestamp) : null;
  const keyframeTimestamp = match.keyframe_timestamp != null ? Number(match.keyframe_timestamp) : NaN;
  const hasKeyframeTimestamp = Number.isFinite(keyframeTimestamp);
  const timestamp = hasKeyframeTimestamp ? keyframeTimestamp : segment_start;
  const timestamp_ms = hasKeyframeTimestamp ? Math.round(timestamp * 1000) : match.timestamp_ms != null ? Number(match.timestamp_ms) : Math.round(segment_start * 1000);
  const fps = Number(match.fps) || 25;
  return { ...match, source, video_stem: match.video_id, keyframe_id: `${match.video_id}:${match.frame_id ?? Math.round(timestamp * fps)}`, frame_id: match.frame_id ?? Math.round(timestamp * fps), timestamp, timestamp_ms, segment_start, segment_end, fps, score: Number(match.match_score ?? 1) };
}

function renderEvidenceCards() {
  const intersectionOnly = Boolean(activeCombinedFilter?.ocr && activeCombinedFilter?.asr && activeCombinedFilter.mode === 'and');
  const matches = [...(activeCombinedFilter?.ocr?.matches || []), ...(activeCombinedFilter?.asr?.matches || [])]
    .filter((match) => !intersectionOnly || activeCombinedFilter.videoSet.has(String(match.video_id || '').toUpperCase()));
  if (!matches.length) { evidenceSection.hidden = true; return; }
  const shown = evidenceShowAll ? matches : matches.slice(0, 8);
  evidenceSection.hidden = false;
  evidenceSection.innerHTML = `<div class="evidence-section-header"><span>Matched evidence (${matches.length}${intersectionOnly ? ' in intersection' : ''})</span><div>${matches.length > 8 ? `<button id="evidence-toggle" type="button" class="secondary-btn">${evidenceShowAll ? 'Show first 8' : `Show all ${matches.length}`}</button>` : ''} <button id="evidence-hide" type="button" class="secondary-btn">Hide</button></div></div><div class="evidence-grid"></div>`;
  const grid = $('.evidence-grid', evidenceSection);
  shown.forEach((match) => grid.append(createResultCard(matchToItem(match), { evidence: true })));
  $('#evidence-toggle', evidenceSection)?.addEventListener('click', () => { evidenceShowAll = !evidenceShowAll; renderEvidenceCards(); });
  $('#evidence-hide', evidenceSection)?.addEventListener('click', () => { evidenceSection.hidden = true; });
}

$('#filter-apply-btn')?.addEventListener('click', runCombinedFilter);
$('#csv-search-btn')?.addEventListener('click', runCombinedFilter);
$('#ocr-reset-filter-btn')?.addEventListener('click', resetCombinedFilter);
[csvQueryInput, asrQueryInput].forEach((input) => input?.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); event.stopPropagation(); runCombinedFilter(); } }));
$$('[data-clear-input]').forEach((button) => button.addEventListener('click', () => { const input = document.getElementById(button.dataset.clearInput); if (input) { input.value = ''; input.focus(); } }));
$$('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  combineMode = button.dataset.mode;
  localStorage.setItem('spatialKis.filterCombineMode', combineMode);
  $$('[data-mode]').forEach((item) => { const active = item.dataset.mode === combineMode; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); });
  const signature = visualSignature();
  const ocrQuery = csvQueryInput?.value.trim() || '';
  const asrQuery = asrQueryInput?.value.trim() || '';
  const missingCachedResult = (ocrQuery && !modalityCaches.ocr.has(cacheKey(signature, ocrQuery)))
    || (asrQuery && !modalityCaches.asr.has(cacheKey(signature, asrQuery)));
  if ((ocrQuery || asrQuery) && missingCachedResult) runCombinedFilter();
  else recombineFromCache();
}));
$$('[data-mode]').forEach((button) => { const active = button.dataset.mode === combineMode; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); });

// Visual search with abort/generation protection.
async function expandPendingConditions(generation) {
  if (!$('#expand')?.checked) return;
  const pending = $$('.condition-wrapper').filter((wrapper) => $('.query-text', wrapper)?.value.trim() && !$('.expanded-input', wrapper)?.value.trim());
  if (pending.length) statusEl.textContent = 'Translating queries…';
  await Promise.all(pending.map(async (wrapper) => {
    const query = $('.query-text', wrapper).value.trim();
    try {
      const response = await fetch('/api/expand', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: query }) });
      const data = await response.json();
      if (response.ok && generation === visualGeneration) { const target = $('.expanded-input', wrapper); target.value = data.variants?.at(-1) || query; target.dataset.auto = 'true'; }
    } catch (_) {}
  }));
}

async function runSearch() {
  const generation = ++visualGeneration;
  visualSearchController?.abort();
  visualSearchController = new AbortController();
  ++filterGeneration;
  ocrAbortController?.abort();
  asrAbortController?.abort();
  const filterButton = $('#filter-apply-btn');
  if (filterButton) { filterButton.disabled = false; filterButton.textContent = 'Apply filters'; }
  await expandPendingConditions(generation);
  if (generation !== visualGeneration) return;
  const conditions = getConditions();
  if (!conditions.length) { statusEl.textContent = 'Enter at least one visual condition.'; return; }
  const scope = $('#use-video-filter')?.checked ? $('#video-filter-input')?.value.trim() : '';
  const videoFilter = scope.split(/[\s,;]+/).map((value) => value.trim()).filter(Boolean);
  const request = { conditions, result_limit: 120, diversify: Boolean($('#diversify')?.checked), expand: Boolean($('#expand')?.checked), max_temporal_gap: Number($('#temporal-gap')?.value) || 45, video_filter: videoFilter.length ? videoFilter : null };
  statusEl.textContent = 'Searching visual keyframes…';
  const searchButton = $('#search');
  searchButton.disabled = true;
  const started = performance.now();
  try {
    const response = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal: visualSearchController.signal });
    const data = await response.json().catch(() => ({}));
    if (generation !== visualGeneration) return;
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    lastVisualResults = data.results || [];
    lastVisualLatencyMs = Math.round(performance.now() - started);
    activeCombinedFilter = null;
    if (csvResultsBar) csvResultsBar.hidden = true;
    if (evidenceSection) evidenceSection.hidden = true;
    renderVisualResults();
    updateFilterSummary();
  } catch (error) {
    if (error.name !== 'AbortError' && generation === visualGeneration) statusEl.textContent = `Visual search failed: ${error.message}`;
  } finally {
    if (generation === visualGeneration) searchButton.disabled = false;
  }
}
$('#search')?.addEventListener('click', runSearch);

// Result cards, density, and submitted visibility.
function temporalPreview(item) {
  const nextList = item.next_matches || item.next_stages || item.next_results || (item.next_match ? [item.next_match] : item.next ? [item.next] : []);
  if (!nextList.length) return { html: '', nextList };
  const html = nextList.map((next, index) => {
    const stage = next.stage || index + 2;
    const ms = itemMs(next);
    const frame = next.frame_id ?? Math.round(Number(next.timestamp || ms / 1000) * 25);
    return `<div class="next-scene-box" data-next-index="${index}" role="button" tabindex="0" aria-label="Inspect scene ${stage} at ${ms} milliseconds">${next.image_path ? `<img src="${imageUrl(next.image_path)}" alt="Scene ${stage} keyframe at ${ms} ms" loading="lazy">` : '<div class="card-media audio-placeholder" role="img" aria-label="No keyframe available"></div>'}<div class="card-submeta scene-caption"><span class="card-tag next-tag">⏩ Scene ${stage}</span><span>${ms} ms</span><span>· #${escapeHtml(String(frame))}</span></div></div>`;
  }).join('');
  return { html, nextList };
}

function createResultCard(item, options = {}) {
  const card = document.createElement('article');
  card.className = 'card';
  const stem = cleanVideoStem(item.video_stem || item.video_id);
  const ms = itemMs(item);
  const timestamp = Number(item.timestamp || ms / 1000);
  const requestedSource = item.source || (options.evidence ? 'ocr' : 'visual');
  const source = requestedSource === 'asr' ? 'asr' : requestedSource === 'ocr' ? 'ocr' : 'visual';
  const submitted = itemIsSubmitted(item);
  if (submitted) card.classList.add('submitted');
  if (copiedKeys.has(`${stem.toUpperCase()}|${ms}`)) card.classList.add('copied');
  const preview = temporalPreview(item);
  const image = imageUrl(item.image_path);
  const evidenceMatch = !options.evidence && activeCombinedFilter
    ? activeCombinedFilter.ocr?.byVideo[stem.toUpperCase()] || activeCombinedFilter.asr?.byVideo[stem.toUpperCase()]
    : null;
  const textMatch = options.evidence ? item : evidenceMatch;
  const textClass = (textMatch?.source || source) === 'asr' ? 'asr-text-snippet' : '';
  const interval = source === 'asr' ? `${formatTime(item.segment_start ?? timestamp, true)}–${formatTime(item.segment_end ?? item.end_timestamp ?? item.segment_start ?? timestamp, true)}` : `#${item.frame_id ?? Math.round(timestamp * 25)}`;
  const primaryFrame = item.frame_id ?? Math.round(timestamp * 25);
  const stage1Caption = preview.nextList.length ? `<div class="card-submeta scene-caption"><span class="card-tag">Scene 1</span><span>${ms} ms</span><span>· #${escapeHtml(String(primaryFrame))}</span></div>` : '';
  card.innerHTML = `${image ? `<img class="stage1-img" src="${image}" alt="${escapeHtml(stem)} scene 1 at ${ms} ms" loading="lazy">` : `<div class="card-media audio-placeholder" role="img" aria-label="No keyframe available"></div>`}${stage1Caption}${preview.html}<div class="card-meta"><span class="video-pill">${escapeHtml(stem)}</span><span class="ms-pill">${ms} ms</span></div><div class="card-submeta"><span>${formatTime(timestamp)}</span><span>· ${escapeHtml(interval)}</span>${source === 'visual' ? `<span>· Score <b>${Number(item.score || 0).toFixed(3)}</b></span><span class="card-tag">${escapeHtml(REGION_NAMES[item.region_id] || item.region_id || 'all')}</span>` : `<span class="modality-badge ${source}">${source.toUpperCase()}</span><span>${escapeHtml(item.match_type || 'match')}</span>`}</div>${textMatch?.text ? `<div class="ocr-text-snippet ${textClass}">${escapeHtml(textMatch.text)}</div>` : ''}<div class="card-actions"><button type="button" class="view-btn">Inspect</button><button type="button" class="secondary-btn copy-btn">Copy</button></div><div class="card-submit-row"><button type="button" class="submit-kis kis-btn" data-submission="${escapeHtml(`${stem.toUpperCase()}|${ms}`)}">KIS</button><button type="button" class="submit-qa qa-btn" data-submission="${escapeHtml(`${stem.toUpperCase()}|${ms}`)}">Q&amp;A</button></div>`;
  $('.stage1-img', card)?.addEventListener('click', () => inspectResult(item));
  $('.view-btn', card).addEventListener('click', () => inspectResult(item));
  $$('.next-scene-box', card).forEach((element) => {
    const inspectScene = () => { const next = preview.nextList[Number(element.dataset.nextIndex)]; if (next) inspectResult({ ...item, ...next }); };
    element.addEventListener('click', inspectScene);
    element.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspectScene(); } });
  });
  $('.copy-btn', card).addEventListener('click', async () => { try { await copyPayload(stem, ms); card.classList.add('copied'); } catch (_) {} });
  $('.kis-btn', card).addEventListener('click', () => submitToDres(stem, ms, 'kis', card));
  $('.qa-btn', card).addEventListener('click', () => submitToDres(stem, ms, 'qa', card));
  return card;
}

function renderVisualResults() {
  if (!results) return;
  results.replaceChildren();
  let visible = lastVisualResults;
  if (activeCombinedFilter && !activeCombinedFilter.direct) visible = visible.filter((item) => activeCombinedFilter.videoSet.has((item.video_stem || cleanVideoStem(item.video_id)).toUpperCase()));
  const beforeHide = visible.length;
  if (hideSubmitted) visible = visible.filter((item) => !itemIsSubmitted(item));
  visible.forEach((item) => results.append(createResultCard(item)));
  if (resultCount) resultCount.textContent = `${visible.length}${hideSubmitted && beforeHide !== visible.length ? ` / ${beforeHide}` : ''} results`;
  if (lastVisualResults.length) {
    const filteredLabel = activeCombinedFilter && !activeCombinedFilter.direct ? ` · ${activeCombinedFilter.videoSet.size}/${getVisualCandidateVideos().length} videos after ${activeCombinedFilter.mode.toUpperCase()}` : '';
    statusEl.textContent = `${visible.length} visual results · ${getVisualCandidateVideos().length} videos · ${lastVisualLatencyMs} ms${filteredLabel}`;
  }
}

function applyDensity() {
  document.body.classList.toggle('grid-compact', gridDensity === 'compact');
  $$('[data-density]').forEach((button) => button.classList.toggle('active', button.dataset.density === gridDensity));
}
$$('[data-density]').forEach((button) => button.addEventListener('click', () => { gridDensity = button.dataset.density; localStorage.setItem('spatialKis.gridDensity', gridDensity); applyDensity(); }));
const hideSubmittedInput = $('#hide-submitted');
if (hideSubmittedInput) { hideSubmittedInput.checked = hideSubmitted; hideSubmittedInput.addEventListener('change', () => { hideSubmitted = hideSubmittedInput.checked; localStorage.setItem('spatialKis.hideSubmitted', String(hideSubmitted)); renderVisualResults(); }); }
applyDensity();
updateFilterSummary();

// Inspect modal, segment playback, neighbors, and keyboard controls.
function closeInspect() {
  neighborsAbortController?.abort();
  neighborsAbortController = null;
  segmentPlayback = false;
  if (player) { player.pause(); player.removeAttribute('src'); player.load(); }
  if (inspect?.open) inspect.close();
}

function currentInspectSeconds() {
  if (player && player.readyState >= 1 && Number.isFinite(player.currentTime)) return Math.max(0, player.currentTime);
  return Math.max(0, Number(currentInspectItem?.timestamp || 0));
}

function stepVideo(deltaSeconds) {
  if (!player) return;
  segmentPlayback = false;
  player.pause();
  player.currentTime = Math.max(0, Math.min(Number.isFinite(player.duration) ? player.duration : Infinity, currentInspectSeconds() + deltaSeconds));
  updateLiveTelemetry();
}

function stepFrames(deltaFrames) {
  const fps = Number(currentInspectItem?.fps) || 25;
  stepVideo(deltaFrames / fps);
}

function updateLiveTelemetry() {
  if (!player || !currentInspectItem || !details) return;
  const fps = Number(currentInspectItem.fps || 25);
  const seconds = currentInspectSeconds();
  const ms = Math.round(seconds * 1000);
  const stem = cleanVideoStem(currentInspectItem.video_id);
  const qa = qaAnswerInput?.value.trim() || '';
  details.textContent = `${stem} · ${formatTime(seconds)} · ${ms} ms · frame ~${Math.round(seconds * fps)}${qa ? ` · Q&A ${qa}-${stem}-${ms}` : ''}`;
  const submission = `${stem.toUpperCase()}|${ms}`;
  const kis = $('#modal-submit-kis');
  const qaButton = $('#modal-submit-qa');
  if (kis) { kis.textContent = `Submit KIS · ${ms} ms`; if (!kis.disabled) kis.dataset.submission = submission; }
  if (qaButton) { qaButton.textContent = `Submit Q&A · ${ms} ms`; if (!qaButton.disabled) qaButton.dataset.submission = submission; }
}

async function loadNeighbors(item, signal) {
  const strip = $('#neighbors-strip');
  if (!strip) return;
  strip.textContent = 'Loading neighbors…';
  const timestamp = Number(item.keyframe_timestamp ?? item.timestamp ?? (item.frame_id != null ? item.frame_id / 25 : 0));
  try {
    const response = await fetch(`/api/neighbors?video_id=${encodeURIComponent(item.video_id)}&timestamp=${encodeURIComponent(timestamp)}&window=35`, { signal });
    const data = await response.json();
    if (signal.aborted) return;
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    strip.replaceChildren();
    (data.neighbors || []).forEach((neighbor) => {
      const element = document.createElement('button');
      element.type = 'button';
      element.className = `neighbor-item${neighbor.keyframe_id === item.keyframe_id ? ' active' : ''}`;
      const ms = itemMs(neighbor);
      element.innerHTML = `${neighbor.image_path ? `<img src="${imageUrl(neighbor.image_path)}" alt="Neighbor ${ms} ms" loading="lazy">` : ''}<div><b>${ms} ms</b></div><div>${formatTime(neighbor.timestamp)} · #${neighbor.frame_id}</div>`;
      element.addEventListener('click', () => { currentInspectItem = { ...item, ...neighbor, source: item.source }; player.currentTime = Number(neighbor.timestamp || 0); updateLiveTelemetry(); $$('.neighbor-item', strip).forEach((node) => node.classList.remove('active')); element.classList.add('active'); });
      strip.append(element);
    });
    if (!strip.children.length) strip.textContent = 'No keyframe neighbors available.';
  } catch (error) { if (error.name !== 'AbortError' && !signal.aborted) strip.textContent = `Neighbors unavailable: ${error.message}`; }
}

async function inspectResult(item) {
  if (!player || !inspect || !item?.video_id) return;
  neighborsAbortController?.abort();
  neighborsAbortController = new AbortController();
  currentInspectItem = item;
  segmentPlayback = false;
  const source = item.source || 'visual';
  const keyframeTime = Math.max(0, Number(item.timestamp || 0));
  const segmentStart = Math.max(0, Number(item.segment_start ?? item.timestamp ?? 0));
  const segmentEnd = Number(item.segment_end ?? item.end_timestamp);
  const transcript = $('#modal-transcript');
  const segmentButton = $('#play-segment-btn');
  const modality = $('#modal-modality');
  if (modality) { modality.textContent = source.toUpperCase(); modality.className = `modality-badge ${source === 'visual' ? '' : source}`; }
  $('#modal-title').textContent = `${cleanVideoStem(item.video_id)} · ${formatTime(keyframeTime)}`;
  if (transcript) {
    transcript.hidden = !item.text;
    transcript.textContent = item.text ? `${source === 'asr' ? `[${formatTime(segmentStart, true)} – ${formatTime(Number.isFinite(segmentEnd) ? segmentEnd : segmentStart, true)}] ` : ''}${item.text}` : '';
  }
  if (segmentButton) segmentButton.hidden = source !== 'asr' || !Number.isFinite(segmentEnd) || segmentEnd <= segmentStart;
  player.pause();
  player.src = `/media/video/${encodeURIComponent(item.video_id)}`;
  player.onloadedmetadata = () => { player.currentTime = keyframeTime; updateLiveTelemetry(); player.play().catch(() => {}); };
  updateLiveTelemetry();
  if (!inspect.open) inspect.showModal();
  await loadNeighbors(item, neighborsAbortController.signal);
}

player?.addEventListener('timeupdate', () => {
  updateLiveTelemetry();
  const segmentEnd = Number(currentInspectItem?.segment_end ?? currentInspectItem?.end_timestamp);
  if (segmentPlayback && Number.isFinite(segmentEnd) && player.currentTime >= segmentEnd) { player.pause(); segmentPlayback = false; $('#play-segment-btn').textContent = 'Play segment'; }
});
['seeking', 'seeked', 'pause', 'loadedmetadata'].forEach((eventName) => player?.addEventListener(eventName, updateLiveTelemetry));
$('#play-segment-btn')?.addEventListener('click', () => {
  if (!currentInspectItem) return;
  if (segmentPlayback) { player.pause(); segmentPlayback = false; $('#play-segment-btn').textContent = 'Play segment'; }
  else { player.currentTime = Math.max(0, Number(currentInspectItem.segment_start ?? currentInspectItem.timestamp ?? 0)); segmentPlayback = true; $('#play-segment-btn').textContent = 'Stop segment'; player.play().catch(() => { segmentPlayback = false; }); }
});
$$('[data-step]').forEach((button) => button.addEventListener('click', () => stepVideo(Number(button.dataset.step))));
$$('[data-frame-step]').forEach((button) => button.addEventListener('click', () => stepFrames(Number(button.dataset.frameStep))));
$('#inspect-close-btn')?.addEventListener('click', closeInspect);
inspect?.addEventListener('close', () => { segmentPlayback = false; player?.pause(); });
inspect?.addEventListener('click', (event) => { if (event.target === inspect) closeInspect(); });
$('#modal-copy-btn')?.addEventListener('click', () => { if (currentInspectItem) copyPayload(cleanVideoStem(currentInspectItem.video_id), Math.round(currentInspectSeconds() * 1000)).catch(() => {}); });
$('#modal-submit-kis')?.addEventListener('click', () => { if (currentInspectItem) submitToDres(currentInspectItem.video_id, Math.round(currentInspectSeconds() * 1000), 'kis'); });
$('#modal-submit-qa')?.addEventListener('click', () => { if (currentInspectItem) submitToDres(currentInspectItem.video_id, Math.round(currentInspectSeconds() * 1000), 'qa'); });

document.addEventListener('keydown', (event) => {
  if (event.isComposing || event.keyCode === 229) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'b') { event.preventDefault(); toggleSidebar(); return; }
  if (!inspect?.open) {
    if (event.key === 'Escape' && isMobile() && document.body.classList.contains('mobile-sidebar-open')) setSidebar(false);
    return;
  }
  const target = event.target;
  if (target?.matches?.('input, textarea, select, [contenteditable], [contenteditable="true"]')) return;
  if (event.key === 'Escape') { event.preventDefault(); closeInspect(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    const mode = event.shiftKey ? 'qa' : 'kis';
    if (currentInspectItem) submitToDres(currentInspectItem.video_id, Math.round(currentInspectSeconds() * 1000), mode);
    return;
  }
  if (!event.ctrlKey && !event.metaKey && event.key.toLowerCase() === 'c') { event.preventDefault(); if (currentInspectItem) copyPayload(cleanVideoStem(currentInspectItem.video_id), Math.round(currentInspectSeconds() * 1000)).catch(() => {}); return; }
  if (event.key === ' ') { event.preventDefault(); player.paused ? player.play().catch(() => {}) : player.pause(); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); stepVideo(event.shiftKey ? -5 : -1); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); stepVideo(event.shiftKey ? 5 : 1); }
  else if (event.key === ',') { event.preventDefault(); stepFrames(-1); }
  else if (event.key === '.') { event.preventDefault(); stepFrames(1); }
});

window.closeInspect = closeInspect;
window.stepVideo = stepVideo;
