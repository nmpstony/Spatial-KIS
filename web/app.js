const regionDefs = [
  { id: "all", label: "all (Toàn khung hình)" },
  { id: "left", label: "left (Bên trái: cột 1, 4, 7)" },
  { id: "center", label: "center (Chính giữa: ô 5)" },
  { id: "right", label: "right (Bên phải: cột 3, 6, 9)" },
  { id: "top", label: "top (Phía trên: hàng 1, 2, 3)" },
  { id: "bottom", label: "bottom (Phía dưới: hàng 7, 8, 9)" },
  { id: "middle_row", label: "middle_row (Hàng giữa: ô 4, 5, 6)" },
  { id: "center_col", label: "center_col (Cột giữa: ô 2, 5, 8)" },
  { id: "top_left", label: "top_left (Góc trên trái: ô 1)" },
  { id: "top_center", label: "top_center (Giữa trên: ô 2)" },
  { id: "top_right", label: "top_right (Góc trên phải: ô 3)" },
  { id: "middle_left", label: "middle_left (Giữa trái: ô 4)" },
  { id: "middle_right", label: "middle_right (Giữa phải: ô 6)" },
  { id: "bottom_left", label: "bottom_left (Góc dưới trái: ô 7)" },
  { id: "bottom_center", label: "bottom_center (Giữa dưới: ô 8)" },
  { id: "bottom_right", label: "bottom_right (Góc dưới phải: ô 9)" }
];

const scenesContainer = document.querySelector('#scenes-container'),
      results = document.querySelector('#results'),
      status = document.querySelector('#status'),
      inspect = document.querySelector('#inspect'),
      player = document.querySelector('#player'),
      dresStatus = document.querySelector('#dres-status');

let currentInspectItem = null;
const submittedKeys = new Set();

function cleanVideoStem(videoId) {
  if (!videoId) return '';
  const base = videoId.split('/').pop().split('\\').pop();
  return base.replace(/\.mp4$/i, '');
}

async function checkServerHealth() {
  const badge = document.querySelector('#backend-badge');
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      badge.textContent = `● Server Ready (${data.backend || 'active'})`;
      badge.style.color = '#7ee787';
    } else {
      badge.textContent = '⚠ Lỗi kết nối Server';
      badge.style.color = '#ff7b72';
    }
  } catch (err) {
    badge.textContent = '⚠ Offline';
    badge.style.color = '#ff7b72';
  }
}
checkServerHealth();

// --- DRES State & Persistence ---
const dresUrlInput = document.querySelector('#dres-url'),
      dresUserInput = document.querySelector('#dres-user'),
      dresPassInput = document.querySelector('#dres-pass'),
      dresSessionInput = document.querySelector('#dres-session'),
      dresEvalSelect = document.querySelector('#dres-eval-select'),
      dresEvalIdInput = document.querySelector('#dres-eval-id'),
      qaAnswerInput = document.querySelector('#qa-answer');

function loadDresStorage() {
  if (localStorage.getItem('dres_url')) dresUrlInput.value = localStorage.getItem('dres_url');
  if (localStorage.getItem('dres_user')) dresUserInput.value = localStorage.getItem('dres_user');
  if (localStorage.getItem('dres_session')) dresSessionInput.value = localStorage.getItem('dres_session');
  if (localStorage.getItem('dres_eval_id')) dresEvalIdInput.value = localStorage.getItem('dres_eval_id');
  if (dresSessionInput.value && dresEvalIdInput.value) {
    dresStatus.textContent = `✅ Đã nạp phiên DRES: session=${dresSessionInput.value.slice(0, 10)}... | eval=${dresEvalIdInput.value}`;
    dresStatus.style.color = '#7ee787';
  }
}

function saveDresStorage() {
  localStorage.setItem('dres_url', dresUrlInput.value.trim());
  localStorage.setItem('dres_user', dresUserInput.value.trim());
  localStorage.setItem('dres_session', dresSessionInput.value.trim());
  localStorage.setItem('dres_eval_id', dresEvalIdInput.value.trim());
}

[dresUrlInput, dresUserInput, dresSessionInput, dresEvalIdInput].forEach(el => {
  if (el) el.addEventListener('change', saveDresStorage);
});

if (dresEvalSelect) {
  dresEvalSelect.addEventListener('change', () => {
    if (dresEvalSelect.value) {
      dresEvalIdInput.value = dresEvalSelect.value;
      saveDresStorage();
    }
  });
}

function populateEvalDropdown(evaluations) {
  if (!Array.isArray(evaluations) || !evaluations.length) return;
  dresEvalSelect.innerHTML = '<option value="">-- Chọn cuộc thi --</option>' +
    evaluations.map(ev => `<option value="${ev.id}">${ev.name || ev.id} (${ev.status || ''})</option>`).join('');
  const active = evaluations.find(ev => ev.status === 'ACTIVE') || evaluations[0];
  if (active && active.id) {
    dresEvalSelect.value = active.id;
    dresEvalIdInput.value = active.id;
    saveDresStorage();
  }
}

document.querySelector('#dres-login-btn').onclick = async () => {
  const base_url = dresUrlInput.value.trim() || 'https://eventretrieval.one';
  const username = dresUserInput.value.trim();
  const password = dresPassInput.value;
  if (!username || !password) {
    dresStatus.textContent = '⚠ Vui lòng nhập Username và Password DRES!';
    dresStatus.style.color = '#ff7b72';
    return;
  }
  dresStatus.textContent = '⏳ Đang đăng nhập vào DRES...';
  try {
    let sessionId = null;
    let evaluations = [];
    // Try backend proxy first, fallback to direct browser fetch
    const r = await fetch('/api/dres/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url, username, password })
    });
    if (r.ok) {
      const data = await r.json();
      sessionId = data.sessionId;
      evaluations = data.evaluations || [];
    } else {
      const directRes = await fetch(`${base_url.replace(/\/$/, '')}/api/v2/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (!directRes.ok) throw new Error(`HTTP ${directRes.status}`);
      const directData = await directRes.json();
      sessionId = directData.sessionId;
      const evRes = await fetch(`${base_url.replace(/\/$/, '')}/api/v2/client/evaluation/list?session=${encodeURIComponent(sessionId)}`);
      if (evRes.ok) evaluations = await evRes.json();
    }
    if (sessionId) {
      dresSessionInput.value = sessionId;
      populateEvalDropdown(evaluations);
      saveDresStorage();
      dresStatus.textContent = `✅ Đăng nhập DRES thành công! Session: ${sessionId} | Evaluation: ${dresEvalIdInput.value || 'chưa có'}`;
      dresStatus.style.color = '#7ee787';
    }
  } catch (err) {
    dresStatus.textContent = `❌ Lỗi đăng nhập DRES: ${err.message}`;
    dresStatus.style.color = '#ff7b72';
  }
};

document.querySelector('#dres-refresh-eval').onclick = async () => {
  const base_url = dresUrlInput.value.trim() || 'https://eventretrieval.one';
  const session_id = dresSessionInput.value.trim();
  if (!session_id) {
    dresStatus.textContent = '⚠ Cần có Session ID trước khi lấy danh sách Evaluation.';
    return;
  }
  try {
    let evaluations = [];
    const r = await fetch(`/api/dres/evaluations?base_url=${encodeURIComponent(base_url)}&session_id=${encodeURIComponent(session_id)}`);
    if (r.ok) {
      const data = await r.json();
      evaluations = data.evaluations || [];
    } else {
      const direct = await fetch(`${base_url.replace(/\/$/, '')}/api/v2/client/evaluation/list?session=${encodeURIComponent(session_id)}`);
      if (direct.ok) evaluations = await direct.json();
    }
    populateEvalDropdown(evaluations);
    dresStatus.textContent = `✅ Đã tải ${evaluations.length} evaluation(s). Active ID: ${dresEvalIdInput.value}`;
    dresStatus.style.color = '#7ee787';
  } catch (err) {
    dresStatus.textContent = `❌ Không lấy được Evaluation: ${err.message}`;
    dresStatus.style.color = '#ff7b72';
  }
};

loadDresStorage();

async function submitToDres(videoId, timeMs, mode = 'kis', cardEl = null) {
  const base_url = (dresUrlInput.value.trim() || 'https://eventretrieval.one').replace(/\/$/, '');
  const session_id = dresSessionInput.value.trim();
  const evaluation_id = dresEvalIdInput.value.trim();
  const videoStem = cleanVideoStem(videoId);
  const ms = Math.max(0, Math.round(Number(timeMs) || 0));
  const qa_answer = qaAnswerInput.value.trim();

  if (mode === 'qa' && !qa_answer) {
    alert('Bạn đang bấm Nộp Q&A nhưng chưa nhập "Đáp án Q&A" ở thanh trên cùng!');
    qaAnswerInput.focus();
    return;
  }

  const dedupKey = `${mode}:${videoStem}:${Math.round(ms / 500)}:${qa_answer}`;
  if (submittedKeys.has(dedupKey)) {
    if (!confirm(`CẢNH BÁO: Bạn vừa nộp ${videoStem} tại ~${ms}ms rồi! Thể lệ cấm nộp trùng kết quả. Bạn có chắc muốn gửi lại không?`)) {
      return;
    }
  }

  const labelPreview = mode === 'qa'
    ? `Q&A: "${qa_answer}-${videoStem}-${ms}"`
    : `KIS: mediaItemName="${videoStem}", start=${ms}, end=${ms}`;

  if (!session_id || !evaluation_id) {
    await navigator.clipboard.writeText(mode === 'qa' ? `${qa_answer}-${videoStem}-${ms}` : `${videoStem}\t${ms}`);
    dresStatus.textContent = `📋 Chưa điền Session/Evaluation ID DRES — Đã tự động COPY vào bộ nhớ tạm: ${labelPreview}`;
    dresStatus.style.color = '#ffd33d';
    return;
  }

  dresStatus.textContent = `⏳ Đang nộp ${labelPreview}...`;
  try {
    let ok = false;
    let respData = null;
    const r = await fetch('/api/dres/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_url,
        session_id,
        evaluation_id,
        mode,
        video_id: videoStem,
        start_ms: ms,
        end_ms: ms,
        qa_answer
      })
    });
    if (r.ok) {
      ok = true;
      respData = await r.json();
    } else {
      // Fallback direct from browser in case server has no outbound internet
      const body = mode === 'qa'
        ? { answerSets: [{ answers: [{ text: `${qa_answer}-${videoStem}-${ms}` }] }] }
        : { answerSets: [{ answers: [{ mediaItemName: videoStem, start: ms, end: ms }] }] };
      const directRes = await fetch(
        `${base_url}/api/v2/submit/${encodeURIComponent(evaluation_id)}?session=${encodeURIComponent(session_id)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        }
      );
      respData = await directRes.json().catch(() => ({}));
      if (!directRes.ok) throw new Error(JSON.stringify(respData));
      ok = true;
    }

    if (ok) {
      submittedKeys.add(dedupKey);
      if (cardEl) cardEl.classList.add('submitted');
      const dresMsg = respData && respData.dres_response ? JSON.stringify(respData.dres_response) : JSON.stringify(respData);
      dresStatus.textContent = `🎉 ĐÃ NỘP THÀNH CÔNG (${labelPreview}) → Phản hồi DRES: ${dresMsg}`;
      dresStatus.style.color = '#7ee787';
    }
  } catch (err) {
    dresStatus.textContent = `❌ Lỗi khi nộp (${labelPreview}): ${err.message}`;
    dresStatus.style.color = '#ff7b72';
  }
}

function closeInspect() {
  if (player) {
    player.pause();
    player.removeAttribute('src');
    player.load();
  }
  if (inspect && inspect.open) {
    inspect.close();
  }
}

function stepVideo(deltaSeconds) {
  if (!player) return;
  player.pause();
  player.currentTime = Math.max(0, (player.currentTime || 0) + deltaSeconds);
  updateLiveTelemetry();
}

function updateLiveTelemetry() {
  if (!player || !currentInspectItem) return;
  const details = document.querySelector('#details');
  const fps = currentInspectItem.fps || 25.0;
  const curSec = player.currentTime || currentInspectItem.timestamp || 0;
  const curMs = Math.round(curSec * 1000);
  const curFrame = Math.round(curSec * fps);
  const stem = cleanVideoStem(currentInspectItem.video_id);
  const qaVal = qaAnswerInput.value.trim();

  if (details) {
    details.innerHTML =
      `Video: <b style="color:#ffdf5d">${stem}</b> &nbsp;|&nbsp; ` +
      `Mốc hiện tại: <b style="color:#7ee787; font-size:15px;">${curMs} ms</b> (${curSec.toFixed(2)}s) &nbsp;|&nbsp; ` +
      `Frame ước tính: <b>#${curFrame}</b> (FPS: ${fps})` +
      (qaVal ? ` &nbsp;|&nbsp; Chuỗi Q&A: <b style="color:#ffd33d">${qaVal}-${stem}-${curMs}</b>` : '');
  }

  const kisBtn = document.querySelector('#modal-submit-kis');
  if (kisBtn) kisBtn.textContent = `🚀 Nộp KIS (${stem} @ ${curMs} ms)`;
  const qaBtn = document.querySelector('#modal-submit-qa');
  if (qaBtn) qaBtn.textContent = `❓ Nộp Q&A (${qaVal || '...'}-${stem}-${curMs})`;
}

if (player) {
  ['timeupdate', 'seeking', 'seeked', 'pause', 'loadedmetadata'].forEach(ev => {
    player.addEventListener(ev, updateLiveTelemetry);
  });
}

if (inspect) {
  inspect.addEventListener('close', () => {
    if (player) {
      player.pause();
      player.removeAttribute('src');
      player.load();
    }
  });
  inspect.addEventListener('click', (e) => {
    if (e.target === inspect) closeInspect();
  });
}

document.querySelector('#modal-copy-btn').onclick = async () => {
  if (!currentInspectItem) return;
  const stem = cleanVideoStem(currentInspectItem.video_id);
  const curMs = Math.round((player.currentTime || currentInspectItem.timestamp || 0) * 1000);
  const qaVal = qaAnswerInput.value.trim();
  const textToCopy = qaVal ? `${qaVal}-${stem}-${curMs}` : `${stem}\t${curMs}`;
  await navigator.clipboard.writeText(textToCopy);
  dresStatus.textContent = `📋 Đã copy vào bộ nhớ tạm: ${textToCopy}`;
  dresStatus.style.color = '#7ee787';
};

document.querySelector('#modal-submit-kis').onclick = () => {
  if (!currentInspectItem) return;
  const curMs = Math.round((player.currentTime || currentInspectItem.timestamp || 0) * 1000);
  submitToDres(currentInspectItem.video_id, curMs, 'kis');
};

document.querySelector('#modal-submit-qa').onclick = () => {
  if (!currentInspectItem) return;
  const curMs = Math.round((player.currentTime || currentInspectItem.timestamp || 0) * 1000);
  submitToDres(currentInspectItem.video_id, curMs, 'qa');
};

function addConditionToScene(conditionsListEl, value = { text: '', region: 'all', weight: 1, expanded: '' }) {
  const wrapper = document.createElement('div');
  wrapper.className = 'condition-wrapper';
  wrapper.innerHTML = `
    <div class="condition">
      <input class="query-text" aria-label="Mô tả" placeholder="Nhập mô tả tiếng Việt hoặc tiếng Anh (Nhấn Enter để tìm ngay)..." value="${value.text}">
      <select class="region-select">
        ${regionDefs.map(x => `<option value="${x.id}" ${x.id === value.region ? 'selected' : ''}>${x.label}</option>`).join('')}
      </select>
      <input class="weight-input" aria-label="Trọng số" type="number" min="0.1" max="10" step="0.1" value="${value.weight}" title="Trọng số (Weight)">
      <button class="expand-btn" type="button" title="Dịch sang tiếng Anh / Mở rộng">🌐 Dịch EN</button>
      <button class="delete-btn" type="button" title="Xóa điều kiện này">×</button>
    </div>
    <div class="expanded-row">
      <span class="expanded-label">✨ Query EN (có thể sửa tay):</span>
      <input class="expanded-input" placeholder="Để trống hệ thống sẽ tự dịch sang tiếng Anh khi tìm kiếm, hoặc nhập trực tiếp tiếng Anh..." value="${value.expanded}">
    </div>
  `;

  const queryInput = wrapper.querySelector('.query-text');
  const expandedInput = wrapper.querySelector('.expanded-input');
  const expandBtn = wrapper.querySelector('.expand-btn');
  const deleteBtn = wrapper.querySelector('.delete-btn');

  deleteBtn.onclick = () => {
    if (conditionsListEl.querySelectorAll('.condition-wrapper').length > 1) {
      wrapper.remove();
    } else {
      queryInput.value = '';
      expandedInput.value = '';
    }
  };

  async function triggerExpand() {
    const text = queryInput.value.trim();
    if (!text) return;
    expandBtn.textContent = '⏳...';
    try {
      const res = await fetch('/api/expand', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (res.ok) {
        const data = await res.json();
        const english = (data.variants && data.variants.length > 1) ? data.variants[data.variants.length - 1] : (data.variants ? data.variants[0] : text);
        expandedInput.value = english;
        expandedInput.dataset.auto = 'true';
      }
    } catch (e) {
      console.error(e);
    } finally {
      expandBtn.textContent = '🌐 Dịch EN';
    }
  }

  expandBtn.onclick = triggerExpand;

  queryInput.addEventListener('input', () => {
    if (expandedInput.dataset.auto === 'true') {
      expandedInput.value = '';
    }
  });
  expandedInput.addEventListener('input', () => {
    expandedInput.dataset.auto = 'manual';
  });

  [queryInput, expandedInput].forEach(inp => {
    inp.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        runSearch();
      }
    });
  });

  conditionsListEl.append(wrapper);
  queryInput.focus();
}

function refreshSceneLabels() {
  const sceneBoxes = [...scenesContainer.querySelectorAll('.scene-box')];
  sceneBoxes.forEach((sceneBox, idx) => {
    const stageNum = idx + 1;
    sceneBox.dataset.stage = String(stageNum);
    sceneBox.classList.toggle('scene-subsequent', stageNum > 1);
    const titleEl = sceneBox.querySelector('.scene-title');
    const addCondBtn = sceneBox.querySelector('.add-cond-btn');
    const removeSceneBtn = sceneBox.querySelector('.remove-scene-btn');
    if (stageNum === 1) {
      titleEl.textContent = '🎬 Cảnh 1 (Khung hình chính — Các điều kiện bên dưới xuất hiện trong CÙNG Cảnh 1)';
    } else {
      titleEl.textContent = `⏩ Cảnh ${stageNum} (Xảy ra tiếp theo sau Cảnh ${stageNum - 1} trong cùng Video)`;
    }
    if (addCondBtn) {
      addCondBtn.textContent = `+ Thêm điều kiện (Cùng Cảnh ${stageNum})`;
    }
    if (removeSceneBtn) {
      removeSceneBtn.style.display = stageNum > 1 ? 'inline-block' : 'none';
    }
  });
}

function addSceneBox() {
  const currentCount = scenesContainer.querySelectorAll('.scene-box').length;
  const stageNum = currentCount + 1;
  const sceneBox = document.createElement('div');
  sceneBox.className = 'scene-box' + (stageNum > 1 ? ' scene-subsequent' : '');
  sceneBox.dataset.stage = String(stageNum);

  sceneBox.innerHTML = `
    <div class="scene-header">
      <span class="scene-title"></span>
      <div class="scene-actions">
        <button type="button" class="add-cond-btn">+ Thêm điều kiện</button>
        <button type="button" class="btn-danger remove-scene-btn" title="Xóa toàn bộ Cảnh này">🗑️ Xóa Cảnh</button>
      </div>
    </div>
    <div class="scene-conditions"></div>
  `;

  const conditionsListEl = sceneBox.querySelector('.scene-conditions');
  sceneBox.querySelector('.add-cond-btn').onclick = () => addConditionToScene(conditionsListEl);
  sceneBox.querySelector('.remove-scene-btn').onclick = () => {
    sceneBox.remove();
    refreshSceneLabels();
  };

  scenesContainer.append(sceneBox);
  refreshSceneLabels();
  addConditionToScene(conditionsListEl);
}

function getConditions() {
  const all = [];
  const sceneBoxes = [...scenesContainer.querySelectorAll('.scene-box')];
  sceneBoxes.forEach((sceneBox, idx) => {
    const stage = idx + 1;
    const wrappers = [...sceneBox.querySelectorAll('.condition-wrapper')];
    for (const wrapper of wrappers) {
      const text = wrapper.querySelector('.query-text').value.trim();
      const region = wrapper.querySelector('.region-select').value;
      const weight = Number(wrapper.querySelector('.weight-input').value) || 1.0;
      const expanded = wrapper.querySelector('.expanded-input').value.trim();
      if (text || expanded) {
        all.push({
          text: text || expanded,
          region,
          weight,
          stage: Math.min(stage, 10),
          variants: expanded ? [expanded] : null
        });
      }
    }
  });
  return all;
}

document.querySelector('#add-scene').onclick = () => addSceneBox();
addSceneBox();

// --- CSV / OCR Search & Video Intersection ---
const csvMetaBadge = document.querySelector('#csv-meta-badge');
const csvFileInput = document.querySelector('#csv-file-input');
const csvUploadBtn = document.querySelector('#csv-upload-btn');
const csvQueryInput = document.querySelector('#csv-query-input');
const csvSearchBtn = document.querySelector('#csv-search-btn');
const useVideoFilterChk = document.querySelector('#use-video-filter');
const videoFilterInput = document.querySelector('#video-filter-input');
const filterL21L24Btn = document.querySelector('#filter-l21-l24-btn');
const filterClearBtn = document.querySelector('#filter-clear-btn');
const csvResultsBar = document.querySelector('#csv-results-bar');
const ocrResultsSection = document.querySelector('#ocr-results-section');

let lastCsvQuery = '';
let lastCsvMatches = [];
let ocrShowAll = false;

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const REGION_NAMES = {
  0: "Toàn ảnh (all)", 1: "Top-Left (1)", 2: "Top-Center (2)", 3: "Top-Right (3)",
  4: "Mid-Left (4)", 5: "Center (5)", 6: "Mid-Right (6)", 7: "Bot-Left (7)",
  8: "Bot-Center (8)", 9: "Bot-Right (9)"
};

function createResultCard(item, opts = {}) {
  const isOcr = Boolean(opts.isOcr);
  const card = document.createElement('article');
  card.className = isOcr ? 'card ocr-card' : 'card';

  const stem = item.video_stem || cleanVideoStem(item.video_id);
  const ms = item.timestamp_ms !== undefined ? item.timestamp_ms : Math.round((item.timestamp || 0) * 1000);
  const tsNum = item.timestamp != null ? Number(item.timestamp) : 0;
  const imgRel = item.image_path
    ? item.image_path.replace(/^images\//, '')
    : (item.frame_id != null ? `${stem}/frame_${item.frame_id}.webp` : '');
  const imgSrc = imgRel ? `/media/keyframe/${imgRel}` : '';

  const nextList = item.next_matches || (item.next_match ? [{ stage: 2, ...item.next_match }] : []);
  let nextTag = '';
  let stage1LabelHtml = '';
  let nextPreviewHtml = '';
  if (!isOcr && nextList.length > 0) {
    stage1LabelHtml = `<div style="font-size:11px; color:#79c0ff; text-align:center; margin-top:2px;">🎬 Cảnh 1: ${ms} ms (#${item.frame_id})</div>`;
    nextTag = nextList
      .map((nm, idx) => `<span class="card-tag next-tag" data-next-idx="${idx}" title="Bấm để xem Cảnh ${nm.stage || idx + 2}">⏩ Cảnh ${nm.stage || idx + 2} @ ${nm.timestamp_ms}ms</span>`)
      .join(' ');
    nextPreviewHtml = nextList
      .map((nm, idx) => {
        const nextImg = `/media/keyframe/${nm.image_path.replace(/^images\//, '')}`;
        const stNum = nm.stage || idx + 2;
        return `
          <div class="next-scene-box" data-next-idx="${idx}" title="Ảnh của Cảnh ${stNum} tiếp theo tại ${nm.timestamp_ms} ms">
            <img src="${nextImg}" loading="lazy" style="aspect-ratio:16/9; width:100%; border-radius:4px; border:1px solid #8957e5; margin-top:4px;">
            <div style="font-size:11px; color:#d2a8ff; text-align:center;">⏩ Cảnh ${stNum}: ${nm.timestamp_ms} ms (#${nm.frame_id})</div>
          </div>
        `;
      })
      .join('');
  }

  const submetaHtml = isOcr
    ? `<span class="ocr-badge">📝 Khớp OCR</span><span>${tsNum.toFixed(2)}s</span>${item.frame_id != null ? `<span>· #${item.frame_id}</span>` : ''}`
    : `<span>${tsNum.toFixed(2)}s</span><span>· #${item.frame_id}</span><span>· Score: <b>${Number(item.score || 0).toFixed(3)}</b></span><span class="card-tag">Vùng: <b>${REGION_NAMES[item.region_id] || item.region_id}</b></span>${nextTag}`;

  const ocrTextHtml = isOcr && item.text
    ? `<div class="ocr-text-snippet" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>`
    : '';

  card.innerHTML = `
    ${imgSrc ? `<img class="stage1-img" src="${imgSrc}" alt="${escapeHtml(item.keyframe_id || stem)}" loading="lazy" title="Bấm để xem video và các frame lân cận">` : ''}
    ${stage1LabelHtml}
    ${nextPreviewHtml}
    <div class="card-meta">
      <span class="video-pill">${escapeHtml(stem)}</span>
      <span class="ms-pill">${ms} ms</span>
    </div>
    <div class="card-submeta">${submetaHtml}</div>
    ${ocrTextHtml}
    <div class="card-actions">
      <button type="button" class="view-btn">▶ Xem Video & Lân cận</button>
      <button type="button" class="btn-secondary copy-btn" title="Copy mã Video + ms">📋</button>
    </div>
    <div class="card-submit-row">
      <button type="button" class="btn-submit-kis kis-btn">🚀 Nộp KIS</button>
      <button type="button" class="btn-submit-qa qa-btn">❓ Nộp Q&A</button>
    </div>
  `;

  const imgEl = card.querySelector('.stage1-img');
  if (imgEl) imgEl.onclick = () => inspectResult(item);
  card.querySelectorAll('.next-scene-box, .next-tag').forEach((el) => {
    el.onclick = (e) => {
      e.stopPropagation();
      const idx = Number(el.dataset.nextIdx) || 0;
      if (nextList[idx]) inspectResult({ ...item, ...nextList[idx] });
    };
  });
  card.querySelector('.view-btn').onclick = () => inspectResult(item);
  card.querySelector('.copy-btn').onclick = async () => {
    const qaVal = qaAnswerInput.value.trim();
    const txt = qaVal ? `${qaVal}-${stem}-${ms}` : `${stem}\t${ms}`;
    await navigator.clipboard.writeText(txt);
    dresStatus.textContent = `📋 Đã copy: ${txt}`;
    dresStatus.style.color = '#7ee787';
  };
  card.querySelector('.kis-btn').onclick = () => submitToDres(stem, ms, 'kis', card);
  card.querySelector('.qa-btn').onclick = () => submitToDres(stem, ms, 'qa', card);
  return card;
}

function renderOcrMatchesSection(query, matches, videoCount) {
  if (!ocrResultsSection) return;
  if (!query || !matches || matches.length === 0) {
    ocrResultsSection.style.display = 'none';
    ocrResultsSection.innerHTML = '';
    return;
  }
  const initialLimit = 8;
  const visibleMatches = ocrShowAll ? matches : matches.slice(0, initialLimit);
  ocrResultsSection.style.display = 'block';
  ocrResultsSection.innerHTML = `
    <div class="ocr-section-header">
      <span>📝 Khung hình khớp bộ lọc OCR "${escapeHtml(query)}" — ${matches.length} frame (${videoCount} video)</span>
      <div style="display:flex; gap:6px; align-items:center;">
        ${matches.length > initialLimit ? `<button type="button" id="ocr-toggle-all-btn" class="btn-secondary">${ocrShowAll ? 'Thu gọn (8 frame)' : `Hiện tất cả (${matches.length} frame)`}</button>` : ''}
        <button type="button" id="ocr-hide-btn" class="btn-secondary" title="Ẩn khung OCR">✕ Ẩn</button>
      </div>
    </div>
    <div class="results ocr-grid"></div>
  `;
  const grid = ocrResultsSection.querySelector('.ocr-grid');
  for (const m of visibleMatches) {
    const item = {
      video_id: m.video_id,
      video_stem: m.video_id,
      keyframe_id: `${m.video_id}:${m.frame_id ?? 0}`,
      frame_id: m.frame_id,
      timestamp: m.timestamp != null ? Number(m.timestamp) : 0,
      timestamp_ms: m.timestamp_ms != null ? m.timestamp_ms : (m.timestamp != null ? Math.round(Number(m.timestamp) * 1000) : 0),
      image_path: m.image_path || (m.frame_id != null ? `images/${m.video_id}/frame_${m.frame_id}.webp` : ''),
      text: m.text || '',
      score: 1.0,
    };
    grid.append(createResultCard(item, { isOcr: true }));
  }
  const toggleBtn = ocrResultsSection.querySelector('#ocr-toggle-all-btn');
  if (toggleBtn) {
    toggleBtn.onclick = () => {
      ocrShowAll = !ocrShowAll;
      renderOcrMatchesSection(query, matches, videoCount);
    };
  }
  const hideBtn = ocrResultsSection.querySelector('#ocr-hide-btn');
  if (hideBtn) {
    hideBtn.onclick = () => {
      ocrResultsSection.style.display = 'none';
    };
  }
}

async function refreshCsvStatus() {
  if (!csvMetaBadge) return;
  try {
    const r = await fetch('/api/csv/status');
    if (!r.ok) return;
    const data = await r.json();
    if (data.total_rows > 0) {
      const fNames = (data.files || []).map((f) => `${f.name} (${f.rows})`).join(', ');
      csvMetaBadge.textContent = `✅ Đã nạp CSV: ${data.total_rows} dòng (${data.unique_videos} video) [${fNames}]`;
      csvMetaBadge.style.color = '#7ee787';
    } else {
      csvMetaBadge.textContent = '⚠ Chưa có file .csv trên server (Bấm "📂 Nạp file .CSV" để tải lên)';
      csvMetaBadge.style.color = '#ffd33d';
    }
  } catch (_) {}
}
refreshCsvStatus();

if (csvUploadBtn && csvFileInput) {
  csvUploadBtn.onclick = () => csvFileInput.click();
  csvFileInput.onchange = async () => {
    const file = csvFileInput.files && csvFileInput.files[0];
    if (!file) return;
    csvMetaBadge.textContent = `⏳ Đang tải ${file.name} lên server...`;
    try {
      const content = await file.text();
      const r = await fetch('/api/csv/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, content })
      });
      if (r.ok) {
        await refreshCsvStatus();
        if (csvQueryInput.value.trim()) {
          await runCsvSearch(false);
        }
      } else {
        csvMetaBadge.textContent = `❌ Lỗi nạp CSV (${r.status})`;
        csvMetaBadge.style.color = '#ff7b72';
      }
    } catch (err) {
      csvMetaBadge.textContent = `❌ Lỗi nạp CSV: ${err.message}`;
      csvMetaBadge.style.color = '#ff7b72';
    }
  };
}

async function runCsvSearch(triggerVisualSearch = false) {
  const q = csvQueryInput ? csvQueryInput.value.trim() : '';
  if (!q) {
    lastCsvQuery = '';
    lastCsvMatches = [];
    if (csvResultsBar) csvResultsBar.style.display = 'none';
    renderOcrMatchesSection('', [], 0);
    return [];
  }
  try {
    const r = await fetch('/api/csv/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, limit: 80 })
    });
    if (!r.ok) return [];
    const data = await r.json();
    const vids = data.video_ids || [];
    lastCsvQuery = q;
    lastCsvMatches = data.matches || [];
    ocrShowAll = false;
    if (videoFilterInput) {
      videoFilterInput.value = vids.join(', ');
    }
    if (useVideoFilterChk && vids.length > 0) {
      useVideoFilterChk.checked = true;
    }
    if (csvResultsBar) {
      csvResultsBar.style.display = 'block';
      if (data.total_rows === 0) {
        csvResultsBar.innerHTML = `<span style="color:#ffd33d">⚠ Trên server chưa có dữ liệu CSV nào. Hãy bấm nút <b>📂 Nạp file .CSV lên Server</b> ở góc phải!</span>`;
        renderOcrMatchesSection('', [], 0);
      } else if (vids.length === 0) {
        csvResultsBar.innerHTML = `<span style="color:#ff7b72">❌ Không tìm thấy từ khóa <b>"${escapeHtml(q)}"</b> trong ${data.total_rows} dòng CSV. (Tạm tắt lọc giao để không bị trống kết quả)</span>`;
        if (useVideoFilterChk) useVideoFilterChk.checked = false;
        renderOcrMatchesSection('', [], 0);
      } else {
        const chipsHtml = lastCsvMatches
          .slice(0, 25)
          .map((m, idx) => {
            const tsLabel = m.timestamp != null ? ` @ ${Number(m.timestamp).toFixed(1)}s` : m.frame_id != null ? ` #${m.frame_id}` : '';
            const safeText = escapeHtml(m.text || '');
            return `<span class="csv-snippet-chip" data-idx="${idx}" title="${safeText}">🎬 ${escapeHtml(m.video_id)}${tsLabel}: ${escapeHtml((m.text || '').slice(0, 48))}</span>`;
          })
          .join('');
        csvResultsBar.innerHTML =
          `<div style="margin-bottom:4px; color:#7ee787;">✅ Tìm thấy <b>${vids.length} mã video</b> khớp từ khóa <b>"${escapeHtml(q)}"</b> (Đã điền vào ô Giao ∩ &amp; hiển thị các frame khớp OCR bên dưới):</div>` +
          `<div>${chipsHtml}</div>`;
        csvResultsBar.querySelectorAll('.csv-snippet-chip').forEach((chip) => {
          chip.onclick = () => {
            const m = lastCsvMatches[Number(chip.dataset.idx)];
            if (m) {
              inspectResult({
                video_id: m.video_id,
                video_stem: m.video_id,
                frame_id: m.frame_id || 0,
                timestamp: m.timestamp != null ? Number(m.timestamp) : 0,
                timestamp_ms: m.timestamp_ms != null ? m.timestamp_ms : (m.timestamp != null ? Math.round(Number(m.timestamp) * 1000) : 0),
                fps: 25.0,
                score: 1.0
              });
            }
          };
        });
        renderOcrMatchesSection(q, lastCsvMatches, vids.length);
      }
    }
    if (triggerVisualSearch && getConditions().length > 0) {
      await runSearch(true);
    }
    return vids;
  } catch (_) {
    return [];
  }
}

if (csvSearchBtn) {
  csvSearchBtn.onclick = () => runCsvSearch(true);
}
if (csvQueryInput) {
  csvQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      runCsvSearch(true);
    }
  });
}
if (filterL21L24Btn) {
  filterL21L24Btn.onclick = () => {
    videoFilterInput.value = 'L21-L24';
    useVideoFilterChk.checked = true;
    if (getConditions().length > 0) runSearch(true);
  };
}
if (filterClearBtn) {
  filterClearBtn.onclick = () => {
    videoFilterInput.value = '';
    lastCsvQuery = '';
    lastCsvMatches = [];
    if (csvQueryInput) csvQueryInput.value = '';
    if (csvResultsBar) csvResultsBar.style.display = 'none';
    renderOcrMatchesSection('', [], 0);
    if (getConditions().length > 0) runSearch(true);
  };
}

async function runSearch(skipAutoCsv = false) {
  const isAutoExpand = document.querySelector('#expand').checked;
  const isDiversify = document.querySelector('#diversify').checked;
  const maxTemporalGap = Number(document.querySelector('#temporal-gap').value) || 45.0;

  const csvQ = csvQueryInput ? csvQueryInput.value.trim() : '';
  if (!skipAutoCsv && csvQ && (csvQ !== lastCsvQuery || (videoFilterInput && !videoFilterInput.value.trim()))) {
    await runCsvSearch(false);
  }

  if (isAutoExpand) {
    const wrappers = [...document.querySelectorAll('.condition-wrapper')];
    const pending = wrappers.filter((w) => {
      const q = w.querySelector('.query-text').value.trim();
      const exp = w.querySelector('.expanded-input');
      return q && !exp.value.trim();
    });
    if (pending.length) {
      status.textContent = '⏳ Đang dịch sang tiếng Anh...';
      await Promise.all(
        pending.map(async (wrapper) => {
          const q = wrapper.querySelector('.query-text').value.trim();
          const exp = wrapper.querySelector('.expanded-input');
          try {
            const res = await fetch('/api/expand', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: q })
            });
            if (res.ok) {
              const data = await res.json();
              const english =
                data.variants && data.variants.length > 1
                  ? data.variants[data.variants.length - 1]
                  : data.variants
                    ? data.variants[0]
                    : q;
              exp.value = english;
              exp.dataset.auto = 'true';
            }
          } catch (_) {}
        })
      );
    }
  }

  const c = getConditions();
  if (!c.length) {
    status.textContent = '⚠ Vui lòng nhập ít nhất 1 điều kiện tìm kiếm.';
    return;
  }

  status.textContent = '⏳ Đang truy vấn trên 633,600 keyframes...';
  results.innerHTML = '';

  const activeFilterStr =
    useVideoFilterChk && useVideoFilterChk.checked && videoFilterInput ? videoFilterInput.value.trim() : '';

  const request = {
    conditions: c,
    diversify: isDiversify,
    expand: isAutoExpand,
    max_temporal_gap: maxTemporalGap,
    video_filter: activeFilterStr ? [activeFilterStr] : null
  };

  const t0 = performance.now();
  try {
    const r = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    const lat = Math.round(performance.now() - t0);
    const textBody = await r.text();
    let data = {};
    try {
      data = JSON.parse(textBody);
    } catch (_) {
      status.textContent = `❌ Lỗi máy chủ (${r.status}): ${textBody.slice(0, 140)}`;
      return;
    }

    if (!r.ok) {
      status.textContent = '❌ Lỗi: ' + (data.detail || JSON.stringify(data));
      return;
    }

    status.textContent = `✅ Hiển thị ${data.results.length} kết quả tốt nhất (${lat} ms)`;

    for (const item of data.results) {
      results.append(createResultCard(item, { isOcr: false }));
    }
  } catch (err) {
    status.textContent = '❌ Lỗi kết nối máy chủ: ' + err.message;
  }
}

document.querySelector('#search').onclick = runSearch;

async function inspectResult(item) {
  if (!player || !inspect) return;
  currentInspectItem = item;
  player.pause();
  player.src = '/media/video/' + encodeURIComponent(item.video_id);
  const startSec = Math.max(0, item.timestamp || 0);
  player.onloadedmetadata = () => {
    player.currentTime = startSec;
    updateLiveTelemetry();
    player.play().catch(() => {});
  };
  if (player.readyState >= 1) {
    player.currentTime = startSec;
    updateLiveTelemetry();
    player.play().catch(() => {});
  }

  updateLiveTelemetry();
  inspect.showModal();

  // Load neighboring keyframes strip
  const strip = document.querySelector('#neighbors-strip');
  if (strip) {
    strip.innerHTML = '<span style="color:#8b949e; font-size:12px;">Đang tải các frame lân cận...</span>';
    try {
      const r = await fetch(`/api/neighbors?video_id=${encodeURIComponent(item.video_id)}&timestamp=${item.timestamp}&window=35`);
      if (r.ok) {
        const data = await r.json();
        strip.innerHTML = '';
        for (const nb of (data.neighbors || [])) {
          const div = document.createElement('div');
          div.className = 'neighbor-item' + (nb.keyframe_id === item.keyframe_id ? ' active' : '');
          const nbImg = `/media/keyframe/${nb.image_path.replace(/^images\//, '')}`;
          const nbMs = nb.timestamp_ms !== undefined ? nb.timestamp_ms : Math.round(nb.timestamp * 1000);
          div.innerHTML = `
            <img src="${nbImg}" loading="lazy">
            <div><b>${nbMs} ms</b></div>
            <div style="color:#8b949e">${nb.timestamp.toFixed(1)}s · #${nb.frame_id}</div>
          `;
          div.onclick = () => {
            currentInspectItem = { ...item, ...nb };
            player.currentTime = nb.timestamp;
            updateLiveTelemetry();
            [...strip.querySelectorAll('.neighbor-item')].forEach(el => el.classList.remove('active'));
            div.classList.add('active');
          };
          strip.append(div);
        }
      }
    } catch (_) {
      strip.innerHTML = '';
    }
  }
}
