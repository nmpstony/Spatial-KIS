const regionDefs = [
  { id: "all", label: "all (Toàn khung hình)" },
  { id: "left", label: "left (Bên trái: cột 1, 4, 7)" },
  { id: "center", label: "center (Chính giữa: ô 5)" },
  { id: "right", label: "right (Bên phải: cột 3, 6, 9)" },
  { id: "top", label: "top (Phía trên: hàng 1, 2, 3)" },
  { id: "bottom", label: "bottom (Phía dưới: hàng 7, 8, 9)" },
  { id: "top_left", label: "top_left (Góc trên trái: ô 1)" },
  { id: "top_right", label: "top_right (Góc trên phải: ô 3)" },
  { id: "bottom_left", label: "bottom_left (Góc dưới trái: ô 7)" },
  { id: "bottom_right", label: "bottom_right (Góc dưới phải: ô 9)" }
];

const box = document.querySelector('#conditions'),
      results = document.querySelector('#results'),
      status = document.querySelector('#status'),
      inspect = document.querySelector('#inspect'),
      player = document.querySelector('#player');

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

if (inspect) {
  inspect.addEventListener('close', () => {
    if (player) {
      player.pause();
      player.removeAttribute('src');
      player.load();
    }
  });

  inspect.addEventListener('click', (e) => {
    if (e.target === inspect) {
      closeInspect();
    }
  });
}

function addCondition(target, value = { text: '', region: 'all', weight: 1, expanded: '' }) {
  const wrapper = document.createElement('div');
  wrapper.className = 'condition-wrapper';
  wrapper.innerHTML = `
    <div class="condition">
      <input class="query-text" aria-label="Mô tả" placeholder="Nhập mô tả (VD: xe hơi màu trắng, người đi bộ...)" value="${value.text}">
      <select class="region-select">
        ${regionDefs.map(x => `<option value="${x.id}" ${x.id === value.region ? 'selected' : ''}>${x.label}</option>`).join('')}
      </select>
      <input class="weight-input" aria-label="Trọng số" type="number" min="0.1" max="10" step="0.1" value="${value.weight}" title="Trọng số (Weight)">
      <button class="expand-btn" type="button" title="Dịch sang tiếng Anh / Mở rộng">🌐 Dịch / Làm giàu</button>
      <button class="delete-btn" type="button" title="Xóa điều kiện">×</button>
    </div>
    <div class="expanded-row">
      <span class="expanded-label">✨ Query sau khi làm giàu (EN):</span>
      <input class="expanded-input" placeholder="Query tiếng Anh dùng để tìm kiếm (tự động dịch khi bật tuỳ chọn hoặc bấm nút bên trên)..." value="${value.expanded}">
    </div>
  `;

  const queryInput = wrapper.querySelector('.query-text');
  const expandedInput = wrapper.querySelector('.expanded-input');
  const expandBtn = wrapper.querySelector('.expand-btn');
  const deleteBtn = wrapper.querySelector('.delete-btn');

  deleteBtn.onclick = () => wrapper.remove();

  async function triggerExpand() {
    const text = queryInput.value.trim();
    if (!text) return;
    expandBtn.textContent = '⏳ Đang dịch...';
    try {
      const res = await fetch('/api/expand', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });
      if (res.ok) {
        const data = await res.json();
        const english = (data.variants && data.variants.length > 1) ? data.variants[1] : (data.variants ? data.variants[0] : text);
        expandedInput.value = english;
      }
    } catch (e) {
      console.error(e);
    } finally {
      expandBtn.textContent = '🌐 Dịch / Làm giàu';
    }
  }

  expandBtn.onclick = triggerExpand;

  queryInput.addEventListener('change', () => {
    const autoExpand = document.querySelector('#expand').checked;
    if (autoExpand && queryInput.value.trim() && !expandedInput.value.trim()) {
      triggerExpand();
    }
  });

  target.append(wrapper);
}

function getConditions(target) {
  return [...target.querySelectorAll('.condition-wrapper')].map(wrapper => {
    const text = wrapper.querySelector('.query-text').value.trim();
    const region = wrapper.querySelector('.region-select').value;
    const weight = Number(wrapper.querySelector('.weight-input').value) || 1.0;
    const expanded = wrapper.querySelector('.expanded-input').value.trim();

    const variants = expanded ? [expanded] : (text ? [text] : []);
    return {
      text: text || expanded,
      region: region,
      weight: weight,
      variants: variants.length ? variants : null
    };
  }).filter(x => x.text);
}

document.querySelector('#add').onclick = () => addCondition(box);
addCondition(box);

document.querySelector('#search').onclick = async () => {
  const isAutoExpand = document.querySelector('#expand').checked;
  const isDiversify = document.querySelector('#diversify').checked;

  if (isAutoExpand) {
    const wrappers = [...document.querySelectorAll('.condition-wrapper')];
    for (const wrapper of wrappers) {
      const q = wrapper.querySelector('.query-text').value.trim();
      const exp = wrapper.querySelector('.expanded-input');
      if (q && !exp.value.trim()) {
        try {
          const res = await fetch('/api/expand', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: q })
          });
          if (res.ok) {
            const data = await res.json();
            const english = (data.variants && data.variants.length > 1) ? data.variants[1] : (data.variants ? data.variants[0] : q);
            exp.value = english;
          }
        } catch (_) {}
      }
    }
  }

  const c = getConditions(box);
  if (!c.length) {
    status.textContent = 'Vui lòng nhập ít nhất 1 điều kiện tìm kiếm.';
    return;
  }

  status.textContent = 'Đang tìm kiếm trên vector database...';
  results.innerHTML = '';

  const request = {
    conditions: c,
    diversify: isDiversify,
    expand: isAutoExpand
  };

  const t0 = performance.now();
  try {
    const r = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    const lat = Math.round(performance.now() - t0);
    const data = await r.json();

    if (!r.ok) {
      status.textContent = 'Lỗi: ' + (data.detail || JSON.stringify(data));
      return;
    }

    status.textContent = `Tìm thấy ${data.results.length} kết quả (${lat} ms)`;

    for (const item of data.results) {
      const card = document.createElement('article');
      card.className = 'card';

      const imgA = `/media/keyframe/${item.image_path.replace('images/', '')}`;

      const regionNames = {
        0: "Toàn ảnh (all)", 1: "Top-Left (ô 1)", 2: "Top (ô 2)", 3: "Top-Right (ô 3)",
        4: "Left (ô 4)", 5: "Center (ô 5)", 6: "Right (ô 6)", 7: "Bottom-Left (ô 7)",
        8: "Bottom (ô 8)", 9: "Bottom-Right (ô 9)"
      };
      const regionTag = `<span class="card-tag">Vùng khớp: <b>${regionNames[item.region_id] || item.region_id}</b></span>`;

      card.innerHTML = `
        <img src="${imgA}" alt="${item.keyframe_id}" loading="lazy">
        <div style="margin-top:6px;">${regionTag}</div>
        <small>
          <b>${item.video_id}</b> · ${item.timestamp.toFixed(2)}s · Score: <b>${item.score.toFixed(3)}</b>
        </small>
        <div style="margin-top:auto; display:flex; gap:6px; padding-top:8px;">
          <button style="flex:1" class="view-btn">▶ Xem video</button>
          <button title="Chính xác" class="fb-ok">✓</button>
          <button title="Không liên quan" class="fb-no">×</button>
        </div>
      `;

      card.querySelector('.view-btn').onclick = () => inspectResult(item);
      card.querySelector('.fb-ok').onclick = () => feedback(item, 'correct');
      card.querySelector('.fb-no').onclick = () => feedback(item, 'irrelevant');
      results.append(card);
    }
  } catch (err) {
    status.textContent = 'Lỗi kết nối máy chủ: ' + err.message;
  }
};

function inspectResult(item) {
  if (!player || !inspect) return;
  player.pause();
  player.src = '/media/video/' + encodeURIComponent(item.video_id);
  player.onloadedmetadata = () => {
    player.currentTime = Math.max(0, (item.timestamp || 0) - 1.0);
    player.play().catch(e => console.log('Autoplay:', e));
  };
  if (player.readyState >= 1) {
    player.currentTime = Math.max(0, (item.timestamp || 0) - 1.0);
    player.play().catch(() => {});
  }

  const details = document.querySelector('#details');
  if (details) {
    details.innerHTML = `<strong>Video:</strong> ${item.video_id} &nbsp;|&nbsp; <strong>Frame:</strong> ${item.frame_id} &nbsp;|&nbsp; <strong>Thời gian:</strong> ${item.timestamp.toFixed(2)}s &nbsp;|&nbsp; <strong>Score:</strong> ${item.score.toFixed(3)}`;
  }
  inspect.showModal();
}

async function feedback(item, label) {
  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keyframe_id: item.keyframe_id, label })
  });
}
