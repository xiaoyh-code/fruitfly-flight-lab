import * as THREE from 'three';

// Recorded MuJoCo states, translated by the exporter into the PLY's coordinates.
// Playback never calls a policy or runs physics in the viewer.
export function validateReplay(data) {
  const vector = (value, size) => Array.isArray(value) && value.length === size && value.every(Number.isFinite);
  if (data?.version !== 1 || data.scene_id !== 'hall' || !Array.isArray(data.episodes) || !data.episodes.length) throw Error('回放資料格式不完整');
  if (!vector(data.proxy?.center_xyz, 3) || !(data.proxy.radius_m > 0) || !(data.proxy.height_m > 0) || !(data.vehicle?.envelope_radius_m > 0)) throw Error('缺少障礙代理或機身尺寸');
  for (const episode of data.episodes) {
    const points = episode.trajectory_xyz, times = episode.times_s;
    if (!Array.isArray(points) || points.length < 2 || !points.every(point => vector(point, 3)) || !Array.isArray(times) || times.length !== points.length || times[0] !== 0 || !times.every((time, i) => Number.isFinite(time) && (i === 0 || time > times[i - 1]))) throw Error(`試驗 ${episode.seed} 嘅軌跡或時間有誤`);
    if (!vector(episode.goal_xyz, 3)) throw Error(`試驗 ${episode.seed} 缺少終點`);
    if (episode.orientation_recorded && (!Array.isArray(episode.trajectory_quat_xyzw) || episode.trajectory_quat_xyzw.length !== points.length || !episode.trajectory_quat_xyzw.every(q => vector(q, 4) && Math.hypot(...q) > .5))) throw Error(`試驗 ${episode.seed} 嘅姿態記錄有誤`);
  }
  return data;
}

export function sampleReplay(episode, time) {
  const times = episode.times_s;
  const clamped = Math.max(0, Math.min(time, times[times.length - 1]));
  let lo = 0, hi = times.length - 1;
  while (hi - lo > 1) { const middle = (lo + hi) >> 1; if (times[middle] <= clamped) lo = middle; else hi = middle; }
  const alpha = (clamped - times[lo]) / (times[hi] - times[lo]);
  const position = new THREE.Vector3(...episode.trajectory_xyz[lo]).lerp(new THREE.Vector3(...episode.trajectory_xyz[hi]), alpha);
  const quaternion = new THREE.Quaternion();
  if (episode.orientation_recorded) quaternion.fromArray(episode.trajectory_quat_xyzw[lo]).normalize().slerp(new THREE.Quaternion().fromArray(episode.trajectory_quat_xyzw[hi]).normalize(), alpha);
  return {position, quaternion, time: clamped};
}

export async function addHallReplay({viewer, drone, url = '/data/hall-replay.json'}) {
  if (!drone) throw Error('Crazyflie 模型未準備好，未能開啟回放');
  document.body.classList.add('replay-mode');
  const manual = document.querySelector('.cf-controls');
  if (manual) { manual.hidden = true; manual.querySelectorAll('button, input').forEach(input => {input.disabled = true;}); }
  const style = document.createElement('style');
  style.textContent = `
    .replay-mode .panel{padding:14px 18px;display:block}.replay-mode .panel>.stat,.replay-mode .panel>.divider,.replay-mode .panel>.controls,.replay-mode .panel>.help,.replay-mode .note{display:none}
    .hr{font-size:11px;color:#dde8d5}.hr-top,.hr-tools,.hr-timeline{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.hr-top{justify-content:space-between;margin-bottom:10px}.hr-heading{font-size:13px;letter-spacing:.02em}.hr-heading em{font-style:normal;font-size:10px;color:#b6c3ae;margin-left:10px}.hr-summary{font-variant-numeric:tabular-nums;color:#d5f58a;font-size:11px}.hr small{font-size:10px;color:#adbca4;line-height:1.6}
    .hr button,.hr select,.hr summary,.hr a{font:inherit;border:1px solid #d7e4c344;color:#e8eedf;background:#142219;padding:8px 11px;border-radius:2px}.hr button{cursor:pointer}.hr button[aria-pressed=true],.hr #hr-play{background:#d5f58a;color:#172018;border-color:#d5f58a}.hr button:disabled{opacity:.4;cursor:wait}.hr select{max-width:100%;min-width:0;flex:1}.hr select:focus-visible,.hr input:focus-visible,.hr summary:focus-visible,.hr a:focus-visible{outline:2px solid #d5f58a;outline-offset:2px}.hr a{color:#bccaad;text-decoration:none}.hr summary{cursor:pointer;list-style:none}.hr details[open] summary{border-color:#d5f58a}
    .hr-timeline{margin-top:10px;flex-wrap:nowrap}.hr-timeline input{flex:1;min-width:40px;accent-color:#d5f58a}.hr-clock{font-variant-numeric:tabular-nums;min-width:86px;text-align:right;font-family:Menlo,monospace;font-size:10px}.hr-clearance{font-variant-numeric:tabular-nums;margin-left:auto;color:#d5f58a}.hr-result{color:#d5f58a}.hr-result.failed{color:#ffc0a6}.hr-details{position:absolute;right:0;bottom:calc(100% + 8px);width:400px;max-width:100%;padding:16px;background:#101c15f5;border:1px solid #d7e4c344;max-height:48dvh;overflow-y:auto;box-shadow:0 8px 28px #0004}.hr-details p{font-size:11px;line-height:1.7;margin:10px 0}.hr-details label{display:flex;align-items:center;gap:8px;font-size:11px}.hr-details select{flex:0;min-width:80px}.hr-run{overflow-wrap:anywhere;color:#a9b99e;font-family:Menlo,monospace;font-size:10px!important}.hr-error{color:#ffc0a6;line-height:1.7}.hr-loading{padding:5px 0}.hr-view-note{display:block;margin-top:6px}
    @media(max-width:760px){.replay-mode header{top:14px;left:16px;max-width:calc(100% - 32px)}.replay-mode h1{font-size:20px;margin:7px 0 4px}.replay-mode header>p{font-size:10px}.replay-mode .eyebrow{font-size:9px}.replay-mode .scene-picker{margin-top:9px;gap:6px}.replay-mode .scene-picker select{padding:6px;max-width:190px}.replay-mode .scene-picker a{display:none}.replay-mode #status{font-size:9px;padding:6px 9px;line-height:1.5}.replay-mode .panel{padding:11px;left:10px;right:10px;bottom:10px}.hr-top{gap:5px;margin-bottom:8px}.hr-heading{font-size:12px}.hr-heading em{font-size:9px;margin-left:5px}.hr-summary{font-size:10px;width:100%}.hr button,.hr select,.hr summary,.hr a{padding:7px 8px;font-size:10px}.hr-tools{gap:6px}.hr-tools select{flex-basis:calc(100% - 65px)}.hr-clearance{font-size:10px}.hr-timeline{gap:6px;margin-top:8px}.hr-view-note{font-size:9px!important;line-height:1.5!important}.hr-details{padding:12px}.hr #hr-restart{font-size:0;padding:7px 8px}.hr #hr-restart:after{content:'↺';font-size:11px}}
    @media(max-width:760px) and (max-height:600px){.replay-mode header{top:10px}.replay-mode header h1,.replay-mode header>p,.replay-mode header>.eyebrow{display:none}.replay-mode .scene-picker{margin-top:0}.replay-mode .scene-picker select{max-width:180px}.replay-mode #status{margin-top:5px}.replay-mode .panel{bottom:8px}.hr-top{margin-bottom:6px}.hr-view-note{max-width:250px}}
  `;
  document.head.append(style);
  const panel = document.createElement('section');
  panel.className = 'hr';
  panel.setAttribute('aria-label', '工廠避障記錄回放');
  const manualUrl = new URL(location.href); manualUrl.searchParams.delete('replay');
  panel.innerHTML = `<div class="hr-loading" role="status">載入工廠驗收軌跡…</div>`;
  document.querySelector('.panel').append(panel);
  drone.visible = false;
  let data;
  try {
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok) throw Error(response.status === 503 || response.status === 404 ? '訓練或驗收未完成，暫時未有回放資料。' : `讀取回放失敗 (${response.status})`);
    data = validateReplay(await response.json());
  } catch (error) {
    panel.replaceChildren();
    const message = document.createElement('p'); message.className = 'hr-error'; message.setAttribute('role', 'alert'); message.textContent = error.message;
    const reload = document.createElement('button'); reload.textContent = '重新載入'; reload.onclick = () => location.reload();
    const back = document.createElement('a'); back.href = manualUrl.href; back.textContent = '手動擺位';
    panel.append(message, reload, document.createTextNode(' '), back);
    return;
  }
  panel.innerHTML = `<div class="hr-top"><div class="hr-heading">工廠避障回放 <em>已錄製 · 非即時訓練</em></div><span id="hr-summary" class="hr-summary"></span></div>
    <div class="hr-tools"><select id="hr-trial" aria-label="選擇驗收試驗"></select><button id="hr-play" type="button">▶ 播放</button><button id="hr-restart" type="button" aria-label="由頭開始回放">↺ 重播</button><button id="hr-follow" type="button" aria-pressed="true">第三身跟機</button><button id="hr-overview" type="button" aria-pressed="false">路線全景</button><details><summary>回放資料</summary><div class="hr-details"><p id="hr-result" class="hr-result"></p><p id="hr-metric"></p><label for="hr-speed">播放速度 <select id="hr-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label><p><label><input id="hr-proxy" type="checkbox" checked> 顯示障礙代理、路線及標記</label></p><p>橙色圓柱係手動建立嘅碰撞代理；只驗證繞過呢個代理，未有完整工廠碰撞地圖。數值係機身包絡至代理嘅水平淨距。</p><p id="hr-orientation"></p><p id="hr-run" class="hr-run"></p><a id="hr-manual">返回手動擺位</a></div></details><span id="hr-clearance" class="hr-clearance"></span></div>
    <div class="hr-timeline"><input id="hr-seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="回放時間"><output id="hr-clock" class="hr-clock">0.00 / 0.00 s</output></div><small id="hr-state" class="hr-view-note" role="status">已暫停 · 先撳播放。模型使用原始尺寸 1×。</small><small class="hr-view-note">只驗證橙色代理；未包含全廠碰撞。</small>`;
  const el = id => panel.querySelector('#hr-' + id);
  el('manual').href = manualUrl.href;
  const total = data.episodes.length, successes = data.episodes.filter(ep => ep.success).length, collisions = data.episodes.filter(ep => ep.collision).length, intersections = data.episodes.filter(ep => !ep.proxy_clear).length;
  el('summary').textContent = `${successes}/${total} 到達 · ${collisions} 次記錄碰撞 · ${intersections} 條路徑穿越代理`;
  el('run').textContent = `Run ${data.run_id}\n驗收匯出 ${data.created_at}`;
  const resultName = episode => episode.collision ? '碰撞' : episode.success ? '到達' : '未到達';
  el('trial').replaceChildren(...data.episodes.map((episode, index) => new Option(`${String(index + 1).padStart(2, '0')} · seed ${episode.seed} · ${resultName(episode)}`, String(index))));

  const graphics = new THREE.Group(); graphics.name = 'Recorded hall trajectory and approximate collision proxy'; viewer.threeScene.add(graphics);
  const proxyGeometry = new THREE.CylinderGeometry(data.proxy.radius_m, data.proxy.radius_m, data.proxy.height_m, 48, 1, true);
  proxyGeometry.rotateX(Math.PI / 2);
  const proxy = new THREE.Group(); proxy.position.fromArray(data.proxy.center_xyz);
  proxy.add(new THREE.Mesh(proxyGeometry, new THREE.MeshBasicMaterial({color: 0xf4a359, transparent: true, opacity: .075, side: THREE.DoubleSide, depthWrite: false})));
  const edges = new THREE.EdgesGeometry(proxyGeometry, 12);
  proxy.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({color: 0xffbb72, transparent: true, opacity: .9})));
  graphics.add(proxy);
  const routeMaterial = new THREE.LineBasicMaterial({color: 0xd5f58a});
  const route = new THREE.Line(new THREE.BufferGeometry(), routeMaterial); graphics.add(route);
  function ring(radius, color) {
    const points = Array.from({length: 65}, (_, i) => new THREE.Vector3(Math.cos(i / 64 * Math.PI * 2) * radius, Math.sin(i / 64 * Math.PI * 2) * radius, 0));
    return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({color, transparent: true, opacity: .85}));
  }
  const start = ring(.10, 0x8acdf5), goal = ring(.15, 0xd5f58a), envelope = ring(data.vehicle.envelope_radius_m, 0xe8eddc);
  graphics.add(start, goal, envelope);
  const endpoint = new THREE.Mesh(new THREE.SphereGeometry(.035, 12, 8), new THREE.MeshBasicMaterial({color: 0xd5f58a})); graphics.add(endpoint);
  drone.scale.setScalar(1); drone.visible = true;
  viewer.camera.near = .001; viewer.camera.updateProjectionMatrix();
  let episode, time = 0, duration = 0, playing = false, speed = 1, following = true, frameId = 0, previousFrame = null;
  const followOffset = new THREE.Vector3(-.35, -.42, .25);
  const render = () => viewer.forceRenderNextFrame();
  let viewport = {width: 1, height: 1, availableHeight: 1};
  function frameVisibleScene() {
    const canvas = document.getElementById('scene').getBoundingClientRect();
    if (!canvas.width || !canvas.height) return;
    const header = document.querySelector('header').getBoundingClientRect();
    const footer = panel.closest('.panel').getBoundingClientRect();
    const top = Math.max(canvas.top + 8, header.bottom + 8);
    const bottom = Math.min(canvas.bottom - 8, footer.top - 8);
    const centerY = Math.max(canvas.top, Math.min(canvas.bottom, (top + bottom) / 2)) - canvas.top;
    viewport = {width: canvas.width, height: canvas.height, availableHeight: Math.max(32, bottom - top)};
    // Shift the real camera projection into the unobstructed space between HUDs.
    // A positive vertical offset moves projected objects upwards, without
    // changing the drone's scale, camera distance, or splat/mesh registration.
    viewer.camera.setViewOffset(canvas.width, canvas.height, 0, canvas.height / 2 - centerY, canvas.width, canvas.height);
    render();
  }
  function cameraAt(position, target) {viewer.camera.position.copy(position); viewer.camera.up.set(0, 0, 1); viewer.controls.target.copy(target); viewer.camera.lookAt(target); viewer.controls.update();}
  function follow() {
    // Keep the original 10.6 cm model legible on narrow browser panels.
    const aspectScale = Math.max(1, .72 / Math.max(.25, viewer.camera.aspect));
    cameraAt(drone.position.clone().addScaledVector(followOffset, aspectScale), drone.position);
  }
  function overview() {
    const bounds = new THREE.Box3().setFromPoints(episode.trajectory_xyz.map(p => new THREE.Vector3(...p)));
    bounds.expandByPoint(new THREE.Vector3(...episode.goal_xyz));
    bounds.expandByPoint(new THREE.Vector3(data.proxy.center_xyz[0] - data.proxy.radius_m, data.proxy.center_xyz[1] - data.proxy.radius_m, 0));
    bounds.expandByPoint(new THREE.Vector3(data.proxy.center_xyz[0] + data.proxy.radius_m, data.proxy.center_xyz[1] + data.proxy.radius_m, data.proxy.height_m));
    bounds.expandByScalar(.16);
    const center = bounds.getCenter(new THREE.Vector3());
    const direction = new THREE.Vector3(-.12, -.5, 1).normalize();
    const right = new THREE.Vector3(0, 0, 1).cross(direction).normalize();
    const up = direction.clone().cross(right);
    const tangent = Math.tan(THREE.MathUtils.degToRad(viewer.camera.fov) / 2);
    const horizontal = tangent * viewer.camera.aspect * .90;
    const vertical = tangent * Math.min(1, viewport.availableHeight / viewport.height) * .90;
    let distance = 1;
    for (const x of [bounds.min.x, bounds.max.x]) for (const y of [bounds.min.y, bounds.max.y]) for (const z of [bounds.min.z, bounds.max.z]) {
      const corner = new THREE.Vector3(x, y, z).sub(center), depth = corner.dot(direction);
      distance = Math.max(distance, depth + Math.abs(corner.dot(right)) / horizontal, depth + Math.abs(corner.dot(up)) / vertical);
    }
    cameraAt(center.clone().addScaledVector(direction, distance), center);
    render();
  }
  function pose() {
    const sample = sampleReplay(episode, time); time = sample.time;
    drone.position.copy(sample.position); drone.quaternion.copy(sample.quaternion); envelope.position.copy(sample.position);
    if (following) follow();
    const clearance = Math.hypot(sample.position.x - data.proxy.center_xyz[0], sample.position.y - data.proxy.center_xyz[1]) - data.proxy.radius_m - data.vehicle.envelope_radius_m;
    el('clearance').textContent = `代理淨距 ${clearance.toFixed(3)} m`;
    el('clearance').style.color = clearance < 0 ? '#ffc0a6' : '';
    el('clock').value = `${time.toFixed(2)} / ${duration.toFixed(2)} s`;
    el('seek').value = String(time);
    render();
  }
  function pause(message = '已暫停 · 模型原始尺寸 1×。') {
    playing = false; previousFrame = null; cancelAnimationFrame(frameId); frameId = 0;
    el('play').textContent = '▶ 播放'; el('state').textContent = message;
  }
  function tick(timestamp) {
    frameId = 0;
    if (!playing) return;
    if (previousFrame !== null) time = Math.min(duration, time + (timestamp - previousFrame) / 1000 * speed);
    previousFrame = timestamp;
    pose();
    if (time >= duration) {pause(`回放完結 · ${resultName(episode)}。可重播或揀另一試驗。`); return;}
    frameId = requestAnimationFrame(tick);
  }
  function play() {
    if (time >= duration) time = 0;
    playing = true; previousFrame = null; el('play').textContent = 'Ⅱ 暫停'; el('state').textContent = `播放記錄 ${speed}× · 原始模型 1× · 未執行新推論。`;
    frameId = requestAnimationFrame(tick);
  }
  function select(index) {
    pause(); episode = data.episodes[index]; time = 0; duration = episode.times_s[episode.times_s.length - 1];
    el('seek').max = String(duration);
    route.geometry.dispose(); route.geometry = new THREE.BufferGeometry().setFromPoints(episode.trajectory_xyz.map(p => new THREE.Vector3(...p)));
    routeMaterial.color.set(episode.success && episode.proxy_clear ? 0xd5f58a : 0xffb899);
    start.position.fromArray(episode.trajectory_xyz[0]); goal.position.fromArray(episode.goal_xyz); endpoint.position.fromArray(episode.trajectory_xyz[episode.trajectory_xyz.length - 1]);
    endpoint.material.color.set(episode.success ? 0xd5f58a : 0xff8d70);
    el('result').textContent = `seed ${episode.seed} · ${resultName(episode)}${episode.failure_reason ? ` · ${episode.failure_reason}` : ''}`;
    el('result').classList.toggle('failed', !episode.success || episode.collision);
    const minClearance = episode.metrics?.swept_proxy_clearance_m;
    el('metric').textContent = `整段記錄：${episode.steps} 步 / ${duration.toFixed(2)} s；連續線段最小代理淨距 ${Number.isFinite(minClearance) ? minClearance.toFixed(3) + ' m' : '未提供'}。${episode.proxy_clear ? '路徑未穿越代理。' : '路徑穿越代理。'}`;
    el('orientation').textContent = episode.orientation_recorded ? '位置與姿態來自 MuJoCo 記錄；影格之間作線性／四元數插值。藍圈起點，綠圈目標，細白圈係機身水平包絡。' : '位置來自 MuJoCo 記錄，影格之間作線性插值；原始記錄冇姿態，機身固定朝向只供顯示。藍圈起點，綠圈目標，細白圈係機身水平包絡。';
    pose(); if (!following) overview();
  }
  el('trial').onchange = () => select(Number(el('trial').value));
  el('play').onclick = () => playing ? pause() : play();
  el('restart').onclick = () => {pause(); time = 0; pose(); play();};
  el('seek').oninput = () => {pause('拖動時間軸 · 已暫停。'); time = Number(el('seek').value); pose();};
  el('speed').onchange = () => {speed = Number(el('speed').value); previousFrame = null; if (playing) el('state').textContent = `播放記錄 ${speed}× · 原始模型 1× · 未執行新推論。`;};
  function setFollowing(value) {following = value; el('follow').setAttribute('aria-pressed', String(value)); el('overview').setAttribute('aria-pressed', String(!value)); if(value) {follow(); render();} else overview();}
  el('follow').onclick = () => setFollowing(true);
  el('overview').onclick = () => setFollowing(false);
  el('proxy').onchange = () => {graphics.visible = el('proxy').checked; render();};
  document.addEventListener('visibilitychange', () => {if (document.hidden && playing) pause('頁面隱藏，回放已自動暫停。');});
  viewer.controls.addEventListener('start', () => {following = false; el('follow').setAttribute('aria-pressed', 'false'); el('overview').setAttribute('aria-pressed', 'false');});
  function resizeFrame() {
    frameVisibleScene();
    if (!episode) return;
    if (following) {follow(); render();}
    else if (el('overview').getAttribute('aria-pressed') === 'true') overview();
  }
  const layoutObserver = new ResizeObserver(resizeFrame);
  for (const element of [document.getElementById('scene'), document.querySelector('header'), panel.closest('.panel')]) layoutObserver.observe(element);
  window.addEventListener('resize', resizeFrame);
  frameVisibleScene();
  select(0);
  return {pause};
}
