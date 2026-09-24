import * as THREE from 'three';

// The same seven visual meshes used by cf2.xml. This is a scene-placement
// tool, not a dynamics simulator or a source of metric campus measurements.
export async function addCrazyflie({viewer, sceneId, cameraUp, focusOnLoad=false}) {
  if (!['hall', 'dajing'].includes(sceneId)) return;
  const response = await fetch(new URL('../assets/crazyflie.json', import.meta.url));
  if (!response.ok) throw new Error(`Crazyflie 模型載入失敗 (${response.status})`);
  const data = await response.json();
  const drone = new THREE.Group();
  drone.name = 'Crazyflie 2 / Menagerie visual meshes';
  for (const part of data.parts) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(part.positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(part.normals, 3));
    geometry.setIndex(part.indices);
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(...part.rgba.slice(0, 3)),
      roughness: .48, metalness: /gold|chrome/.test(part.material) ? .65 : .05,
      depthTest: true, depthWrite: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = part.name;
    drone.add(mesh);
  }
  const campus = sceneId === 'dajing';
  const up = new THREE.Vector3(...cameraUp).normalize();
  const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), up);
  const start = new THREE.Vector3(...(campus ? [8, -4.5, 0] : [13.55, -3.15, 1.2]));
  const nativeWidth = Math.max(data.bounds.size[0], data.bounds.size[1]);
  const initialScale = campus ? .35 / nativeWidth : 1;
  let scale = initialScale, heading = 0;
  drone.position.copy(start);
  drone.quaternion.copy(orientation);
  drone.scale.setScalar(scale);
  viewer.threeScene.add(drone);
  viewer.threeScene.add(new THREE.AmbientLight(0xffffff, .9));
  const light = new THREE.DirectionalLight(0xffffff, 1.4);
  light.position.copy(start).add(new THREE.Vector3(-3, -4, 7).applyQuaternion(orientation));
  light.target = drone;
  viewer.threeScene.add(light);

  const style = document.createElement('style');
  style.textContent = `
    .cf-controls{width:100%;display:flex;align-items:center;gap:7px;flex-wrap:wrap;font-size:11px;border-top:1px solid #d7e4c326;padding-top:9px;color:#dce8cb}
    .cf-controls b{font-size:12px;font-weight:500;margin-right:4px}.cf-controls button,.cf-controls summary{font:inherit;padding:7px 10px;border:1px solid #d7e4c355;background:#142219;color:#e8ebdb;cursor:pointer;list-style:none}
    .cf-controls summary:focus-visible,.cf-controls input:focus-visible{outline:2px solid #d7f58a;outline-offset:2px}
    .cf-controls small{font-size:10px;color:#b2bda9}.cf-controls details[open] summary{background:#d7f58a;color:#172018}
    .cf-editor{position:absolute;bottom:calc(100% + 8px);right:0;width:290px;max-width:100%;max-height:min(400px,55dvh);overflow-y:auto;padding:14px;background:#101c15f5;border:1px solid #d7e4c355;box-shadow:0 8px 28px #0005}
    .cf-editor p{margin:0 0 10px;font-size:11px;line-height:1.6}.cf-fields{display:flex;gap:7px;margin:10px 0}.cf-fields label{flex:1;min-width:0}.cf-fields input{display:block;width:100%;font:inherit;padding:7px;margin-top:5px;color:#e8ebdb;background:#0b120e;border:1px solid #d7e4c355}
    .cf-pad{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}.cf-pose{font-variant-numeric:tabular-nums;font-family:monospace;margin-top:10px}.cf-editor label[for=cf-scale]{display:block;margin:10px 0}.cf-editor input[type=range]{width:100%}
    @media(max-width:700px){.cf-controls{gap:5px;padding-top:7px}.cf-controls b{font-size:11px}.cf-controls button,.cf-controls summary{padding:7px;font-size:10px}.cf-controls>small{width:100%;font-size:9px}}
  `;
  document.head.append(style);
  const panel = document.createElement('div');
  panel.className = 'cf-controls';
  panel.setAttribute('aria-label', 'Crazyflie 場景模型');
  panel.innerHTML = `<b>● Crazyflie 2</b><button id="cf-focus" type="button">聚焦架機</button><button id="cf-visible" type="button" aria-pressed="true">隱藏</button>
    <details><summary>調整位置</summary><div class="cf-editor">
      <p>手動擺位 · 未啟動飛行控制<br>${campus ? '校園比例與碰撞未校準；尺寸只供展示。' : '原始模型 1×；工廠碰撞地圖未完整。'}</p>
      <div class="cf-pad"><button data-move="0,-1" aria-label="向左移">← 左</button><button data-move="0,1" aria-label="向右移">右 →</button><button data-move="1,1">前 ↑</button><button data-move="1,-1">後 ↓</button><button data-move="2,1">升高</button><button data-move="2,-1">降低</button></div>
      <small>每步 ${campus ? '0.25 場景單位' : '0.05 m'} · 相機同步移動</small>
      <form id="cf-position"><div class="cf-fields">${['x','y','z'].map(a=>`<label>${a.toUpperCase()}<input name="${a}" type="number" step="any" required aria-label="Crazyflie ${a.toUpperCase()} 座標"></label>`).join('')}</div><button type="submit">套用座標</button></form>
      <label for="cf-scale">${campus ? '顯示寬度（場景單位）' : '模型顯示倍率'} <output id="cf-scale-value"></output></label>
      <input id="cf-scale" type="range" min="${campus ? '.1' : '1'}" max="${campus ? '1' : '5'}" step="${campus ? '.05' : '.5'}" value="${campus ? '.35' : '1'}">
      <div class="cf-pad"><button id="cf-turn" type="button">轉向 45°</button><button id="cf-reset" type="button">重設擺位</button></div>
      <div id="cf-pose" class="cf-pose"></div>
    </div></details><small id="cf-caption"></small>`;
  const container = document.querySelector('.bottom, .panel');
  container.style.flexWrap = 'wrap';
  container.append(panel);
  const note = document.querySelector('.note');
  if(note) new ResizeObserver(() => {note.style.bottom = `${innerHeight-container.getBoundingClientRect().top+12}px`;}).observe(container);
  const el = id => panel.querySelector('#' + id);
  const form = el('cf-position');
  const render = () => viewer.forceRenderNextFrame();
  function update() {
    ['x','y','z'].forEach(a => { form.elements[a].value = drone.position[a].toFixed(3); });
    el('cf-pose').textContent = `xyz ${drone.position.toArray().map(n=>n.toFixed(3)).join(', ')}`;
    el('cf-scale-value').value = campus ? (nativeWidth*scale).toFixed(2) : `${scale.toFixed(1)}×`;
    el('cf-caption').textContent = campus ? `校園擺位 · 顯示寬 ${ (nativeWidth*scale).toFixed(2)} 場景單位 · 未校準` : `工廠擺位 · 模型 ${scale.toFixed(1)}× · ${scale === 1 ? '原始尺寸' : '已放大顯示'}`;
    render();
  }
  function show() { drone.visible = true; el('cf-visible').textContent = '隱藏'; el('cf-visible').setAttribute('aria-pressed', 'true'); }
  function focus() {
    show();
    const width = nativeWidth * scale;
    // Maintain a visible model even in a narrow phone/browser panel.
    const aspect = Math.max(.25, viewer.camera.aspect);
    const distance = width * Math.max(3.2, 1.5/aspect);
    const offset = new THREE.Vector3(-1, -1.2, .9).normalize().multiplyScalar(distance).applyQuaternion(orientation);
    viewer.camera.near = Math.min(.001, width/100);
    viewer.camera.updateProjectionMatrix();
    viewer.camera.up.copy(up);
    viewer.camera.position.copy(drone.position).add(offset);
    viewer.controls.target.copy(drone.position);
    viewer.camera.lookAt(drone.position);
    viewer.controls.update();
    render();
  }
  function move(position) {
    const delta = position.clone().sub(drone.position);
    drone.position.copy(position);
    viewer.camera.position.add(delta); viewer.controls.target.add(delta); light.position.add(delta);
    viewer.controls.update(); update();
  }
  el('cf-focus').onclick = focus;
  el('cf-visible').onclick = () => { drone.visible = !drone.visible; el('cf-visible').textContent = drone.visible ? '隱藏' : '顯示'; el('cf-visible').setAttribute('aria-pressed', String(drone.visible)); render(); };
  panel.querySelectorAll('[data-move]').forEach(button => { button.onclick = () => {
    const [axis, sign] = button.dataset.move.split(',').map(Number);
    const delta = new THREE.Vector3().setComponent(axis, sign*(campus ? .25 : .05)).applyQuaternion(orientation);
    move(drone.position.clone().add(delta));
  }; });
  form.onsubmit = event => { event.preventDefault(); const values=['x','y','z'].map(a=>form.elements[a].valueAsNumber); if(values.every(n=>Number.isFinite(n)&&Math.abs(n)<=1000)) move(new THREE.Vector3(...values)); };
  el('cf-scale').oninput = () => { scale = Number(el('cf-scale').value)/(campus ? nativeWidth : 1); drone.scale.setScalar(scale); update(); };
  el('cf-turn').onclick = () => { heading += Math.PI/4; drone.quaternion.copy(orientation).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),heading)); render(); };
  el('cf-reset').onclick = () => { scale=initialScale;heading=0;drone.scale.setScalar(scale);drone.quaternion.copy(orientation);el('cf-scale').value=campus?'.35':'1';move(start);focus(); };
  update();
  if(focusOnLoad) focus();
  return drone;
}
