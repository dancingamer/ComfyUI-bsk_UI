/*
 * bsk_prompt_autocomplete_lib.js
 * --------------------------------------------------------------------------
 * 用户可编辑的「提示词自动补全配置」通用库。
 *
 * 设计目标：
 *  1. 让用户手动维护自己的补全词库（JSON 文件），与内置 BSK_KW_LIB 解耦。
 *  2. 配置可多选同时生效；可一键禁用内置补全，仅用用户配置。
 *  3. 完全可复用：视频工作台（vs_prompt_editor.js）与面板（panel.js）共用同一套
 *     管理窗口与候选数据，通过 window.BSK_AUTO_LIB 接口暴露。
 *
 * 数据格式（JSON，每条配置一个文件内容）：
 *  {
 *    "name": "我的风格词",            // 配置名（可选，缺省取文件名）
 *    "entries": [
 *      { "keyword": "cinematic", "title": "电影感", "desc": "宽幅电影质感", "modes": "both" },
 *      { "keyword": "(word:1.3)", "title": "强权重", "desc": "提高该词权重", "modes": "concat" }
 *    ]
 *  }
 *  - modes: "both" | "ref" | "concat"（ref=多参考 Ref2VA，concat=首尾帧拼接 FL2VA）
 *  - keyword 为补全插入文本；title/desc 仅作中文说明展示。
 *
 * 存储：localStorage（命名空间 bsk_ac_lib）。每个配置一个独立条目，便于增删。
 */

(function () {
  'use strict';

  var NS = 'bsk_ac_lib';
  var KEY_CONFIGS = NS + ':configs';        // { [id]: {id,name,entries} }
  var KEY_ENABLED = NS + ':enabled';         // { [id]: true }
  var KEY_DISABLE_BUILTIN = NS + ':disable_builtin'; // "1" | "0"
  var KEY_REMOVED_BUILTIN = NS + ':removed_builtin'; // "1" 表示用户删除过内置示例，不再自动注入

  // 内置示例补全配置（首次无配置时自动注入；可被用户删除，删后不再自动出现）
  var BUILTIN_SAMPLE_ID = 'bsk_builtin_sample';
  var SAMPLE = {
    id: BUILTIN_SAMPLE_ID,
    name: '内置示例',
    builtin: true,
    entries: [
      { keyword: 'cinematic', title: '电影感', desc: '宽幅电影质感、浅景深', modes: 'both' },
      { keyword: '(word:1.3)', title: '强权重', desc: '提高该词权重', modes: 'both' },
      { keyword: 'close-up', title: '特写', desc: '人物/物体特写镜头', modes: 'both' },
      { keyword: 'slow motion', title: '慢动作', desc: '放慢动作节奏', modes: 'concat' },
      { keyword: 'subject_definitions', title: '主体定义', desc: 'Ref2VA 模式用于定义主体', modes: 'ref' },
      { keyword: 'reference_label', title: '参考标签', desc: 'Ref2VA 模式绑定参考素材', modes: 'ref' }
    ]
  };

  // ===== 底层存储 =====
  function safeParse(s, def) {
    try { return s ? JSON.parse(s) : def; } catch (e) { return def; }
  }
  function loadConfigs() {
    return safeParse(localStorage.getItem(KEY_CONFIGS), {}) || {};
  }
  function saveConfigs(map) {
    localStorage.setItem(KEY_CONFIGS, JSON.stringify(map));
  }
  function loadEnabled() {
    return safeParse(localStorage.getItem(KEY_ENABLED), {}) || {};
  }
  function saveEnabled(map) {
    localStorage.setItem(KEY_ENABLED, JSON.stringify(map));
  }

  function genId() {
    return 'cfg_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7);
  }

  // ===== 对外数据接口 =====
  var LIB = {};

  // 是否禁用内置补全（BSK_KW_LIB）
  LIB.isBuiltinDisabled = function () {
    return localStorage.getItem(KEY_DISABLE_BUILTIN) === '1';
  };
  LIB.setBuiltinDisabled = function (v) {
    localStorage.setItem(KEY_DISABLE_BUILTIN, v ? '1' : '0');
    LIB._notify();
  };

  // 全部配置（内置示例排在最前，其余按名称排序）
  LIB.getAll = function () {
    var map = loadConfigs();
    var list = Object.keys(map).map(function (id) { return map[id]; });
    // 用户未删除过内置示例时，自动注入（再保险：防止存储被外部清空）
    if (localStorage.getItem(KEY_REMOVED_BUILTIN) !== '1') {
      var exists = list.some(function (c) { return c.id === BUILTIN_SAMPLE_ID; });
      if (!exists) {
        LIB.save(SAMPLE, BUILTIN_SAMPLE_ID); // 持久化，使其参与启用/补全
        list.push(SAMPLE);
      }
    }
    return list.sort(function (a, b) {
      if (a.builtin && !b.builtin) return -1;
      if (!a.builtin && b.builtin) return 1;
      return (a.name || '').localeCompare(b.name || '', 'zh');
    });
  };

  LIB.getEnabled = function () {
    return loadEnabled();
  };

  // 保存/新建一个配置；id 缺省则新建
  LIB.save = function (config, id) {
    var map = loadConfigs();
    var enabled = loadEnabled();
    var cfgId = id || config.id || genId();
    var entry = {
      id: cfgId,
      name: (config.name || '未命名配置').trim(),
      entries: Array.isArray(config.entries) ? config.entries : []
    };
    map[cfgId] = entry;
    if (!(cfgId in enabled)) enabled[cfgId] = true; // 新配置默认启用
    saveConfigs(map);
    saveEnabled(enabled);
    LIB._notify();
    return entry;
  };

  LIB.remove = function (id) {
    var map = loadConfigs();
    var enabled = loadEnabled();
    if (id === BUILTIN_SAMPLE_ID) {
      // 删除内置示例后不再自动注入
      localStorage.setItem(KEY_REMOVED_BUILTIN, '1');
    }
    delete map[id];
    delete enabled[id];
    saveConfigs(map);
    saveEnabled(enabled);
    LIB._notify();
  };

  LIB.setEnabled = function (id, on) {
    var enabled = loadEnabled();
    enabled[id] = !!on;
    saveEnabled(enabled);
    LIB._notify();
  };

  // 合并所有已启用配置的候选（带 modes 过滤）；mode: "ref" | "concat"
  // 内置示例视为普通配置参与补全（除非已被用户删除）
  LIB.getUserCandidates = function (mode) {
    var map = loadConfigs();
    var enabled = loadEnabled();
    var out = [];
    var ids = Object.keys(map);
    if (localStorage.getItem(KEY_REMOVED_BUILTIN) !== '1' && !map[BUILTIN_SAMPLE_ID]) {
      map = JSON.parse(JSON.stringify(map));
      map[BUILTIN_SAMPLE_ID] = SAMPLE;
      enabled[BUILTIN_SAMPLE_ID] = true;
      ids = Object.keys(map);
    }
    ids.forEach(function (id) {
      if (!enabled[id]) return;
      var cfg = map[id];
      (cfg.entries || []).forEach(function (e) {
        if (!e || !e.keyword) return;
        var m = e.modes || 'both';
        if (m !== 'both' && m !== mode) return;
        out.push({
          keyword: e.keyword,
          title: e.title || '',
          desc: e.desc || '',
          modes: m,
          source: cfg.name || '用户配置'
        });
      });
    });
    return out;
  };

  // 变更订阅（编辑器刷新补全用）
  LIB._subs = [];
  LIB._notify = function () {
    LIB._subs.forEach(function (fn) { try { fn(); } catch (e) {} });
  };
  LIB.subscribe = function (fn) {
    LIB._subs.push(fn);
    return function () {
      LIB._subs = LIB._subs.filter(function (x) { return x !== fn; });
    };
  };

  // ===== 工具栏按钮钩子（由编辑器在初始化时注册一个打开函数）=====
  // 注意：opener 必须直接渲染（buildAndOpen），绝不能回调用 openManager，否则无限递归。
  var _openFn = null;
  LIB.registerOpener = function (fn) { _openFn = fn; };
  LIB.openManager = function (rootEl) {
    if (_openFn) return _openFn(rootEl);
    buildAndOpen(rootEl);
  };
  // 供 opener 直接渲染窗口用（绕过 openManager 的 opener 判断，避免递归）
  LIB._build = buildAndOpen;

  // ===== 导入 / 导出 =====
  LIB.exportFile = function (id) {
    var map = loadConfigs();
    var cfg = map[id];
    if (!cfg) return;
    var text = JSON.stringify(cfg, null, 2);
    downloadText((cfg.name || 'autocomplete') + '.json', text);
  };

  LIB.importFile = function (file, onDone) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var obj = JSON.parse(reader.result);
        if (!obj || !Array.isArray(obj.entries)) {
          alert('文件格式错误：缺少 entries 数组');
          if (onDone) onDone(false);
          return;
        }
        // 文件名（去扩展）作为默认配置名
        var fname = (file.name || 'imported').replace(/\.json$/i, '');
        if (!obj.name) obj.name = fname;
        LIB.save(obj);
        if (onDone) onDone(true);
      } catch (e) {
        alert('解析失败：' + e.message);
        if (onDone) onDone(false);
      }
    };
    reader.readAsText(file);
  };

  // ===== 通用设置窗口 =====
  var CSS_INJECTED = false;
  function injectCSS() {
    if (CSS_INJECTED) return;
    CSS_INJECTED = true;
    var css = [
      '.bsk-ac-mask{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:100000;display:flex;align-items:center;justify-content:center;}',
      '.bsk-ac-win{width:560px;max-height:86vh;background:#454545;color:#e6e6e6;border:1px solid #3a3d4a;border-radius:10px;display:flex;flex-direction:column;font-size:15px;box-shadow:0 12px 40px rgba(0,0,0,.5);}',
      '.bsk-ac-head{padding:14px 18px;border-bottom:1px solid #3a3d4a;display:flex;align-items:center;justify-content:space-between;font-weight:600;}',
      '.bsk-ac-body{padding:14px 18px;overflow:auto;}',
      '.bsk-ac-foot{padding:12px 18px;border-top:1px solid #3a3d4a;display:flex;gap:10px;flex-wrap:wrap;}',
      '.bsk-ac-row{display:flex;align-items:center;gap:10px;padding:8px 6px;border-radius:6px;}',
      '.bsk-ac-row:hover{background:#2a2d38;}',
      '.bsk-ac-row label{flex:1;cursor:pointer;display:flex;align-items:center;gap:10px;}',
      '.bsk-ac-name{font-size:15px;}',
      '.bsk-ac-meta{font-size:13px;color:#9aa0ab;margin-left:auto;}',
      '.bsk-ac-del{background:#3a2530;color:#ff9a9a;border:1px solid #5a2a3a;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:15px;min-width:64px;}',
      '.bsk-ac-del:hover{background:#4a2a3a;}',
      '.bsk-ac-del.arming{background:#6a1f2a;color:#fff;font-weight:600;}',
      '.bsk-ac-edit{background:#2f3d4a;color:#cfe3ff;border:1px solid #3a5066;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:15px;min-width:64px;}',
      '.bsk-ac-edit:hover{background:#3a4d5a;}',
      '.bsk-ac-badge{font-size:12px;color:#9aa0ab;background:#2a2d38;border:1px solid #3a3d4a;border-radius:10px;padding:1px 8px;margin-left:8px;}',
      '.bsk-ac-opt{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:15px;}',
      '.bsk-ac-btn{background:#3a3d4a;color:#e6e6e6;border:1px solid #4a4d5a;border-radius:6px;padding:7px 14px;cursor:pointer;font-size:15px;}',
      '.bsk-ac-btn:hover{background:#474b5a;}',
      '.bsk-ac-btn.primary{background:#3b6ea5;border-color:#4a82c0;}',
      '.bsk-ac-empty{color:#9aa0ab;font-size:14px;padding:10px 4px;}',
      '.bsk-ac-link{font-size:13px;color:#7fb0e0;cursor:pointer;text-decoration:underline;}',
      'input.bsk-ac-chk{width:18px;height:18px;cursor:pointer;}',
      // 单条配置编辑窗口（大窗口 + 单行布局）
      '.bsk-ac-edit-win{width:100vw;height:100vh;max-height:100vh;}',
      '.bsk-ac-topbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px;border-bottom:1px solid #3a3d4a;}',
      '.bsk-ac-name-input{flex:1;min-width:160px;font-size:15px;padding:7px 9px;background:#4d4d4d;color:#e6e6e6;border:1px solid #3a3d4a;border-radius:6px;box-sizing:border-box;}',
      '.bsk-ac-mode-btn{background:#2f3d4a;color:#cfe3ff;border:1px solid #3a5066;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:15px;white-space:nowrap;}',
      '.bsk-ac-mode-btn:hover{background:#3a4d5a;}',
      '.bsk-ac-topbar .bsk-ac-btn{padding:7px 14px;}',
      '.bsk-ac-format{font-size:13px;color:#9aa0ab;padding:8px 16px 0;}',
      '.bsk-ac-ta-wrap{flex:1;display:flex;padding:8px 16px 14px;min-height:0;}',
      '.bsk-ac-ta-wrap textarea{flex:1;width:100%;font-size:14px;line-height:1.6;padding:8px;background:#4d4d4d;color:#e6e6e6;border:1px solid #3a3d4a;border-radius:6px;font-family:Consolas,Menlo,monospace;box-sizing:border-box;resize:none;}',
      '.bsk-ac-foot.align-right{justify-content:flex-end;}'
    ].join('\n');
    var style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }

  function closeOnEsc(mask, onClose) {
    function h(e) {
      if (e.key === 'Escape') {
        document.removeEventListener('keydown', h);
        if (mask.parentNode) mask.parentNode.removeChild(mask);
        if (onClose) onClose();
      }
    }
    document.addEventListener('keydown', h);
    return h;
  }

  function buildAndOpen(rootEl) {
    injectCSS();
    var mask = document.createElement('div');
    mask.className = 'bsk-ac-mask';
    var win = document.createElement('div');
    win.className = 'bsk-ac-win';

    var armTimers = {}; // id -> {timer, btn}
    function clearArm(id) {
      if (armTimers[id]) {
        clearInterval(armTimers[id].timer);
        var b = armTimers[id].btn;
        b.classList.remove('arming');
        b.textContent = '删除';
        delete armTimers[id];
      }
    }

    function renderBody() {
      var body = win.querySelector('.bsk-ac-body');
      body.innerHTML = '';

      // 禁用内置补全选项
      var opt = document.createElement('div');
      opt.className = 'bsk-ac-opt';
      var chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.className = 'bsk-ac-chk';
      chk.checked = LIB.isBuiltinDisabled();
      chk.onchange = function () {
        LIB.setBuiltinDisabled(chk.checked);
        var tip = win.querySelector('.bsk-ac-tip');
        tip.textContent = chk.checked ? '已禁用内置补全，仅用户配置生效' : '内置补全已启用';
      };
      var lbl = document.createElement('label');
      lbl.style.cursor = 'pointer';
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode(' 禁用内置默认补全（只使用下方用户配置）'));
      opt.appendChild(lbl);
      body.appendChild(opt);

      var tip = document.createElement('div');
      tip.className = 'bsk-ac-tip';
      tip.style.cssText = 'font-size:13px;color:#9aa0ab;margin-bottom:8px;';
      tip.textContent = chk.checked ? '已禁用内置补全，仅用户配置生效' : '勾选可只让用户配置生效';
      body.appendChild(tip);

      // 配置清单（多选）
      var cfgs = LIB.getAll();
      if (!cfgs.length) {
        var empty = document.createElement('div');
        empty.className = 'bsk-ac-empty';
        empty.textContent = '暂无配置。点击下方「保存」「导入」添加你自己的补全词库。';
        body.appendChild(empty);
      } else {
        var enabled = LIB.getEnabled();
        cfgs.forEach(function (cfg) {
          clearArm(cfg.id);
          var row = document.createElement('div');
          row.className = 'bsk-ac-row';

          var lab = document.createElement('label');
          var c = document.createElement('input');
          c.type = 'checkbox';
          c.className = 'bsk-ac-chk';
          c.checked = !!enabled[cfg.id];
          c.onchange = function () { LIB.setEnabled(cfg.id, c.checked); };
          var nm = document.createElement('span');
          nm.className = 'bsk-ac-name';
          nm.textContent = cfg.name || '未命名';
          lab.appendChild(c);
          lab.appendChild(nm);
          if (cfg.builtin) {
            var badge = document.createElement('span');
            badge.className = 'bsk-ac-badge';
            badge.textContent = '内置';
            lab.appendChild(badge);
          }
          var meta = document.createElement('span');
          meta.className = 'bsk-ac-meta';
          meta.textContent = (cfg.entries ? cfg.entries.length : 0) + ' 项';
          lab.appendChild(meta);

          // 编辑按钮：打开单条配置编辑窗口
          var edit = document.createElement('button');
          edit.className = 'bsk-ac-edit';
          edit.textContent = '编辑';
          edit.onclick = function () { editConfig(cfg, renderBody); };

          var del = document.createElement('button');
          del.className = 'bsk-ac-del';
          del.textContent = '删除';
          del.onclick = function () {
            if (armTimers[cfg.id]) { // 第二次点击：确认删除
              clearInterval(armTimers[cfg.id].timer);
              delete armTimers[cfg.id];
              LIB.remove(cfg.id);
              renderBody();
            } else { // 第一次点击：5 秒倒计时
              var n = 5;
              del.classList.add('arming');
              del.textContent = n;
              var timer = setInterval(function () {
                n--;
                if (n <= 0) {
                  clearArm(cfg.id);
                } else {
                  del.textContent = n;
                }
              }, 1000);
              armTimers[cfg.id] = { timer: timer, btn: del };
            }
          };

          row.appendChild(lab);
          row.appendChild(edit);
          row.appendChild(del);
          body.appendChild(row);
        });
      }
    }

    // 头部
    var head = document.createElement('div');
    head.className = 'bsk-ac-head';
    head.innerHTML = '<span>自动补全配置</span>';
    var x = document.createElement('span');
    x.textContent = '✕';
    x.style.cssText = 'cursor:pointer;font-size:18px;color:#9aa0ab;';
    x.onclick = function () { close(); };
    head.appendChild(x);
    win.appendChild(head);

    // 主体
    var bodyEl = document.createElement('div');
    bodyEl.className = 'bsk-ac-body';
    win.appendChild(bodyEl);

    // 底部按钮
    var foot = document.createElement('div');
    foot.className = 'bsk-ac-foot';

    var btnNew = document.createElement('button');
    btnNew.className = 'bsk-ac-btn primary';
    btnNew.textContent = '新建配置';
    btnNew.onclick = function () { editConfig(null, renderBody); };

    var btnImport = document.createElement('button');
    btnImport.className = 'bsk-ac-btn';
    btnImport.textContent = '导入';
    btnImport.onclick = function () {
      var inp = document.createElement('input');
      inp.type = 'file';
      inp.accept = '.json,application/json';
      inp.onchange = function () {
        if (inp.files && inp.files[0]) {
          LIB.importFile(inp.files[0], function (ok) { if (ok) renderBody(); });
        }
      };
      inp.click();
    };

    var btnExportAll = document.createElement('button');
    btnExportAll.className = 'bsk-ac-btn';
    btnExportAll.textContent = '导出全部';
    btnExportAll.onclick = function () {
      var cfgs = LIB.getAll();
      if (!cfgs.length) { alert('暂无可导出的配置'); return; }
      downloadText('autocomplete_all.json', JSON.stringify(cfgs, null, 2));
    };

    foot.appendChild(btnNew);
    foot.appendChild(btnImport);
    foot.appendChild(btnExportAll);
    win.appendChild(foot);

    mask.appendChild(win);
    (rootEl || document.body).appendChild(mask);

    var escHandler = closeOnEsc(mask, close);
    function close() {
      Object.keys(armTimers).forEach(clearArm);
      document.removeEventListener('keydown', escHandler);
      if (mask.parentNode) mask.parentNode.removeChild(mask);
    }
    mask.onclick = function (e) { if (e.target === mask) close(); };

    renderBody();
  }

  // 单条配置编辑窗口（新建/改名 + 编辑 entries 文本）
  function editConfig(cfg, onSaved) {
    injectCSS();
    var isNew = !cfg;
    var cur = cfg ? JSON.parse(JSON.stringify(cfg)) : { name: '', entries: [] };

    // 三种模式的「一行示例」模板：点按钮往编辑框追加该行，便于用户照着改
    // keyword 用占位说明词，title/desc 为中文，modes 对应模式
    var MODE_SAMPLES = {
      both: {
        label: '通用',
        title: '通用模式：首尾帧拼接与多参考两种模式都能用的补全词（最常用，默认都生效）',
        line: 'your_keyword | 中文名 | 该词作用的中文说明 | both'
      },
      concat: {
        label: '首尾帧',
        title: '首尾帧拼接模式（FL2VA）：只在「首尾帧拼接」生成时生效的补全词，例如镜头运动、节奏、转场等',
        line: 'your_keyword | 中文名 | 该词作用的中文说明 | concat'
      },
      ref: {
        label: '多参考',
        title: '多参考模式（Ref2VA）：只在「多模态参考」生成时生效的补全词，例如主体定义、参考素材标签等',
        line: 'your_keyword | 中文名 | 该词作用的中文说明 | ref'
      }
    };

    var mask = document.createElement('div');
    mask.className = 'bsk-ac-mask';
    var win = document.createElement('div');
    win.className = 'bsk-ac-win bsk-ac-edit-win';

    win.innerHTML =
      '<div class="bsk-ac-head"><span>' + (isNew ? '新建自动补全配置' : '编辑配置') + '</span>' +
      '<span style="cursor:pointer;font-size:18px;color:#9aa0ab;" data-x>✕</span></div>' +
      // 顶部单行：配置名 + 三个模式插入按钮 + 保存 + 取消
      '<div class="bsk-ac-topbar">' +
      '  <input class="bsk-ac-name-input" placeholder="配置名（如：电影感风格词）" />' +
      '  <button type="button" class="bsk-ac-mode-btn" data-mode="both" title="' + MODE_SAMPLES.both.title + '">＋' + MODE_SAMPLES.both.label + '示例</button>' +
      '  <button type="button" class="bsk-ac-mode-btn" data-mode="concat" title="' + MODE_SAMPLES.concat.title + '">＋' + MODE_SAMPLES.concat.label + '示例</button>' +
      '  <button type="button" class="bsk-ac-mode-btn" data-mode="ref" title="' + MODE_SAMPLES.ref.title + '">＋' + MODE_SAMPLES.ref.label + '示例</button>' +
      '  <button class="bsk-ac-btn primary" data-save>保存</button>' +
      '  <button class="bsk-ac-btn" data-cancel>取消</button>' +
      '</div>' +
      '<div class="bsk-ac-format">词条格式：每行一条，用 <b>|</b> 分隔四个字段 —— 补全词 | 中文名 | 说明 | 模式。后三项可省略，模式缺省为 both。也可直接粘贴整段 JSON（含 entries 数组）。</div>' +
      '<div class="bsk-ac-ta-wrap">' +
      '  <textarea class="bsk-ac-entries" placeholder="在此编辑补全词……"></textarea>' +
      '</div>' +
      '<div class="bsk-ac-foot align-right">' +
      '  <span class="bsk-ac-err" style="color:#ff9a9a;font-size:13px;margin-right:auto;"></span>' +
      '</div>';

    var nameInput = win.querySelector('.bsk-ac-name-input');
    var ta = win.querySelector('.bsk-ac-entries');
    var err = win.querySelector('.bsk-ac-err');
    nameInput.value = cur.name || '';

    // 预填 entries 文本
    ta.value = cur.entries.map(function (e) {
      return [e.keyword, e.title || '', e.desc || '', e.modes || 'both'].join(' | ');
    }).join('\n');

    // 模式示例按钮：往编辑框追加一行对应模式示例
    win.querySelectorAll('.bsk-ac-mode-btn').forEach(function (btn) {
      btn.onclick = function () {
        var line = MODE_SAMPLES[btn.getAttribute('data-mode')].line;
        var v = ta.value;
        if (v.length && !v.endsWith('\n')) v += '\n';
        ta.value = v + line;
        ta.focus();
        // 光标移到末尾
        ta.setSelectionRange(ta.value.length, ta.value.length);
      };
    });

    function close() {
      if (mask.parentNode) mask.parentNode.removeChild(mask);
    }

    win.querySelector('[data-x]').onclick = close;
    mask.onclick = function (e) { if (e.target === mask) close(); };
    win.querySelector('[data-cancel]').onclick = close;

    win.querySelector('[data-save]').onclick = function () {
      var name = nameInput.value.trim() || '未命名配置';
      var raw = ta.value.trim();
      var entries = [];
      var parseErr = '';

      if (raw.indexOf('{') === 0) {
        // 整段 JSON
        try {
          var obj = JSON.parse(raw);
          if (obj && Array.isArray(obj.entries)) {
            entries = obj.entries;
            if (obj.name) name = obj.name;
          } else { parseErr = 'JSON 需包含 entries 数组'; }
        } catch (e) { parseErr = 'JSON 解析失败：' + e.message; }
      } else {
        raw.split(/\r?\n/).forEach(function (line, i) {
          line = line.trim();
          if (!line) return;
          var parts = line.split('|').map(function (s) { return s.trim(); });
          if (!parts[0]) { parseErr = '第 ' + (i + 1) + ' 行缺少补全词'; return; }
          var mode = parts[3] || 'both';
          if (['both', 'ref', 'concat'].indexOf(mode) < 0) mode = 'both';
          entries.push({ keyword: parts[0], title: parts[1] || '', desc: parts[2] || '', modes: mode });
        });
      }

      if (parseErr) { err.textContent = parseErr; return; }
      cur.name = name;
      cur.entries = entries;
      LIB.save(cur, isNew ? undefined : cur.id);
      close();
      if (onSaved) onSaved();
    };

    mask.appendChild(win);
    document.body.appendChild(mask);
    nameInput.focus();
  }
  LIB._editConfig = editConfig;

  // 简易 downloadText（避免依赖编辑器内部函数）
  function downloadText(filename, text) {
    var blob = new Blob([text], { type: 'application/json;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  window.BSK_AUTO_LIB = LIB;
})();
