/* =========================================================================
 * BSK 提示词语法高亮 mode + textarea 兼容代理层（CodeMirror 5）
 * 用于 panel.js 的提示词库编辑框 与 节点卡片多行文本。
 * 提供：
 *   1) CodeMirror.defineMode('bsk-prompt')：
 *      - 行首 # 注释 → 绿色
 *      - @ 开头的 Anima 画师 → 紫色（单独一类）
 *      - `name (alias)` 形式的画师/角色/特殊触发词 → 蓝色
 *      - `(tag:weight)` 带权重提示词 → 按权重 0~10 色相渐变（权重越高越红）
 *   2) CodeMirror.defineTheme('bsk')：深色/浅色通用主题
 *   3) window.createBskTextareaProxy(cm, wrapper)：把 CodeMirror 包装成
 *      兼容 textarea API 的对象（.value/.selectionStart/.setSelectionRange/
 *      .scrollTop/.style.fontSize/.style.background/.addEventListener/
 *      .dispatchEvent/.focus/.parentElement/.closest），让 panel.js 里
 *      大量现有 textarea 操作代码无需改动。
 *   4) window.createBskPromptEditor(mountEl, opts)：一步创建 {cm, proxy}
 *
 * 注意：window 暴露语句放在文件最前（函数声明提升），即使下方 mode/theme
 * 注册抛错，createBskPromptEditor 也保证可用；错误会打印到 console 便于诊断。
 * ========================================================================= */
(function () {
  'use strict';
  if (!window.CodeMirror) { console.error('[bsk_prompt_mode] CodeMirror 未加载'); return; }

  // ===== textarea 兼容代理层（先定义并暴露）=====
  function createBskTextareaProxy(cm, wrapper) {
    var listeners = {};      // type -> [fn]
    var suppressChange = false;

    function posToOffset(pos) {
      var lines = cm.getValue().split('\n');
      var off = 0;
      for (var i = 0; i < pos.line && i < lines.length; i++) off += lines[i].length + 1;
      off += pos.ch;
      return off;
    }
    function offsetToPos(offset) {
      var text = cm.getValue();
      var line = 0, ch = 0, i;
      for (i = 0; i < offset && i < text.length; i++) {
        if (text[i] === '\n') { line++; ch = 0; } else ch++;
      }
      return { line: line, ch: ch };
    }

    var proxy = {
      get value() { return cm.getValue(); },
      set value(v) {
        // 程序化写回：对齐 vs_prompt_editor（视频工作台提示词编辑器）已验证的
        // 稳定路径 —— 用 replaceRange 做最小范围替换，绝不调用 setCursor/scrollTo。
        // 原因：在部分 CodeMirror 版本（如 ComfyUI 内嵌环境旧版/双加载）中，
        // setValue 后同步调用 setCursor({scroll:false})/scrollTo 会导致渲染错乱、
        // 编辑框内容"消失"，需鼠标点击触发重绘才恢复。replaceRange 不移动光标、
        // 不滚动、同步渲染，与视频工作台行为一致。
        v = v || '';
        var oldText = cm.getValue();
        if (oldText === v) return;
        suppressChange = true;
        try {
          var maxP = Math.min(oldText.length, v.length);
          var pl = 0;
          while (pl < maxP && oldText.charCodeAt(pl) === v.charCodeAt(pl)) pl++;
          var sl = 0;
          while (sl < maxP - pl && oldText.charCodeAt(oldText.length - 1 - sl) === v.charCodeAt(v.length - 1 - sl)) sl++;
          if (pl + sl >= maxP * 0.5) {
            // 公共部分占比高 → 仅替换中间差异段，最小化重绘且保持光标/滚动
            cm.replaceRange(v.substring(pl, v.length - sl), cm.posFromIndex(pl), cm.posFromIndex(oldText.length - sl));
          } else {
            cm.setValue(v);
          }
        } catch (e) {
          try { cm.setValue(v); } catch (e2) {}
        }
        suppressChange = false;
      },
      get selectionStart() { return posToOffset(cm.getCursor('from')); },
      get selectionEnd() { return posToOffset(cm.getCursor('to')); },
      setSelectionRange: function (start, end) {
        if (end == null) end = start;
        // 不带 options 的 setSelection：兼容旧版/双加载 CodeMirror
        // （部分版本不支持 {scroll:false} 第三参数，会当作 anchor 处理导致异常）
        cm.setSelection(offsetToPos(start), offsetToPos(end));
        if (!cm.hasFocus()) cm.focus();
      },
      get scrollTop() { return cm.getScrollInfo().top; },
      set scrollTop(v) { cm.scrollTo(null, v); },
      get scrollLeft() { return cm.getScrollInfo().left; },
      set scrollLeft(v) { cm.scrollTo(v, null); },
      style: {
        get fontSize() { return wrapper.style.fontSize; },
        set fontSize(v) {
          wrapper.style.fontSize = v;
          if (cm.refresh) setTimeout(function () { try { cm.refresh(); } catch (e) {} }, 0);
        },
        get background() { return wrapper.style.background; },
        set background(v) { wrapper.style.background = v; }
      },
      addEventListener: function (type, fn) {
        if (type === 'input') {
          cm.on('change', function (cm_, change) {
            if (suppressChange) return;
            if (change.origin === 'setValue') return;
            fn({ type: 'input', target: proxy });
          });
          (listeners.input = listeners.input || []).push(fn);
        } else if (type === 'scroll') {
          cm.on('scroll', function () { fn({ type: 'scroll', target: proxy }); });
          (listeners.scroll = listeners.scroll || []).push(fn);
        } else {
          wrapper.addEventListener(type, fn);
          (listeners[type] = listeners[type] || []).push(fn);
        }
      },
      removeEventListener: function (type, fn) {
        var arr = listeners[type];
        if (arr) {
          var idx = arr.indexOf(fn);
          if (idx >= 0) arr.splice(idx, 1);
        }
        wrapper.removeEventListener(type, fn);
      },
      dispatchEvent: function (e) {
        var arr = listeners[e.type] || [];
        for (var i = 0; i < arr.length; i++) arr[i](e);
      },
      focus: function () { cm.focus(); },
      get parentElement() { return wrapper.parentElement; },
      closest: function (sel) { return wrapper.closest ? wrapper.closest(sel) : null; },
      // 右键坐标 → 字符索引（CodeMirror 右键不移动光标，需用 coordsChar 换算）
      _clickCharIndex: function (clientX, clientY) {
        try {
          var pos = cm.coordsChar({ left: clientX, top: clientY });
          return posToOffset(pos);
        } catch (e) { return 0; }
      },
      _cm: cm,
      _wrapper: wrapper
    };
    // 反查用：外部模块（如 prompt_image_library.js）拿到 cm 后能取回这个代理，
    // 从而继续用 textarea 风格 API 读写提示词。
    try { cm.__bskProxy = proxy; } catch (e) {}
    return proxy;
  }

  function createBskPromptEditor(mountEl, opts) {
    opts = opts || {};
    if (!mountEl) return { cm: null, proxy: null };
    var cm = CodeMirror(mountEl, {
      value: opts.value || '',
      mode: 'bsk-prompt',
      theme: 'bsk',
      lineWrapping: true,
      lineNumbers: false,
      indentUnit: 2,
      tabSize: 2,
      placeholder: opts.placeholder || '',
      readOnly: opts.readOnly || false
    });
    var proxy = createBskTextareaProxy(cm, cm.getWrapperElement());
    return { cm: cm, proxy: proxy };
  }

  // ===== 暴露 API（提前，确保 panel.js 一定能拿到）=====
  window.createBskTextareaProxy = createBskTextareaProxy;
  window.createBskPromptEditor = createBskPromptEditor;
  // 版本标记：便于确认浏览器加载的是新文件（旧缓存无此标记）
  window.BSK_PROMPT_MODE_VERSION = '0.7.0';

  // ===== 权重颜色：0~10 映射为 0~20 档（每 0.5 一档），色相渐变 =====
  function weightColor(w, isDark) {
    var r = w - 1; // 以 1.0 为基准色
    if (w <= 1) {
      var t = Math.max(0, 1 - w);
      if (isDark) return 'hsl(' + (210 + t * 18) + ', ' + (25 + t * 45) + '%, ' + (60 + t * 15) + '%)';
      return 'hsl(' + (210 + t * 18) + ', ' + (25 + t * 45) + '%, ' + (30 + t * 15) + '%)';
    }
    var tt = Math.min(1, r / 9);
    if (isDark) return 'hsl(' + (45 - tt * 45) + ', ' + (85 + tt * 15) + '%, ' + (55 + tt * 15) + '%)';
    return 'hsl(' + (45 - tt * 45) + ', ' + (75 + tt * 25) + '%, ' + (30 + tt * 12) + '%)';
  }

  // ===== 基础主题 CSS（不依赖 CodeMirror.defineTheme，直接定义 .cm-s-bsk 样式）=====
  // CodeMirror 的 theme 选项只负责给容器加 cm-s-bsk class，外观完全由 CSS 决定。
  // 某些环境（如与 ComfyUI 自带 CodeMirror 冲突）defineTheme 可能缺失，故这里兜底注入。
  try {
    function injectBskCss() {
      if (document.getElementById('bsk-prompt-css')) return;
      var css = [
        '.cm-s-bsk.CodeMirror { background: transparent; color: #e8e2d0; }',
        '.cm-s-bsk .CodeMirror-cursor { border-left-color: #9fc1ff; }',
        '.cm-s-bsk .CodeMirror-selected { background: rgba(76,110,245,0.35) !important; }',
        '.cm-s-bsk.CodeMirror-focused .CodeMirror-selected { background: rgba(76,110,245,0.35) !important; }',
        '.cm-s-bsk .CodeMirror-activeline-background { background: rgba(255,255,255,0.03); }',
        '.cm-s-bsk .CodeMirror-placeholder { color: rgba(109,106,96,0.9); }',
        '.cm-s-bsk span.cm-comment { color: #5cb85c !important; font-style: italic !important; }',
        '.cm-s-bsk span.cm-anima-artist { color: #c792ea !important; font-weight: 600 !important; }',
        '.cm-s-bsk span.cm-artist-char { color: #82aaff !important; font-weight: 600 !important; }',
        '.cm-s-bsk span.cm-weighted { font-weight: 600 !important; }',
        // 浅色/粉白主题适配
        '.prompt-library-panel.light-mode .cm-s-bsk.CodeMirror { color: #2d3540; }',
        '.prompt-library-panel.light-mode .cm-s-bsk .CodeMirror-cursor { border-left-color: #667eea; }',
        '.prompt-library-panel.light-mode .cm-s-bsk span.cm-comment { color: #2e7d32 !important; }',
        '.prompt-library-panel.light-mode .cm-s-bsk span.cm-anima-artist { color: #8e44ad !important; }',
        '.prompt-library-panel.light-mode .cm-s-bsk span.cm-artist-char { color: #1d4ed8 !important; }'
      ].join('\n');
      var style = document.createElement('style');
      style.id = 'bsk-prompt-css';
      style.textContent = css;
      (document.head || document.documentElement).appendChild(style);
    }
    injectBskCss();
  } catch (e) {
    console.warn('[bsk_prompt_mode] 基础主题 CSS 注入失败:', e);
  }

  // ===== 权重 CSS（try-catch，失败不阻断核心 API）=====
  try {
    function injectWeightCss() {
      if (document.getElementById('bsk-prompt-weight-css')) return;
      var css = '';
      for (var i = 0; i <= 20; i++) {
        var w = i / 2;
        css += '.prompt-library-panel.dark-mode .cm-s-bsk .cm-w' + i + ',';
        css += '.config-card.dark-mode .cm-s-bsk .cm-w' + i + ',';
        css += '.config-card .cm-s-bsk .cm-w' + i + '{color:' + weightColor(w, true) + ' !important;}\n';
        css += '.prompt-library-panel.light-mode .cm-s-bsk .cm-w' + i + ',';
        css += '.config-card.light-mode .cm-s-bsk .cm-w' + i + '{color:' + weightColor(w, false) + ' !important;}\n';
      }
      var style = document.createElement('style');
      style.id = 'bsk-prompt-weight-css';
      style.textContent = css;
      (document.head || document.documentElement).appendChild(style);
    }
    injectWeightCss();
  } catch (e) {
    console.warn('[bsk_prompt_mode] 权重 CSS 注入失败:', e);
  }

  // ===== mode：bsk-prompt（try-catch，失败不阻断核心 API）=====
  try {
    CodeMirror.defineMode('bsk-prompt', function () {
      var RE_COMMENT = /^\s*#/;
      // 行中注释：# 开头，匹配到下一个逗号/全角逗号/行尾为止。
      // 用于行内被注释的提示词（Tag 禁用项 `# xxx`、中文注释 `# .. xxx`、翻译 `# . xxx`）
      // 变绿，而同一行其他未注释提示词保持普通色。
      var RE_INLINE_COMMENT = /^#\s?[^,，\n]*/;
      var RE_ANIMA = /^@[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5_\-.]*/;
      // @Anima 画师权重后缀（@wlop:1.40 中的 :1.40），独立 token 按权重渐变着色
      var RE_ANIMA_WEIGHT = /^:\s*(\d+(?:\.\d+)?)/;
      // 权重括号整体：(Curvilinear perspective:1.31) / (dog:0.12)；
      // 支持转义括号 \(...\)（ComfyUI 中避免括号被识别为权重标记）
      var RE_WEIGHT = /^\\?\([^()]*:\s*\d+(?:\.\d+)?\\?\)/;
      var RE_WEIGHT_NUM = /:(\s*\d+(?:\.\d+)?)\\?\)$/;
      // 括号块（无冒号）：(lucky star) / \(lucky star\) —— 作为整体消费不散开
      var RE_PAREN = /^\\?\([^()]{1,120}\\?\)/;
      // 画师/角色/触发词 `name (alias)` / `name \(alias\)`：名字部分允许空格，优先于普通 tag 匹配
      var RE_ALIAS = /^[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5'\- ]{0,60}\\?\([^():]{1,80}\\?\)/;
      // 纯名字（含空格，例如 "Izumi Konata"），普通色但整体消费
      var RE_NAME = /^[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5'\- ]{1,60}/;
      var RE_TAG = /^[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5'\-]{0,80}/;
      return {
        token: function (stream) {
          // 整行注释
          if (stream.sol() && RE_COMMENT.test(stream.string)) {
            stream.skipToEnd();
            return 'comment';
          }
          // 行中注释（非行首的 # 开头，到下一个逗号/行尾）：
          // 必须放在 @anima/权重/别名之前，使被注释的 @wlop:1.40 / (tag:1.5) 整体变绿
          if (!stream.sol() && stream.match(RE_INLINE_COMMENT)) {
            return 'comment';
          }
          // @Anima 画师（如 @wlop）
          if (stream.match(RE_ANIMA)) {
            return 'anima-artist';
          }
          // @Anima 画师权重后缀 :1.40 —— 按权重渐变着色
          if (stream.match(RE_ANIMA_WEIGHT)) {
            var aM = RE_ANIMA_WEIGHT.exec(stream.current());
            var aW = aM ? parseFloat(aM[1]) : 1;
            var aLevel = Math.max(0, Math.min(20, Math.round(aW * 2)));
            return 'weighted w' + aLevel;
          }
          // 带权重片段 (tag:1.5) / \(tag:1.5\) —— 必须优先于别名（不含冒号）
          if (stream.match(RE_WEIGHT)) {
            var m = stream.current().match(RE_WEIGHT_NUM);
            var w = m ? parseFloat(m[1]) : 1;
            var level = Math.max(0, Math.min(20, Math.round(w * 2)));
            // 注意：CodeMirror 渲染时会给 style 每段自动加 cm- 前缀，
            // 这里必须返回 'wX'（最终 class 为 cm-wX）；写 'cm-wX' 会变 cm-cm-wX 导致权重色失效
            return 'weighted w' + level;
          }
          // 画师/角色/特殊触发词：`name (alias)` 形式（含空格名，支持转义括号）
          if (stream.match(RE_ALIAS)) {
            return 'artist-char';
          }
          // 括号块整体（无权重）：(lucky star) / \(lucky star\) —— 整体消费不散开
          if (stream.match(RE_PAREN)) {
            return null;
          }
          // 纯名字（含空格）整体消费，避免被普通 tag 拆分
          if (stream.match(RE_NAME)) {
            return null;
          }
          // 普通 token
          if (stream.match(RE_TAG)) return null;
          stream.next();
          return null;
        }
      };
    });
    CodeMirror.defineMIME('text/x-bsk-prompt', 'bsk-prompt');
  } catch (e) {
    console.error('[bsk_prompt_mode] mode 注册失败:', e);
  }

  // ===== 主题：bsk（外观已由注入的 .cm-s-bsk CSS 完整覆盖）=====
  // defineTheme 在部分环境缺失（如与 ComfyUI 自带 CodeMirror 冲突），
  // 此时主题外观仍正常，故静默降级不打印错误。
  try {
    if (typeof CodeMirror.defineTheme === 'function') {
      CodeMirror.defineTheme('bsk', {
        background: 'transparent',
        foreground: '#e8e2d0',
        cursor: '#9fc1ff',
        selection: 'rgba(76,110,245,0.35)',
        activeLine: 'rgba(255,255,255,0.03)'
      });
    }
  } catch (e) { /* 忽略：CSS 已兜底 */ }
})();
