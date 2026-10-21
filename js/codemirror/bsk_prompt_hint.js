/* =========================================================================
 * BSK 提示词编辑器增强层（CodeMirror 5）
 * 提供：
 *   1) CodeMirror.hint.bskPrompt —— 上下文感知提示词自动补全
 *      - H3 官方关键词库（window.BSK_KW_LIB，由 vs_prompt_editor.js 暴露）
 *      - 提示词库 tags / @画师 / 分类名（通过 window.BSK_HINT_OPTS.tagsFn 等动态获取）
 *      - 内置模板（权重括号、[Shot N]、<Subject N>、分类头等）
 *      - 文档内已有词（CodeMirror.hint.anyword 合并）
 *   2) window.bskSetupEditorEnhancements(cm, opts) —— 统一挂载全部增强：
 *      自动补全 / 括号匹配 / 当前行高亮 / 自动刷新 / 搜索 / 光标词高亮 /
 *      多光标(核心自带) / 分类折叠 / 细滚动条 / 底部滚动留白 / 状态栏(字符·标签·括号)
 *
 * 依赖（js/codemirror/addon/ 下的 UMD addon，缺失时自动降级不报错）：
 *   hint/show-hint, hint/anyword-hint, edit/matchbrackets, selection/active-line,
 *   display/autorefresh, search/search, search/searchcursor, search/match-highlighter,
 *   dialog/dialog, fold/foldcode, fold/foldgutter, fold/comment-fold,
 *   scroll/simplescrollbars, scroll/scrollpastend
 * ========================================================================= */
(function () {
  'use strict';
  if (!window.CodeMirror) { console.error('[bsk_prompt_hint] CodeMirror 未加载'); return; }
  var CodeMirror = window.CodeMirror;

  // ===== 注入样式（hint 列表 / 状态栏 / 折叠箭头 统一 ≥15px 与配色）=====
  function injectCss() {
    if (document.getElementById('bsk-hint-css')) return;
    var css = [
      /* 补全下拉：字体/行高/宽度明确撑大，!important 覆盖 CodeMirror 默认 90% 字号，
         深色主题下可见性高 */
      '.CodeMirror-hints { font-size: 15px !important; z-index: 20000; min-width: 260px; max-width: 640px; max-height: 320px; overflow-y: auto; padding: 5px !important; line-height: 22px !important; background: #26262c !important; border: 1px solid rgba(255,255,255,.16) !important; border-radius: 6px !important; box-shadow: 0 6px 24px rgba(0,0,0,.5) !important; }',
      '.CodeMirror-hint { font-size: 15px !important; padding: 6px 10px !important; color: #e8e2d0 !important; line-height: 22px !important; white-space: normal; word-break: break-word; }',
      '.CodeMirror-hint:hover, .CodeMirror-hint-active { color: #fff !important; background: rgba(102,126,234,.4) !important; }',
      '.bsk-hint-kw .CodeMirror-hint { color: #7aa2ff !important; }',
      '.bsk-hint-tag .CodeMirror-hint { color: #f78fb3 !important; }',
      '.bsk-hint-artist .CodeMirror-hint { color: #c792ea !important; }',
      '.bsk-hint-tpl .CodeMirror-hint { color: #7fe3c8 !important; }',
      '.bsk-hint-cat .CodeMirror-hint { color: #ffd27a !important; }',
      '.bsk-hint-main { font-weight: 600; }',
      '.bsk-hint-cn { margin-left: 8px; font-size: 15px !important; color: #c9c2ad !important; }',
      '.bsk-hint-desc { margin-left: 8px; font-size: 15px !important; color: #9a9a9a !important; }',
      /* light-mode：补全下拉/状态栏适配浅色背景 */
      '.prompt-library-panel.light-mode .CodeMirror-hints, .vs-prompt-editor.light-mode .CodeMirror-hints { background: #fff !important; border-color: rgba(0,0,0,.18) !important; box-shadow: 0 6px 24px rgba(0,0,0,.18) !important; }',
      '.prompt-library-panel.light-mode .CodeMirror-hint, .vs-prompt-editor.light-mode .CodeMirror-hint { color: #222 !important; }',
      '.prompt-library-panel.light-mode .CodeMirror-hint:hover, .prompt-library-panel.light-mode .CodeMirror-hint-active, .vs-prompt-editor.light-mode .CodeMirror-hint:hover, .vs-prompt-editor.light-mode .CodeMirror-hint-active { color: #000 !important; background: rgba(102,126,234,.18) !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-desc { color: #777 !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-cn { color: #6b7280 !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-kw .CodeMirror-hint { color: #1a4f9c !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-tag .CodeMirror-hint { color: #b0356e !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-artist .CodeMirror-hint { color: #7c4fb8 !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-tpl .CodeMirror-hint { color: #0e8a6a !important; }',
      '.prompt-library-panel.light-mode .bsk-hint-cat .CodeMirror-hint { color: #a06b00 !important; }',
      /* 搜索对话框文字 ≥15px */
      '.CodeMirror-dialog { font-size: 15px; }',
      '.CodeMirror-dialog input { font-size: 15px; }',
      /* 状态栏（编辑器底部） */
      '.cm-status-bar { position: absolute; left: 0; right: 0; bottom: 0; height: 26px; line-height: 26px; padding: 0 10px; font-size: 15px; color: rgba(230,225,210,.65); background: rgba(0,0,0,.32); border-top: 1px solid rgba(255,255,255,.07); pointer-events: none; z-index: 5; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }',
      '.CodeMirror-scroll { padding-bottom: 26px !important; }',
      '.prompt-library-panel.light-mode .cm-status-bar { color: rgba(0,0,0,.55); background: rgba(0,0,0,.05); border-top-color: rgba(0,0,0,.08); }',
      '.prompt-library-panel.light-mode .cm-status-bar .cm-status-warn { color: #dc2626; }',
      '.cm-status-bar .cm-status-warn { color: #ff9e9e; }'
    ].join('\n');
    var style = document.createElement('style');
    style.id = 'bsk-hint-css';
    style.textContent = css;
    (document.head || document.documentElement).appendChild(style);
  }
  try { injectCss(); } catch (e) { console.warn('[bsk_prompt_hint] CSS 注入失败:', e); }

  // ===== H3 智能补全（结构化感知，mode='h3'）=====
  var H3_FIELD_RE = /^(integrated_multimodal_description|overall_soundscape|non_diegetic_music|subject_definitions|summary|retention_analysis|detailed_description)\s*:/;
  var H3_LANG_LIST = ['English', 'Chinese', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Spanish', 'Portuguese', 'Italian', 'Arabic'];

  // 候选行渲染（模块级，buildCandidates / buildH3Candidates 共用）
  // show-hint 的 render 回调签名：render(li, completionData, hintItem)
  // 第三参数才是当前候选项；旧代码误读第二参数(data=整个补全对象)导致
  // displayText 为 undefined → 候选项文本空白、li 只剩 padding 高度
  function renderHintItem(el, data, cur) {
    cur = cur || data;
    el.innerHTML = '';
    // 主显 = 插入的英文（displayText）；cn = 中文名称；desc = 中文说明
    var main = document.createElement('span');
    main.className = 'bsk-hint-main';
    main.textContent = cur.displayText || cur.text || '';
    el.appendChild(main);
    if (cur.cn) {
      var cn = document.createElement('span');
      cn.className = 'bsk-hint-cn';
      cn.textContent = ' ' + cur.cn;
      el.appendChild(cn);
    }
    if (cur.desc) {
      var d = document.createElement('span');
      d.className = 'bsk-hint-desc';
      d.textContent = ' ' + cur.desc;
      el.appendChild(d);
    }
  }

  // 当前编辑器是否为 H3 mode（vs_prompt_editor.js 的 CodeMirror mode 'h3'）
  function isH3Mode(cm) {
    try { return (cm.getOption('mode') || '') === 'h3'; } catch (e) { return false; }
  }

  // 当前工作台 MiniMax 模式：'ref' = 多模态参考（Ref2VA）| 'concat' = 首尾帧拼接（FL2VA）
  function currentH3Mode() {
    try {
      var st = window.BSK_VS_STATE;
      return (st && st.minimaxMode === 'ref') ? 'ref' : 'concat';
    } catch (e) { return 'concat'; }
  }
  // 关键词条目是否对当前模式可用（modes: 'both' | 'ref' | 'concat'，默认 both）
  function kwVisibleInMode(k) {
    var m = (k && k.modes) || 'both';
    return m === 'both' || m === currentH3Mode();
  }

  // 从光标行向上找最近的 H3 段头（字段名 / 对齐段），返回 'summary'|'alignment'|...|''
  function detectH3Section(cm, lineNo) {
    for (var i = lineNo; i >= 0; i--) {
      var t = cm.getLine(i) || '';
      if (/For the target video,|How the reference pictures align/.test(t)) return 'alignment';
      var m = t.match(H3_FIELD_RE);
      if (m) return m[1];
    }
    return '';
  }

  // 文档内 max 编号 + 1（如 [Shot N] / <Subject N> / (Sx)）
  function nextH3Seq(cm, re) {
    var text = cm.getValue();
    var arr = text.match(re);
    if (!arr) return 1;
    var max = 0;
    for (var i = 0; i < arr.length; i++) {
      var n = parseInt((arr[i].match(/\d+/) || [0])[0], 10);
      if (n > max) max = n;
    }
    return max + 1;
  }

  // [Shot N] At MM:SS.mmm 时间戳：优先读 BSK_VS_STATE 的选中段/组起始帧换算，
  // 取不到则返回空（调用处输出不带时间戳的 [Shot N]）
  function h3ShotTimestamp() {
    try {
      var st = window.BSK_VS_STATE;
      if (!st) return '';
      var fps = (st.params && st.params.frameRate) || 24;
      var start = null;
      var sel = st.selectedTarget;
      if (sel) {
        if (sel.groupId) {
          // concat 模式：起始帧 = 前序组 durationFrames 累加
          var groups = st.concatGroups || [];
          for (var i = 0; i < groups.length; i++) {
            if (groups[i].id === sel.groupId) {
              var acc = 1;
              for (var j = 0; j < i; j++) acc += (groups[j].durationFrames || (st.params && st.params.durationFrames) || 1);
              start = acc;
              break;
            }
          }
        } else if (sel.type === 'ref') {
          // ref 模式：匹配 minimaxRef 段，取其 start
          var segs = st.segments || [];
          for (var k = 0; k < segs.length; k++) {
            var r = segs[k].minimaxRef;
            if (r && r.slot === sel.slot && r.slotKind === sel.kind) { start = segs[k].start; break; }
          }
        }
      }
      if (start == null) return '';
      var sec = Math.max(0, Math.round((Number(start) - 1) / fps));
      var mm = Math.floor(sec / 60);
      var ss = Math.floor(sec % 60);
      var mmm = Math.round((sec - Math.floor(sec)) * 1000);
      function p2(n) { return (n < 10 ? '0' : '') + n; }
      function p3(n) { return (n < 100 ? (n < 10 ? '00' : '0') : '') + n; }
      return 'At ' + p2(mm) + ':' + p2(ss) + '.' + p3(mmm) + ', ';
    } catch (e) { return ''; }
  }

  // H3 结构化智能补全：段落上下文感知 + 短前缀自动编号 + 中文 title/desc 匹配
  function buildH3Candidates(cm, opts) {
    var cur = cm.getCursor();
    var line = cm.getLine(cur.line) || '';
    var before = line.slice(0, cur.ch);
    var m = before.match(/[\w\u4e00-\u9fa5@:\.\-\[\]<>]*$/);
    var word = m ? m[0] : '';
    var wordStart = cur.ch - word.length;
    var section = detectH3Section(cm, cur.line);
    var wordLower = word.toLowerCase();

    // 动态数据源
    var tags = [], artists = [], categories = [];
    if (opts.tagsFn) { try { tags = opts.tagsFn() || []; } catch (e) {} }
    if (opts.artistsFn) { try { artists = opts.artistsFn() || []; } catch (e) {} }
    if (opts.categoriesFn) { try { categories = opts.categoriesFn() || []; } catch (e) {} }

    // 模式隔离：Ref2VA 专属关键词（subject_definitions/参考标签/任务类型/保留标记）仅多参考模式补全
    // 用户禁用内置补全时清空 kwLib（仅保留用户输入/文档已有）
    var kwLib = window.BSK_AUTO_LIB && window.BSK_AUTO_LIB.isBuiltinDisabled()
      ? []
      : ((window.BSK_KW_LIB && window.BSK_KW_LIB.flat) ? window.BSK_KW_LIB.flat : []).filter(kwVisibleInMode);

    var list = [], seen = {};
    function push(text, displayText, desc, cls, cn) {
      if (!text || seen[text]) return;
      seen[text] = 1;
      var o = { text: text, displayText: displayText || text, desc: desc || '', className: cls || '', render: renderHintItem };
      if (cn) o.cn = cn;
      list.push(o);
    }
    // 中文匹配：keyword + title + desc 三路（title/desc 为中文描述，中文输入也能搜到）
    function hit(k, w) {
      if (!w) return true;
      var low = w.toLowerCase();
      var key = ((k.keyword || k.insert) || '').toLowerCase();
      var title = (k.title || '').toLowerCase();
      var desc = (k.desc || '').toLowerCase();
      return key.indexOf(low) >= 0 || title.indexOf(w) >= 0 || desc.indexOf(w) >= 0;
    }
    // 段落过滤：section 为空（新文档/段外）时全推
    function inSection(k) {
      var sec = k.section || '';
      if (!section) return true;
      if (section === 'alignment') return sec === 'ref' || sec === 'field';
      if (section === 'summary') return sec === 'summary' || sec === 'field';
      if (section === 'retention_analysis') return sec === 'retention' || sec === 'field';
      if (section === 'subject_definitions') return sec === 'ref' || sec === 'field';
      if (section === 'overall_soundscape' || section === 'non_diegetic_music') return sec === 'field' || sec === 'dialogue' || sec === 'special';
      return sec === 'shot' || sec === 'camera' || sec === 'style' || sec === 'dialogue' || sec === 'ref' || sec === 'special';
    }

    // ---- 场景 A：行首（新段起始）→ 字段段头 + 对齐段模板（按模式隔离）----
    if (before.trim() === '') {
      if (!section) {
        // 三段基础字段两种模式都有
        push('integrated_multimodal_description: ', 'integrated_multimodal_description', '主描述段（T2VA/I2VA/FL2VA/L2VA 必填），全英文', 'bsk-hint-kw', '主描述');
        push('overall_soundscape: ', 'overall_soundscape', '整体音景段：环境音/物理动作音/非语言人声（1–4 句）；整片静音才写 N/A', 'bsk-hint-kw', '整体音景');
        push('non_diegetic_music: ', 'non_diegetic_music', '背景音乐段：仅观众听到的配乐（1–3 句）；无配乐写 N/A', 'bsk-hint-kw', '背景音乐');
        if (currentH3Mode() === 'ref') {
          // 多参考 Ref2VA：六段结构，补四段专属段头
          push('subject_definitions:\n', 'subject_definitions', 'Ref2VA 第 1 段：定义 <Subject/Picture/Video/Audio> 标签', 'bsk-hint-kw', '主体定义');
          push('summary: ', 'summary', 'Ref2VA 第 2 段：任务综述，带 [任务类型 + ...] 前缀', 'bsk-hint-kw', '任务综述');
          push('retention_analysis:\n', 'retention_analysis', 'Ref2VA 第 3 段：逐素材声明保留程度（可见 4 种 / 音频 4 种）', 'bsk-hint-kw', '保留分析');
          push('detailed_description: ', 'detailed_description', 'Ref2VA 第 4 段：主描述，建议 350–500 词', 'bsk-hint-kw', '详细描述');
        } else {
          // 首尾帧 FL2VA：三段 + 对齐段模板（Ref2VA 无对齐段，不提示）
          push('How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the S.SS-second mark of the target video.', 'FL2VA 首尾帧对齐段', 'FL2VA：Picture 1 首帧 0.00s + Picture 2 尾帧 S.SS s（单镜 Shot 1）', 'bsk-hint-tpl');
          push('For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.', 'I2VA 首帧对齐段', 'I2VA：首帧 0.00s 锚定 <Picture 1>', 'bsk-hint-tpl');
          push('How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.', 'L2VA 尾帧对齐段', 'L2VA：尾帧 S.SS s 锚定 <Picture 1>，N=最后一镜序号', 'bsk-hint-tpl');
        }
      } else {
        // 段内行首：推段内允许的高频词（限前 12 个避免刷屏）；主显英文、附中文名
        var kwInSec = kwLib.filter(inSection);
        var cnt = 0;
        for (var si = 0; si < kwInSec.length && cnt < 12; si++) {
          var kk = kwInSec[si];
          push(kk.insert || kk.keyword, kk.insert || kk.keyword, kk.desc || '', 'bsk-hint-kw', kk.title || '');
          cnt++;
        }
      }
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 B：短前缀（编号自动递增 / 对白 / 语言标签）----
    if (/^</.test(word)) {
      push('<d>[Chinese] </d>', '<d>[Chinese] </d>', '对白块：语言标签用英文枚举（如 [Chinese]），<d> 内逐字保留原话', 'bsk-hint-tpl');
      // 参考标签 <Subject/Picture/Video/Audio N> 仅多参考（Ref2VA）模式可用；首尾帧对齐段用无尖括号写法
      if (currentH3Mode() === 'ref') {
        push('<Subject ' + nextH3Seq(cm, /<Subject\s+\d+>/g) + '> ', '<Subject N> ', '参考主体（编号自动递增），定义写入 subject_definitions', 'bsk-hint-tpl');
        push('<Picture ' + nextH3Seq(cm, /<Picture\s+\d+>/g) + '> ', '<Picture N> ', '参考图（编号自动递增），整图当帧锚点才用', 'bsk-hint-tpl');
        push('<Video ' + nextH3Seq(cm, /<Video\s+\d+>/g) + '> ', '<Video N> ', '参考视频（编号自动递增）', 'bsk-hint-tpl');
        push('<Audio ' + nextH3Seq(cm, /<Audio\s+\d+>/g) + '> ', '<Audio N> ', '参考音频（编号自动递增，绑定说话人可附 (Sx)）', 'bsk-hint-tpl');
      }
      push('<scenetrans> ', '<scenetrans>', '跨切连续标记：对白跨切两端都放', 'bsk-hint-tpl');
      push('<cutoff>', '<cutoff>', '语音截断标记', 'bsk-hint-tpl');
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }
    if (/^\(/.test(word)) {
      var nS = nextH3Seq(cm, /\(S\d+\)/g);
      push('(S' + nS + ') ', '(S1) ', '说话人 ID（全局顺序自动递增，跨镜复用同 ID）', 'bsk-hint-tpl');
      push('(S' + nS + ',S' + (nS + 1) + ') ', '(S1,S2) ', '多人同说 ID（按顺序递增）', 'bsk-hint-tpl');
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }
    if (/^\[/.test(word)) {
      // 任务类型（summary 前缀）：输入 [video / [key / [audio 等必须能直接命中
      kwLib.forEach(function (k) {
        if (k.section === 'summary' && hit(k, word)) {
          push(k.insert || k.keyword, k.insert || k.keyword, k.desc || '', 'bsk-hint-tpl', k.title || '');
        }
      });
      push('[Shot ' + nextH3Seq(cm, /\[Shot\s+\d+\]/g) + '] ' + h3ShotTimestamp(), '[Shot N] At MM:SS.mmm, ', '镜头块（编号自动递增；时间戳来自当前时间线选中段/组）', 'bsk-hint-tpl', '镜头块');
      for (var li = 0; li < H3_LANG_LIST.length; li++) {
        push('[' + H3_LANG_LIST[li] + ']', '[' + H3_LANG_LIST[li] + ']', '对白语言标签（英文枚举，禁止 [中文]）', 'bsk-hint-tpl', '语言标签');
      }
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 C：段感知词表（keyword + 中文 title/desc 匹配）----
    // 主显英文插入文本 + 中文名（cn）+ 中文说明（desc）；段过滤优先，
    // 若当前段内无任何命中则回退全库（避免「打出关键词却没列表」）
    var secPool = kwLib.filter(inSection);
    var secHit = word ? secPool.filter(function (k) { return hit(k, word); }) : secPool;
    var pool = secHit.length ? secHit : kwLib.filter(function (k) { return hit(k, word); });
    pool.forEach(function (k) {
      push(k.insert || k.keyword, k.insert || k.keyword, k.desc || '', 'bsk-hint-kw', k.title || '');
    });

    // ---- 用户自动补全配置候选（多选配置合并，modes 隔离由库内完成）----
    if (window.BSK_AUTO_LIB) {
      var userH3 = window.BSK_AUTO_LIB.getUserCandidates(currentH3Mode());
      userH3.forEach(function (u) {
        if (hit(u, word)) {
          push(u.keyword, u.keyword, (u.desc ? u.desc + ' · ' : '') + (u.source || '用户配置'), 'bsk-hint-kw', u.title || '');
        }
      });
    }

    // ---- 场景 D：文档已有标签（H3 下仅保留短词/带编号/含中文项，过滤句子碎片）----
    tags.forEach(function (t) {
      var s = t.text;
      if (!s || s.indexOf('@') === 0) return;
      s = String(s).trim();
      if (!s || s.length > 30) return;
      if (!hit({ keyword: s, title: '', desc: '' }, word)) return;
      push(s + ' ', s, t.count ? '出现 ' + t.count + ' 次' : '文档已有', 'bsk-hint-tag');
    });
    if (CodeMirror.hint && CodeMirror.hint.anyword) {
      try {
        var any = CodeMirror.hint.anyword(cm, { word: /[\w\u4e00-\u9fa5'\-]{3,}/ });
        if (any && any.list) {
          any.list.forEach(function (x) {
            var t = typeof x === 'string' ? x : x.text;
            if (t && t.length >= 3 && t.length <= 30 && hit({ keyword: t, title: '', desc: '' }, word)) {
              push(t + ' ', t, '文档已有', 'bsk-hint-tag');
            }
          });
        }
      } catch (e) {}
    }

    // 排序：词表 > 短前缀模板 > 文档已有；同分按文本
    var order = { 'bsk-hint-kw': 0, 'bsk-hint-tpl': 1, 'bsk-hint-tag': 2 };
    list.sort(function (a, b) {
      var oa = order[a.className] !== undefined ? order[a.className] : 3;
      var ob = order[b.className] !== undefined ? order[b.className] : 3;
      if (oa !== ob) return oa - ob;
      var da = (a.desc || '').length, db = (b.desc || '').length;
      if (db !== da) return db - da;
      return a.text.localeCompare(b.text);
    });
    if (list.length > 40) list = list.slice(0, 40);

    return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
  }

  // ===== 候选构建（上下文感知）=====
  function buildCandidates(cm, opts) {
    // H3 结构化智能补全优先（vs_prompt_editor 的 h3 mode）
    if (isH3Mode(cm)) return buildH3Candidates(cm, opts);
    var cur = cm.getCursor();
    var line = cm.getLine(cur.line) || '';
    var before = line.slice(0, cur.ch);

    // 当前正在输入的词（含 @ : [ ] < > . - 与字母/数字/中文）
    var m = before.match(/[\w\u4e00-\u9fa5@:\.\-\[\]<>]*$/);
    var word = m ? m[0] : '';
    var wordStart = cur.ch - word.length;

    // 动态数据源
    var tags = [];
    var artists = [];
    var categories = [];
    if (opts.tagsFn) { try { tags = opts.tagsFn() || []; } catch (e) {} }
    if (opts.artistsFn) { try { artists = opts.artistsFn() || []; } catch (e) {} }
    if (opts.categoriesFn) { try { categories = opts.categoriesFn() || []; } catch (e) {} }

    // H3 官方关键词库（用户禁用内置时清空）
    var kwLib = window.BSK_AUTO_LIB && window.BSK_AUTO_LIB.isBuiltinDisabled()
      ? []
      : ((window.BSK_KW_LIB && window.BSK_KW_LIB.flat) ? window.BSK_KW_LIB.flat : []);

    var list = [];
    var seen = {};

    function push(text, displayText, desc, cls, cn) {
      if (!text || seen[text]) return;
      seen[text] = 1;
      var o = { text: text, displayText: displayText || text, desc: desc || '', className: cls || '', render: renderHint };
      if (cn) o.cn = cn;
      list.push(o);
    }

    function renderHint(el, data, cur) {
      renderHintItem(el, data, cur);
    }

    // 匹配函数：前缀优先，包含次之
    function matches(item) {
      if (!word) return true;
      var low = word.toLowerCase();
      var t = item.toLowerCase();
      return t.indexOf(low) === 0 || t.indexOf(low) > 0;
    }

    // ---- 场景 1：@ 画师 ----
    var atMatch = before.match(/@([\w\u4e00-\u9fa5]*)$/);
    if (atMatch) {
      var atPrefix = atMatch[1];
      artists.forEach(function (a) {
        var t = a.text || a;
        if (t.indexOf('@') === 0 && t.slice(1).toLowerCase().indexOf(atPrefix.toLowerCase()) === 0) {
          push(t + ', ', t, a.count ? '出现 ' + a.count + ' 次' : '画师', 'bsk-hint-artist');
        }
      });
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 2：分类头 # -- ----
    var catMatch = before.match(/#\s*--\s*([\w\u4e00-\u9fa5]*)$/);
    if (catMatch) {
      var catPrefix = catMatch[1];
      categories.forEach(function (c) {
        var n = c.name || c;
        if (n.toLowerCase().indexOf(catPrefix.toLowerCase()) === 0) {
          push('# --' + n, '# --' + n, c.count ? '含 ' + c.count + ' 项' : '分类', 'bsk-hint-cat');
        }
      });
      var catStart = cur.ch - catPrefix.length;
      return { list: list, from: CodeMirror.Pos(cur.line, catStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 3：方括号模板 ----
    if (/^\[/.test(word)) {
      kwLib.forEach(function (k) {
        if (k.keyword && k.keyword.indexOf('[') === 0 && matches(k.keyword)) {
          push(k.insert || k.keyword, k.keyword.split(' ')[0], k.desc || '', 'bsk-hint-tpl');
        }
      });
      push('[Shot 1] At 00:00.000, ', '[Shot N] At MM:SS.mmm, ', '镜头块标记', 'bsk-hint-tpl');
      push('[keyframe completion] ', '[keyframe completion]', '关键帧补全任务', 'bsk-hint-tpl');
      push('[video continuation] ', '[video continuation]', '视频续写任务', 'bsk-hint-tpl');
      push('[Chinese]', '[Chinese]', '对白语言标签', 'bsk-hint-tpl');
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 4：尖括号模板 ----
    if (/^</.test(word)) {
      kwLib.forEach(function (k) {
        if (k.keyword && k.keyword.indexOf('<') === 0 && matches(k.keyword)) {
          push(k.insert || k.keyword, k.keyword, k.desc || '', 'bsk-hint-tpl');
        }
      });
      push('<Subject 1> ', '<Subject N> ', '参考主体标签', 'bsk-hint-tpl');
      push('<Picture 1> ', '<Picture N> ', '参考图标签', 'bsk-hint-tpl');
      push('<Video 1> ', '<Video N> ', '参考视频标签', 'bsk-hint-tpl');
      push('<Audio 1> ', '<Audio N> ', '参考音频标签', 'bsk-hint-tpl');
      push('<scenetrans> ', '<scenetrans>', '跨切连续标记', 'bsk-hint-tpl');
      push('<cutoff>', '<cutoff>', '语音截断标记', 'bsk-hint-tpl');
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 5：括号模板 / 权重 ----
    if (/^\(/.test(word)) {
      push('(word:1.2)', '(word:1.2)', '权重模板', 'bsk-hint-tpl');
      push('(word)', '(word)', '无权重括号', 'bsk-hint-tpl');
      push('(S1)', '(S1)', '说话人 ID', 'bsk-hint-tpl');
      push('(S1,S2)', '(S1,S2)', '多人说话 ID', 'bsk-hint-tpl');
      tags.forEach(function (t) {
        var n = t.text;
        if (n.indexOf('(') === 0 && matches(n)) {
          push(n + ', ', n, t.count ? '出现 ' + t.count + ' 次' : '', 'bsk-hint-tag');
        }
      });
      return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
    }

    // ---- 场景 6：默认（tags + H3 关键词 + 模板 + 文档已有词）----
    // H3 关键词（字段/镜头/转场等）；主显英文插入文本 + 中文名
    kwLib.forEach(function (k) {
      var key = k.keyword || k.insert || '';
      if (matches(key)) {
        push(k.insert || key, k.insert || key, k.desc || '', 'bsk-hint-kw', k.title || '');
      }
    });
    // 用户自动补全配置候选（普通模式按 concat 过滤：both/concat 生效）
    if (window.BSK_AUTO_LIB) {
      var userC = window.BSK_AUTO_LIB.getUserCandidates('concat');
      userC.forEach(function (u) {
        var keyTxt = u.keyword || '';
        if (matches(keyTxt)) {
          push(keyTxt, keyTxt, (u.desc ? u.desc + ' · ' : '') + (u.source || '用户配置'), 'bsk-hint-kw', u.title || '');
        }
      });
    }
    // 提示词库 tags（带权重括号的 tag 保留原始形态）
    tags.forEach(function (t) {
      if (t.text.indexOf('@') === 0) return;       // 画师单独场景
      if (t.text.indexOf('(') === 0 && word.indexOf('(') === -1) return; // 括号项在场景 5
      if (matches(t.text)) {
        push(t.text + ', ', t.text, t.count ? '出现 ' + t.count + ' 次' : '', 'bsk-hint-tag');
      }
    });
    // 模板
    push('masterpiece, ', 'masterpiece', '品质词', 'bsk-hint-tpl');
    push('best quality, ', 'best quality', '品质词', 'bsk-hint-tpl');
    push('# --分类名', '# --分类名', '新建分类', 'bsk-hint-cat');
    // 文档内已有词（anyword 合并，限长词减少噪音）
    if (CodeMirror.hint && CodeMirror.hint.anyword) {
      try {
        var any = CodeMirror.hint.anyword(cm, { word: /[\w\u4e00-\u9fa5'\-]{3,}/ });
        if (any && any.list) {
          any.list.forEach(function (x) {
            var t = typeof x === 'string' ? x : x.text;
            if (t && t.length >= 3 && matches(t)) {
              push(t + ', ', t, '文档已有', 'bsk-hint-tag');
            }
          });
        }
      } catch (e) {}
    }

    // 排序：H3 关键词 > tag(按次数) > 模板；同分按文本
    var order = { 'bsk-hint-kw': 0, 'bsk-hint-tag': 1, 'bsk-hint-tpl': 2, 'bsk-hint-cat': 3, '': 4 };
    list.sort(function (a, b) {
      var oa = order[a.className] !== undefined ? order[a.className] : 4;
      var ob = order[b.className] !== undefined ? order[b.className] : 4;
      if (oa !== ob) return oa - ob;
      var da = (a.desc || '').length, db = (b.desc || '').length;
      if (db !== da) return db - da;
      return a.text.localeCompare(b.text);
    });
    if (list.length > 40) list = list.slice(0, 40);

    return { list: list, from: CodeMirror.Pos(cur.line, wordStart), to: CodeMirror.Pos(cur.line, cur.ch) };
  }

  // ===== 注册 hint =====
  if (CodeMirror.hint) {
    CodeMirror.hint.bskPrompt = function (cm, options) {
      var opts = options || {};
      var shared = window.BSK_HINT_OPTS || {};
      return buildCandidates(cm, {
        tagsFn: opts.tagsFn || shared.tagsFn,
        artistsFn: opts.artistsFn || shared.artistsFn,
        categoriesFn: opts.categoriesFn || shared.categoriesFn
      });
    };
  }

  // ===== 状态栏 =====
  var H3_SEC_NAMES = {
    alignment: '对齐段',
    integrated_multimodal_description: '主描述',
    overall_soundscape: '整体音景',
    non_diegetic_music: '背景音乐',
    subject_definitions: '主体定义',
    summary: '任务综述',
    retention_analysis: '保留分析',
    detailed_description: '详细描述'
  };
  function attachStatusBar(cm) {
    var wrapper = cm.getWrapperElement();
    if (!wrapper || wrapper.querySelector('.cm-status-bar')) return;
    var bar = document.createElement('div');
    bar.className = 'cm-status-bar';
    wrapper.appendChild(bar);
    function update() {
      var text = cm.getValue();
      var chars = text.length;
      var tags = text.split(/[,，]/).filter(function (s) {
        return s.trim() && !/^\s*#/.test(s);
      }).length;
      var open = (text.match(/\(/g) || []).length;
      var close = (text.match(/\)/g) || []).length;
      var diff = open - close;
      var paren = diff === 0 ? '括号 平衡' : (diff > 0 ? '缺 ' + diff + ' 个 )' : '多 ' + (-diff) + ' 个 )');
      var html = '';
      // 当前光标所在 H3 段（段感知提示，帮用户确认正写在哪个字段）
      if (isH3Mode(cm)) {
        var sec = detectH3Section(cm, cm.getCursor().line);
        if (sec) {
          var label = H3_SEC_NAMES[sec] || sec;
          html += '当前段 ' + label + ' · ';
        }
      }
      html += '字符 ' + chars + ' · 标签 ' + tags + ' · ' + (diff === 0 ? paren : '<span class="cm-status-warn">' + paren + '</span>');
      bar.innerHTML = html;
    }
    cm.on('change', update);
    cm.on('cursorActivity', update);
    update();
  }

  // ===== 分类折叠 rangeFinder（# -- 分类头块）=====
  function categoryRangeFinder(cm, start) {
    var lineText = cm.getLine(start.line) || '';
    if (!/^\s*#\s*--/.test(lineText)) {
      // 回退到通用注释折叠
      if (CodeMirror.fold && CodeMirror.fold.comment) {
        return CodeMirror.fold.comment(cm, start);
      }
      return null;
    }
    var from = CodeMirror.Pos(start.line, lineText.length);
    var i, t;
    for (i = start.line + 1; i < cm.lineCount(); i++) {
      t = cm.getLine(i) || '';
      if (/^\s*#\s*--/.test(t)) return { from: from, to: CodeMirror.Pos(i, 0) };
    }
    return { from: from, to: CodeMirror.Pos(cm.lineCount(), 0) };
  }

  // ===== 统一增强入口 =====
  window.bskSetupEditorEnhancements = function (cm, opts) {
    if (!cm) return;
    opts = opts || {};

    // 记录动态数据源（供 hint 函数读取）
    if (opts.tagsFn || opts.artistsFn || opts.categoriesFn) {
      window.BSK_HINT_OPTS = {
        tagsFn: opts.tagsFn,
        artistsFn: opts.artistsFn,
        categoriesFn: opts.categoriesFn
      };
    }

    // hint 列表容器：show-hint 默认把 .CodeMirror-hints 挂到 <body>（z-index 20000），
    // 在视频工作台 z-index 99999 浮层 / 提示词编辑器 99990 浮层下会被盖住
    // （键盘可选但列表不可见）。指定容器后列表挂进容器内部，坐标偏移由 show-hint
    // 在 container !== body 时自动校准。所有 showHint 调用（inputRead 与手动）均生效。
    if (opts.hintContainer) {
      try {
        cm.setOption('hintOptions', Object.assign({}, cm.getOption('hintOptions') || {}, { container: opts.hintContainer }));
      } catch (e) { /* addon 缺失时静默 */ }
    }

    // ---- 基本 option（缺失 addon 时安全忽略）----
    var newOpts = {};
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.matchBrackets) {
      newOpts.matchBrackets = opts.matchBrackets !== false;
    }
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.styleActiveLine) {
      newOpts.styleActiveLine = opts.activeLine !== false;
    }
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.scrollPastEnd) {
      newOpts.scrollPastEnd = true;
    }
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.scrollbarStyle && CodeMirror.scrollbarModel) {
      newOpts.scrollbarStyle = 'simple';
    }
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.highlightSelectionMatches) {
      newOpts.highlightSelectionMatches = { showToken: /[\w\u4e00-\u9fa5'\-]+/, annotateScrollbar: false };
    }
    if (CodeMirror.optionHandlers && CodeMirror.optionHandlers.foldGutter && opts.foldCategories !== false) {
      newOpts.foldGutter = { rangeFinder: categoryRangeFinder };
      newOpts.foldOptions = { rangeFinder: categoryRangeFinder };
    }
    cm.setOption('foldGutter', newOpts.foldGutter);
    cm.setOption('foldOptions', newOpts.foldOptions);
    cm.setOption('matchBrackets', newOpts.matchBrackets);
    cm.setOption('styleActiveLine', newOpts.styleActiveLine);
    cm.setOption('scrollPastEnd', newOpts.scrollPastEnd);
    if (newOpts.scrollbarStyle) cm.setOption('scrollbarStyle', newOpts.scrollbarStyle);
    if (newOpts.highlightSelectionMatches) cm.setOption('highlightSelectionMatches', newOpts.highlightSelectionMatches);

    // ---- extraKeys（与已有 keymap 合并）----
    var prevKeys = cm.getOption('extraKeys') || {};
    var newKeys = {};
    if (CodeMirror.commands && CodeMirror.commands.findPersistent) {
      newKeys['Ctrl-F'] = 'findPersistent';
      newKeys['Cmd-F'] = 'findPersistent';
      newKeys['Ctrl-G'] = 'findNext';
      newKeys['Cmd-G'] = 'findNext';
      newKeys['Shift-Ctrl-G'] = 'findPrev';
      newKeys['Shift-Cmd-G'] = 'findPrev';
      newKeys['Ctrl-Alt-F'] = 'replace';
      newKeys['Cmd-Alt-F'] = 'replace';
    }
    if (CodeMirror.commands && CodeMirror.commands.fold && opts.foldCategories !== false) {
      newKeys['Ctrl-Q'] = function (c) { if (c.somethingSelected()) c.replaceSelection(c.getSelection()); c.foldCode(c.getCursor()); };
      newKeys['Ctrl-Shift-['] = 'foldAll';
      newKeys['Ctrl-Shift-]'] = 'unfoldAll';
    }
    cm.setOption('extraKeys', mergeKeys(prevKeys, newKeys));

    // ---- 自动补全 ----
    if (opts.autoHint !== false && CodeMirror.hint && CodeMirror.hint.bskPrompt && CodeMirror.showHint) {
      cm.on('inputRead', function (c, change) {
        if (c.state.completionActive) return;
        var text = change.text && change.text[0];
        if (!text) return;
        var cursor = c.getCursor();
        var b = (c.getLine(cursor.line) || '').slice(0, cursor.ch);
        var w = (b.match(/[\w\u4e00-\u9fa5@:\.\-\[\]<>]*$/) || [''])[0];
        // 敲空格 / 敲结束标点（, ，、；等）不弹候选，已经弹着的也要收掉：
        // 写完一个标签后的 "， " 往往只是想换行，候选开着会让回车变成
        // 「选候选」而不是换行。想补全接着打字即可，字数一到自然会弹
        // （下面的长度分支）。
        if (/^\s+$/.test(text) || /[,，;；、。！!？?…]/.test(text)) {
          if (c.state.completionActive && c.state.completionActive.close) {
            try { c.state.completionActive.close(); } catch (e) {}
          }
          return;
        }
        var trigger = false;
        if (isH3Mode(c)) {
          // H3：符号输入即触发（覆盖 <d / (S / [Shot 等 2 字符短前缀），普通词 2 字符即触发
          if (/[@:\[(<]/.test(text)) trigger = true;
          else if (w.length >= 2 && !/[<>\@]/.test(w)) trigger = true;
        } else {
          if (/[@:\[(]/.test(text)) trigger = true;
          else if (text === '<') { if (!/^<\/?d/i.test(w)) trigger = true; }
          else if (w.length >= 3 && !/[<>\@]/.test(w)) trigger = true;
        }
        if (trigger) {
          CodeMirror.showHint(c, CodeMirror.hint.bskPrompt, { completeSingle: false, closeOnUnfocus: true });
        }
      });
    }

    // ---- 状态栏 ----
    if (opts.statusBar !== false) {
      attachStatusBar(cm);
    }
  };

  function mergeKeys(a, b) {
    var out = {};
    for (var k in a) if (Object.prototype.hasOwnProperty.call(a, k)) out[k] = a[k];
    for (var k2 in b) if (Object.prototype.hasOwnProperty.call(b, k2)) out[k2] = b[k2];
    return out;
  }

  window.BSK_PROMPT_HINT_VERSION = '1.1.0';
})();
