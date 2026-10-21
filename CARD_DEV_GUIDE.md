# bsk_UI 自定义卡片开发指南

## 目录结构

```
ComfyUI-bsk_UI/
├── __init__.py           # 主入口（已自动加载 nodes/ 目录）
├── nodes/                # ← 放后端节点定义
│   ├── hello_world_node.py
│   └── your_node.py      # 你的新节点
└── js/
    ├── panel.js          # 主面板（已支持卡片注册）
    ├── rgba_image_editor.js
    └── cards/            # ← 放前端卡片渲染器
        ├── hello_world_card.js
        └── your_card.js  # 你的新卡片
```

## 添加一个新功能只需两步

### 第一步：创建后端节点

在 `nodes/` 下新建一个 `.py` 文件，例如 `my_greeting_node.py`：

```python
class MyGreetingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "世界"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("问候语",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/测试"

    def execute(self, name):
        return (f"你好, {name}!",)

# 以下两行是必须的注册信息（格式固定）
NODE_CLASS_MAPPINGS = {"MyGreetingNode": MyGreetingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyGreetingNode": "BSK 问候"}
```

> 注意：不需要手动修改 `__init__.py`，系统会自动发现 `nodes/` 目录下的所有 `.py` 文件。

### 第二步：创建前端卡片

在 `js/cards/` 下新建一个 `.js` 文件，例如 `my_greeting_card.js`：

```javascript
/**
 * MyGreetingNode 卡片渲染器
 */

(function() {
  'use strict';

  // 节点类型名称（必须与后端 NODE_CLASS_MAPPINGS 的 key 一致）
  var NODE_TYPE = 'MyGreetingNode';

  // 卡片渲染函数
  function renderCard(container, nodeData, api) {
    // container : 卡片的 body 容器，往里面加 DOM
    // nodeData  : 节点数据（{id, classType, title, inputs, ...}）
    // api       : 面板工具
    //   api.getValue(key)   - 读取节点输入的值
    //   api.setValue(key, v) - 设置节点输入的值
    //   api.baseUrl          - ComfyUI 服务器地址

    // --- 示例：简单问候卡片 ---
    var savedName = api.getValue('name') || '世界';

    var display = document.createElement('div');
    display.style.cssText = 'padding:20px;text-align:center;font-size:20px;';
    display.textContent = '你好, ' + savedName + '!';
    container.appendChild(display);

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-input';
    input.value = savedName;
    input.oninput = function(e) {
      display.textContent = '你好, ' + e.target.value + '!';
      api.setValue('name', e.target.value);  // 保存到面板
    };
    container.appendChild(input);
  }

  // 注册到全局卡片注册表（必须）
  window.__bskCards = window.__bskCards || {};
  window.__bskCards[NODE_TYPE] = renderCard;
})();
```

### 完成！重启 ComfyUI 即可看到效果

---

## 动态输入 Autogrow 支持（高级）

> 若你只需要给某个节点加几个固定输入，跳过本节即可；**本节面向"端口数量不固定、可连线增长"的动态输入**。

### 这是什么

新版 ComfyUI（`comfy_api` / `io.ComfyNode` 新式节点体系）提供 `Autogrow` 动态输入：一个输入可展开成**任意数量的槽位端口**（如 `images.image_1`、`images.image_2` …）。前端连线后自动追加空端口，断开后自动收缩，后端节点收到一个 dict。

**bsk_UI 工作流编辑器（`js/workflow_editor.js`）已完整支持该机制**，后端节点按下方方式定义即可，前端无需再改代码。注意：**旧式 `INPUT_TYPES` 节点无法使用 Autogrow**，必须用新式 `io.ComfyNode` / `io.Schema` 写法。

### 后端如何定义（新式 comfy_api）

```python
# nodes/my_autogrow_node.py
import comfy_api.nodes as io

class MyAutogrowNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # 模板一：前缀递增槽位 image_1, image_2, ... image_8
        prefix_tpl = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),
            prefix="image",
            min=1,
            max=8,
        )
        # 模板二：固定命名槽位 a, b, c（names 模式）
        names_tpl = io.Autogrow.TemplateNames(
            input=io.Float.Input("float"),
            names=["a", "b", "c"],
            min=1,
        )
        return io.Schema(
            node_id="MyAutogrowNode",
            display_name="BSK Autogrow 示例",
            category="bsk_UI/测试",
            inputs=[
                io.Autogrow.Input("images", template=prefix_tpl),
                io.Autogrow.Input("floats", template=names_tpl),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, floats: io.Autogrow.Type) -> io.NodeOutput:
        # images = {"image_1": ..., "image_2": ...}（dict，key 为槽位名，不含输入 id）
        print(len(images))  # 实际连线数
        return io.NodeOutput(images.get("image_1"))
```

关键点：
- 槽位名 = `TemplatePrefix.prefix + 序号`（序号从 1 开始）或 `TemplateNames.names` 数组里的名字
- `execute` 收到的参数是 **dict**（`io.Autogrow.Type`），key 即槽位名，**不含** `Autogrow.Input` 的 id（如 `images`）
- 模板的 `input` 类型决定槽位端口类型；一个节点可挂多个 Autogrow 输入，也可与静态输入混用
- 展开/重组由框架完成，后端无需额外处理 `dynamic_paths`

### object_info 序列化格式（前端识别的依据）

Autogrow 输入在 `/object_info` 中序列化为：

```json
["COMFY_AUTOGROW_V3", {
  "template": {
    "input": { "required": { "image": ["IMAGE", {}] } },
    "prefix": "image",
    "min": 1,
    "max": 8
  }
}]
```

- **前缀模式**：带 `prefix` / `min` / `max`
- **names 模式**：带 `names`（数组）/ `min`，无 `prefix` / `max`
- 判断描述符：`isAutogrowDef(def)` ⇔ `def[0] === 'COMFY_AUTOGROW_V3' && def[1] && def[1].template`

### 前端提交格式（写 workflow API payload 时）

- 槽位用**点号展开键**：`"images.image_1": [...]`、`"images.image_2": [...]`
- 后端通过 `dynamic_paths` 自动重组为嵌套字典 `{"images": {"image_1": ..., "image_2": ...}}` 再传给 `execute`
- **空槽位（未连线）不要提交**；bsk_UI 序列化时只收集值非空的槽位键

### bsk_UI 前端已实现的支持（js/workflow_editor.js）

所有 Autogrow 逻辑集中在 `workflow_editor.js` 约 line 350-475 的辅助函数区：

| 函数 | 作用 |
|---|---|
| `isAutogrowDef(def)` | 判断输入定义是否为 Autogrow 描述符 |
| `getAutogrowMeta(def)` | 解析出 `{prefix, names, min, max, tplDef, typeName}` |
| `getAutogrowInputs(nodeDef)` | 收集节点所有 Autogrow 输入（required + optional） |
| `getAutogrowFlowInputs(node, nodeDef, outputTypes)` | 计算当前应渲染的槽位端口列表（含自动增长逻辑） |
| `getNodeFlowInputs(node, nodeDef, outputTypes)` | 统一入口：静态 flow 输入 + 动态输入节点 + Autogrow 槽位 |
| `isAutogrowSlotName(nodeDef, inputName)` | 判断 `base.slot` 是否为某节点的 Autogrow 槽位键 |
| `collectAutogrowInputs(node, nodeDef, inputs)` | 序列化时把已有的非空槽位键并入目标 inputs |

**槽位端口命名约定**：渲染端口名 = `base.slot`（如 `images.image_1`）；UI 标签只显示点号后部分（`image_1`）。

**自动增长/收缩规则**（`getAutogrowFlowInputs`）：
- 槽位数 = `max(min, 已连线数)`；若最后一个槽位已连线且未达 `max`，追加 1 个空槽位 → 连续连线会不断增长
- 断开槽位连线 → 删除 `node.inputs[base.slot]` → 重渲染收缩

**已接入的关键路径（改动前端时务必保持以下约定）**：
1. **渲染**：`inputList` 先用 `!isAutogrowDef(d)` 过滤掉描述符，再合并 `getAutogrowFlowInputs` 输出作为 flow 端口
2. **统一入口**：所有"取节点输入列表"的地方必须走 `getNodeFlowInputs(node, def, this.outputTypes)`（高亮、贝塞尔连线、节点高度、连接匹配、面板同步等），**不要**直接 `inputList.filter(...)`，否则会漏掉 Autogrow 槽位
3. **序列化**（3 处：`_buildApiWorkflow`、`exportWorkflow`、发送面板）：遍历静态输入后调用 `collectAutogrowInputs(node, def, inputs)`
4. **节点创建** `_addNode`：`isAutogrowDef(iDef)` 时 `continue`（不生成假静态输入）
5. **加载/粘贴恢复**：对 `isAutogrowSlotName(def, name)` 且值非空的键写入 `node.inputs`
6. **面板同步** `_syncPanelConfigToWorkflow`：收集 Autogrow 槽位连线

### 如何新增一个带 Autogrow 的节点（无需改前端）

1. 按上文"后端如何定义"写新式节点，放到 `nodes/` 下，**重启 ComfyUI**
2. 前端无需改动：工作流编辑器自动识别 `COMFY_AUTOGROW_V3` 描述符并渲染槽位端口
3. 从模块列表拖入节点，连线第一个端口后会自动出现下一个空端口

> 排查提示：槽位端口不显示时，先查 `/object_info` 里该输入是否序列化为 `["COMFY_AUTOGROW_V3", {...}]`；若仍是普通静态定义则不是 Autogrow 节点。

---

## 卡片渲染器 API 参考

渲染器函数签名：`renderCard(container, nodeData, api)`

### container (DOM 元素)
卡片的 body 容器，所有 UI 元素都加到这里。

> ⚠️ **禁止对 `container` 设置内联 `display` 样式！**
> `container` 就是面板里的 `.config-card-body`，标题栏 `▼` 折叠依赖 `.config-card.collapsed .config-card-body { display:none }`。
> 若你写了 `container.style.display = 'flex'` 或 `container.style.cssText = '...;display:flex;...'`，
> **内联样式优先级高于 class 选择器**，会压制折叠规则，导致点 `▼` 时 `collapsed` 类已加上但内容仍显示（实测 `body.display` 仍是 `flex` 而非 `none`）。
> 正确做法：把布局样式放到一个内部 wrapper 上，再把 wrapper 挂到 `container`：
> ```javascript
> var root = document.createElement('div');
> root.style.cssText = 'display:flex;flex-direction:column;gap:12px;padding:12px;';
> container.appendChild(root);
> // 之后所有 UI 元素都 append 到 root，而非 container
> ```

### nodeData (对象)
```javascript
{
  id: 3,            // 节点在工作流中的 ID
  classType: 'MyNode',  // 节点类型名称
  title: '节点标题',
  inputs: [         // 节点输入列表
    { key: 'name', label: '名称', value: '...' }
  ]
}
```

### api (面板工具)
|| 方法 | 说明 |
|---|---|
| `api.getValue('field_name')` | 读取某个输入字段的值 |
| `api.setValue('field_name', value)` | 设置某个输入字段的值，面板会自动保存 |
| `api.baseUrl` | ComfyUI 服务器地址 |
| `api.nodeId` | 当前节点 ID |
| `api.cardValues` | 所有卡片值的引用（只读场景用） |
| `api.execute()` | 触发面板执行工作流（提交到队列） |
| `api.showToast(msg)` | 显示一条提示消息 |

---

## 注意事项

1. **节点类型名必须一致**：后端 `NODE_CLASS_MAPPINGS` 的 key 必须与前端 JS 中的 `NODE_TYPE` 完全一致
2. **卡片 JS 放在 `js/cards/`** 下，嵌入模式和独立模式都会自动加载：
   - **嵌入模式**（ComfyUI 原生界面）：通过 `WEB_DIRECTORY = "js"` 自动加载 `js/` 下所有 `.js` 文件
   - **独立模式**（`/x/` 页面）：服务端会自动扫描 `js/cards/` 目录并注入脚本，**无需手动修改 `comfyui_panel.html`**
3. **卡片不需要 `__init__.py`**：前端 `js/cards/*.js` 由 `/x/` 服务端自动扫描注入；嵌入模式下 `js/` 下所有 `.js` 被 ComfyUI 自动加载。后端 `nodes/*.py` 也无需改 `__init__.py`（系统自动发现 `nodes/` 目录），但**新增/修改后端 `.py` 必须重启 ComfyUI** 才会重新加载
4. **值持久化**：通过 `api.setValue()` 设置的值会自动保存，无需额外存储
5. **生效方式区分**：
   - 后端 `nodes/*.py` 改动 → **必须重启 ComfyUI** 才重新加载
   - 前端 `js/cards/*.js` 改动 → **无需重启**，刷新页面即可（经隧道/CDN 部署疑似缓存旧代码时，用 Ctrl+Shift+R 硬刷新；仍无效见下方「常见陷阱」缓存问题）

---

## UI 文本尺寸规范（强制）

- **所有卡片内可见文本（标签、下拉项、输入框、滑块数值、开关文字、说明文案 `note` 等）字号不得小于 `14px`。**
- 禁止为了“省空间”把文字压到 12px / 11px 等小字号；空间不足优先用折叠、滚动或换行，而不是缩小文字。
- 建议用一个常量统一控制（如 `var FONT = '14px'`），避免散落硬编码；层级区分可往大调（15px / 16px），不要往小调。
- 说明性 `note` 文案同样适用 14px 下限：可用降低不透明度来弱化视觉权重，不要缩小字号。
- 内联 `innerHTML` 里的 `<code>` 等元素若被全局样式压小，需显式设置 `font-size` 不低于 14px。

---

## 常见陷阱与排查

### 1. 标题栏 `▼` 折叠失效（卡片内容不隐藏）
见上方 `container` API 说明：**禁止在 `container`（即 `.config-card-body`）上设内联 `display`**。内联样式优先级高于面板折叠规则，会导致 `collapsed` 类加上、内容却仍显示。布局样式一律放到内部 wrapper（见该处示例）。

### 2. 改了 `js/cards/*.js` 页面却跑旧代码
`/x/` 路由注入卡片脚本时路径无版本参数，经 cnb.run 隧道/CDN 时会被服务端层缓存（非浏览器磁盘缓存，无痕窗口 + 重启 ComfyUI 都清不掉）。
- **诊断**：Console 执行 `fetch('./js/cards/你的卡片.js').then(r=>r.text())` dump 实际加载源码，对比磁盘文件确认是否缓存。
- **解决**：先硬刷新（Ctrl+Shift+R）；根本修复是在 `__init__.py` 注入处给路径加 `?t=<文件mtime>` 版本参数（加载逻辑已兼容 query string）。

### 3. 卡片设的值没传到后端节点
确认用 `api.setValue('key', v)` 写入（面板侧 `cardValues` 存储），`execute` 时面板通用循环会自动把所有 `cardValues` 注入 `inputs[key]`。注意：
- `key` 必须与后端 `INPUT_TYPES` 的字段名**完全一致**；
- 后端节点需含 `NODE_CLASS_MAPPINGS` 被自动发现（无需改 `__init__.py`）。
