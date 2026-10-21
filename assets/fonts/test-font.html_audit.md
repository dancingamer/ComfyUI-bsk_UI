# test-font.html 代码审查清单

## 文件信息
- **路径**: Q:\AI\ComfyUI-bsk_UI\ComfyUI-bsk_UI\release\assets\fonts\test-font.html
- **类型**: HTML
- **行数**: 129

## 问题清单

### 严重 (High)
| # | 位置 | 问题描述 | 建议修复 | 优先级 |
|---|------|----------|----------|--------|
| 1 | line 14 | 本地字体文件路径问题：`@font-face` 引用 `'PlayfairDisplay-Black.ttf'`（line 14）使用相对 URL。此文件存在于 `assets/fonts/` 目录中，从正确目录打开时可以加载。但如果通过 `file://` 从父文件夹双击打开，会静默 404。 | 使用从文档根目录开始的绝对路径或添加注释说明此测试的正确打开方式。 | P1 |
| 2 | line 30 | 断裂的 ComfyUI 路径引用：`@font-face` line 30 引用 `url('/custom_nodes/ComfyUI_bsk_UI/assets/fonts/PlayfairDisplay-Black.ttf')`。前导 `/` 是从 Web 服务器根目录的绝对路径。仅在 ComfyUI 服务器上以该精确路径提供服务时才有效。通过 `file://` 打开会产生 `file:///custom_nodes/...` 的 404。 | 使用前缀为此 HTML 文件相对 ComfyUI 节点的完整路径，或文档说明此测试仅在 ComfyUI 内工作。 | P1 |

### 中等 (Medium)
| # | 位置 | 问题描述 | 建议修复 | 优先级 |
|---|------|----------|----------|--------|
| 1 | 全局 | 缺少 Content-Security-Policy。 | 添加 CSP meta 标签。 | P2 |
| 2 | line 34 | 使用 `@import` 而非 `<link>`：`<style>` 块内 line 34 的 `@import url(...)` 阻塞字体的并行下载。应使用 `<link rel="preload">` 或 `<link>` 标签。 | 将 Google Fonts 移至 `<head>` 中的 `<link>` 标签。 | P2 |

### 低优 (Low)
| # | 位置 | 问题描述 | 建议修复 | 优先级 |
|---|------|----------|----------|--------|
| 1 | line 95-96 | `<h1>` 无 `aria-label` 或可访问描述。页面标题在 `<h1>` 和 `<title>` 中重复。虽语义正确但可改进。 | 添加 `<h1 role="heading" aria-level="1">` 或保持现状。 | P3 |

## 总结
- 严重问题: 2 个
- 中等问题: 2 个
- 低优问题: 1 个
- 总计: 5 个
