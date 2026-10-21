/**
 * BSK_Hub 중계 노드 - 동적 입력/출력 확장
 *
 * 기능: 연결 상태에 따라 자동으로 추가/삭제 (inN + outN) End point pair
 * - 기본 1 그룹 in1 + out1（직통：out1 = in1）
 * - 어떤 inN 또는 outN이 연결되면 끝에 자동으로 추가 in(N+1) + out(N+1)
 * - 최소 1개의 연결되지 않은 빈 엔드포인트를 유지하여 계속 연결
 * - 엔드포인트 기본 "*"（어떤 유형이든), 연결 후 실제 데이터 유형을 표시(reroute 스타일)）
 *
 * 기존 패턴 재사용 js/string_merge_dynamic.js（bsk_StringMergeWithSwitch）。
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "BSK_Hub";
const MAX_PAIRS = 16;

function isInName(name) { return name && /^in\d+$/.test(name); }
function isOutName(name) { return name && /^out\d+$/.test(name); }

/**
 * 재계산 노드 동적 종단점：
 * 1. 현재 수집 in*/out* 슬롯 및 연결 상태
 * 2. 마지막 엔드포인트 쌍이 연결되어 있고 한계에 도달하지 않은 경우 마지막에 새 쌍을 추가합니다
 * 3. 마지막에 연결되지 않은 엔드포인트를 삭제하지만, 최소 1개의 빈 엔드포인트는 항상 유지합니다
 * 4. 연결된 엔드포인트에서 실제 데이터 타입 표시 (직통: outN 타입) = inN 유형）
 */
function updateHubSockets(node) {
    if (node.type !== NODE_TYPE) return;
    if (node.__hubUpdating) return;
    node.__hubUpdating = true;
    try {
        // ---- 수집 현재 엔드포인트 쌍 ----
        const pairs = [];
        for (let n = 1; n <= MAX_PAIRS; n++) {
            let inIdx = -1, outIdx = -1;
            for (let i = 0; i < node.inputs.length; i++) {
                if (isInName(node.inputs[i] && node.inputs[i].name)) { inIdx = i; break; }
            }
            for (let i = 0; i < node.outputs.length; i++) {
                if (isOutName(node.outputs[i] && node.outputs[i].name)) { outIdx = i; break; }
            }
            if (inIdx === -1 && outIdx === -1) continue;
            pairs.push({
                n, inIdx, outIdx,
                inConnected: inIdx >= 0 && node.inputs[inIdx].link != null,
                outConnected: outIdx >= 0 && node.outputs[outIdx].link != null,
            });
        }

        const maxN = pairs.length ? Math.max(...pairs.map(p => p.n)) : 0;
        const lastPair = pairs.length ? pairs[pairs.length - 1] : null;
        const lastConnected = lastPair ? (lastPair.inConnected || lastPair.outConnected) : false;

        // ---- 2. 새 관계 추가 ----
        if (lastConnected && maxN < MAX_PAIRS) {
            const next = maxN + 1;
            node.addInput(`in${next}`, "*");
            node.addOutput(`out${next}`, "*");
        }

        // ---- 3. 마무리 부분에 연결되지 않은 그룹을 회수 (최소 1 그룹의 빈 엔드포인트를 유지)）----
        let removed = true;
        while (removed) {
            removed = false;
            const cur = [];
            for (let n = 1; n <= MAX_PAIRS; n++) {
                let inIdx = -1, outIdx = -1;
                for (let i = 0; i < node.inputs.length; i++) {
                    if (isInName(node.inputs[i] && node.inputs[i].name)) { inIdx = i; break; }
                }
                for (let i = 0; i < node.outputs.length; i++) {
                    if (isOutName(node.outputs[i] && node.outputs[i].name)) { outIdx = i; break; }
                }
                if (inIdx === -1 && outIdx === -1) continue;
                cur.push({
                    n, inIdx, outIdx,
                    inConnected: inIdx >= 0 && node.inputs[inIdx].link != null,
                    outConnected: outIdx >= 0 && node.outputs[outIdx].link != null,
                });
            }
            if (cur.length <= 1) break; // 최소 1 그룹 유지
            const tail = cur[cur.length - 1];
            if (!tail.inConnected && !tail.outConnected) {
                // 먼저 큰 인덱스를 삭제하여 오프셋을 방지하고, 배열의 실제 소유에 따라 삭제합니다.
                const idxs = [tail.inIdx, tail.outIdx].filter(i => i >= 0).sort((a, b) => b - a);
                for (const idx of idxs) {
                    if (isInName(node.inputs[idx] && node.inputs[idx].name)) node.removeInput(idx);
                    else if (isOutName(node.outputs[idx] && node.outputs[idx].name)) node.removeOutput(idx);
                }
                removed = true;
            }
        }

        // ---- 4. 연결된 엔드포인트의 실제 데이터 타입 표시 ----
        for (let n = 1; n <= MAX_PAIRS; n++) {
            let inIdx = -1, outIdx = -1;
            for (let i = 0; i < node.inputs.length; i++) {
                if (isInName(node.inputs[i] && node.inputs[i].name)) { inIdx = i; break; }
            }
            for (let i = 0; i < node.outputs.length; i++) {
                if (isOutName(node.outputs[i] && node.outputs[i].name)) { outIdx = i; break; }
            }
            if (inIdx === -1 && outIdx === -1) continue;

            // inN 유형: 상류 연결선에서 가져옴
            let inType = "*";
            if (inIdx >= 0 && node.inputs[inIdx].link != null && node.graph && node.graph.links) {
                const link = node.graph.links[node.inputs[inIdx].link];
                if (link) {
                    const src = node.graph.getNodeById(link.origin_id);
                    if (src && src.outputs && src.outputs[link.origin_slot]) {
                        inType = src.outputs[link.origin_slot].type || "*";
                    }
                }
            }
            // outN 자료 썸네일 (whatdreamscost 하위 디렉토리 지원)）
            let outType = inType;
            if (outIdx >= 0 && node.outputs[outIdx].link != null && node.graph && node.graph.links) {
                const link = node.graph.links[node.outputs[outIdx].link];
                if (link) {
                    const dst = node.graph.getNodeById(link.target_id);
                    if (dst && dst.inputs && dst.inputs[link.target_slot]) {
                        const t = dst.inputs[link.target_slot].type;
                        if (t && t !== "*") outType = t;
                    }
                }
            }

            if (inIdx >= 0) node.inputs[inIdx].type = inType;
            if (outIdx >= 0) node.outputs[outIdx].type = outType;
        }

        if (node.setSize) node.setSize(node.computeSize());
    } finally {
        node.__hubUpdating = false;
    }
}

app.registerExtension({
    name: "BSK.Hub.DynamicSockets",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== NODE_TYPE) return;

        // 초기화: INPUT_TYPES/RETURN_TYPES에 따라 자동으로 생성된 엔드포인트를 제거하고, 1 그룹만 추가
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            const inIdxs = [];
            const outIdxs = [];
            for (let i = 0; i < this.inputs.length; i++) {
                if (isInName(this.inputs[i] && this.inputs[i].name)) inIdxs.push(i);
            }
            for (let i = 0; i < this.outputs.length; i++) {
                if (isOutName(this.outputs[i] && this.outputs[i].name)) outIdxs.push(i);
            }
            // 후방에서 전방으로 삭제하여 인덱스 오프셋 방지
            inIdxs.sort((a, b) => b - a).forEach(i => this.removeInput(i));
            outIdxs.sort((a, b) => b - a).forEach(i => this.removeOutput(i));

            this.addInput("in1", "*");
            this.addOutput("out1", "*");

            if (this.setSize) this.setSize(this.computeSize());
            return r;
        };

        // 노드 연결 변경: 한쪽 (입력/출력) 측의 변경 시 엔드포인트 재계산
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side, slot, connected, linkInfo, output) {
            const r = origOnConnectionsChange
                ? origOnConnectionsChange.apply(this, arguments)
                : undefined;
            updateHubSockets(this);
            return r;
        };
    },
});
