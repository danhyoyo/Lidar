# PaperNN Fullstack Architecture & Canonical Academic Patterns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform PaperNN Studio into a modular, production-ready fullstack web application featuring a Python FastAPI backend and a React + TypeScript + Vite frontend. Ground every component in peer-reviewed deep learning literature (CVPR, ICCV, NeurIPS, ICLR), provide a 1-click insertable library of canonical academic patterns (ResNet Basic/Bottleneck, MobileNetV2, ConvNeXt, ViT, Swin, DiT, SE, CBAM, FPN, U-Net), and deliver an advanced interactive canvas with magnetic connection ports and customizable orthogonal/dashed arrows.

**Architecture:** Client-server architecture separating UI canvas manipulation from compute services. The frontend uses React 18, TypeScript, TailwindCSS, Lucide icons, and an SVG canvas with interactive anchor ports, customizable wire routing (orthogonal elbows, dashed skips, Bezier curves), and a compound pattern insertion engine. The backend uses Python FastAPI with Pydantic v2 schemas for diagram validation, pattern registry, vector/high-res raster export generation, and optional PyTorch module inspection.

**Tech Stack:** Python 3.10, FastAPI, Pydantic v2, Pytest, React 18, TypeScript, Vite, TailwindCSS, Zustand, Lucide-React, SVG 1.1.

**Spec:** [docs/superpowers/specs/2026-10-12-papernn-fullstack-spec.md](docs/superpowers/specs/2026-10-12-papernn-fullstack-spec.md) / [papernn_fullstack_plan.md](file:///home/duyennh/.gemini/antigravity-cli/brain/32ad8bc9-e267-4bd4-bade-420b6e44b71a/papernn_fullstack_plan.md)

## Global Constraints

- Implement cleanly inside `tools/visualization/papernn_app/` without touching `tools/visualization/nn_svg/` or `papernn_studio/`.
- Every atomic primitive and pattern must strictly adhere to conventions from seminal literature (He et al. 2016, Vaswani et al. 2017, Sandler et al. 2018, Hu et al. 2018, Dosovitskiy et al. 2021, Liu et al. 2022). Zero fabricated or non-standard blocks.
- Frontend must be standalone, bundleable via Vite into static assets, with responsive layout and 60 FPS canvas interactions.
- Backend must use Python 3.10 and FastAPI with RESTful JSON APIs and comprehensive `pytest` test suites.
- All SVG outputs must remain 100% clean vector paths without raster compression artifacts.
- No placeholders, TODOs, or unverified methods in code deliverables.

---

### Task 1: Backend Data Contracts & Schema Models (Pydantic v2)

**Files:**
- Create: `tools/visualization/papernn_app/backend/app/models/schemas.py`
- Create: `tools/visualization/papernn_app/backend/app/models/__init__.py`
- Test: `tools/visualization/papernn_app/backend/tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `NodeLayout`: `x`, `y`, `width`, `height`, `fill`, `stroke`, `font_size`, `rotation`, `shape_style` (`auto`, `card`, `trapezoid`, `slim`, `circle`).
  - `Node`: `id`, `label`, `caption`, `subtitle`, `kind`, `shape`, `repeat`, `layout`.
  - `Edge`: `id`, `source`, `target`, `source_port`, `target_port`, `label`, `kind` (`flow`, `skip`, `lateral`), `routing` (`orthogonal`, `curved`, `straight`), `is_dashed`, `waypoints`.
  - `Group`: `id`, `label`, `node_ids`, `fill`, `stroke`, `padding`.
  - `DiagramView`: `title`, `description`, `nodes`, `edges`, `groups`, `show_legend`.
  - `DiagramDocument`: `version`, `name`, `source`, `views`, `editor`.
  - `PatternTemplate`: `key`, `name`, `category`, `paper_reference`, `description`, `nodes`, `edges`, `groups`.

- [ ] **Step 1: Write the failing test**

Create `tools/visualization/papernn_app/backend/tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from app.models.schemas import DiagramDocument, Node, Edge, NodeLayout, DiagramView, PatternTemplate

def test_node_schema_validation():
    layout = NodeLayout(x=100, y=150, width=160, height=72, shape_style="card")
    node = Node(id="conv1", label="Conv 3x3", kind="conv", shape=[64, 112, 112], layout=layout)
    assert node.id == "conv1"
    assert node.layout.shape_style == "card"
    assert node.shape == [64, 112, 112]

def test_edge_schema_with_dashed_and_routing():
    edge = Edge(
        id="e_residual",
        source="in",
        target="add1",
        source_port="bottom",
        target_port="bottom",
        kind="skip",
        is_dashed=True,
        routing="orthogonal",
        label="Residual skip",
        waypoints=[{"x": 100, "y": 300}, {"x": 400, "y": 300}]
    )
    assert edge.is_dashed is True
    assert edge.routing == "orthogonal"
    assert len(edge.waypoints) == 2

def test_pattern_template_schema():
    template = PatternTemplate(
        key="resnet_basic",
        name="ResNet BasicBlock",
        category="residual",
        paper_reference="He et al., CVPR 2016",
        description="Two 3x3 convs with identity shortcut",
        nodes=[],
        edges=[]
    )
    assert template.paper_reference == "He et al., CVPR 2016"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools/visualization/papernn_app/backend python3 -m pytest tools/visualization/papernn_app/backend/tests/test_schemas.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Write minimal implementation**

Create `tools/visualization/papernn_app/backend/app/models/schemas.py`:
Implement models including `PatternTemplate` with exact type validations.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools/visualization/papernn_app/backend python3 -m pytest tools/visualization/papernn_app/backend/tests/test_schemas.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_app/backend/
git commit -m "feat(papernn_backend): define pydantic schemas including academic pattern models"
```

---

### Task 2: Canonical Academic Patterns Registry & FastAPI Gateway

**Files:**
- Create: `tools/visualization/papernn_app/backend/app/services/patterns.py`
- Create: `tools/visualization/papernn_app/backend/app/api/patterns.py`
- Create: `tools/visualization/papernn_app/backend/app/main.py`
- Test: `tools/visualization/papernn_app/backend/tests/test_patterns.py`

**Interfaces:**
- Produces:
  - `GET /api/v1/patterns`: List all 10 canonical academic patterns with paper citations:
    1. `resnet_basic` (He et al., CVPR 2016)
    2. `resnet_bottleneck` (He et al., CVPR 2016)
    3. `mobilenet_v2` (Sandler et al., CVPR 2018)
    4. `convnext_block` (Liu et al., CVPR 2022)
    5. `mhsa_module` (Vaswani et al., NeurIPS 2017)
    6. `se_block` (Hu et al., CVPR 2018)
    7. `vit_encoder` (Dosovitskiy et al., ICLR 2021)
    8. `foundation_dit_block` (Touvron et al. 2023 / Peebles & Xie 2023)
    9. `fpn_merge` (Lin et al., CVPR 2017)
    10. `unet_skip_block` (Ronneberger et al., MICCAI 2015)
  - `GET /api/v1/patterns/{key}`: Returns pre-wired subgraph with relative coordinates.

- [ ] **Step 1: Write the failing test**

Create `tools/visualization/papernn_app/backend/tests/test_patterns.py`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_canonical_academic_patterns():
    response = client.get("/api/v1/patterns")
    assert response.status_code == 200
    patterns = response.json()
    keys = [p["key"] for p in patterns]
    assert "resnet_basic" in keys
    assert "resnet_bottleneck" in keys
    assert "mobilenet_v2" in keys
    assert "vit_encoder" in keys
    assert "mhsa_module" in keys
    assert "se_block" in keys

def test_get_vit_pattern_structure():
    response = client.get("/api/v1/patterns/vit_encoder")
    assert response.status_code == 200
    vit = response.json()
    assert vit["paper_reference"] == "Dosovitskiy et al., ICLR 2021"
    # Must have 2 residual skip edges with is_dashed = True
    dashed_edges = [e for e in vit["edges"] if e.get("is_dashed") is True]
    assert len(dashed_edges) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools/visualization/papernn_app/backend python3 -m pytest tools/visualization/papernn_app/backend/tests/test_patterns.py -q`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Implement `patterns.py` with the complete registry of 10 canonical patterns and wire them to `/api/v1/patterns`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools/visualization/papernn_app/backend python3 -m pytest tools/visualization/papernn_app/backend/tests/test_patterns.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/visualization/papernn_app/backend/app/services/patterns.py tools/visualization/papernn_app/backend/app/api/patterns.py
git commit -m "feat(papernn_backend): add canonical academic patterns registry with 10 peer-reviewed blocks"
```

---

### Task 3: Vector & High-Res Publication Export Engine

**Files:**
- Create: `tools/visualization/papernn_app/backend/app/services/exporter.py`
- Create: `tools/visualization/papernn_app/backend/app/api/export.py`
- Test: `tools/visualization/papernn_app/backend/tests/test_export.py`

**Interfaces:**
- Produces:
  - `POST /api/v1/export/svg`: Clean vector SVG generation with exact IEEE/CVPR physical millimeter scaling.
  - `POST /api/v1/export/png`: 600 DPI publication rasterization.
  - `POST /api/v1/export/pdf`: Vector PDF with embedded fonts.

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Implement exporter.py**
- [ ] **Step 3: Run test to verify it passes**
- [ ] **Step 4: Commit**

```bash
git add tools/visualization/papernn_app/backend/app/services/exporter.py tools/visualization/papernn_app/backend/app/api/export.py
git commit -m "feat(papernn_backend): implement publication-grade SVG, PDF and 600 DPI PNG exporter"
```

---

### Task 4: Frontend Scaffolding, Theme & Canonical Types

**Files:**
- Create: `tools/visualization/papernn_app/frontend/package.json`
- Create: `tools/visualization/papernn_app/frontend/vite.config.ts`
- Create: `tools/visualization/papernn_app/frontend/src/types/diagram.ts`
- Create: `tools/visualization/papernn_app/frontend/src/index.css`
- Create: `tools/visualization/papernn_app/frontend/src/App.tsx`

**Interfaces:**
- Produces: TypeScript types mirroring backend Pydantic models: `NodeData`, `EdgeData`, `PatternTemplate`, `DiagramState`, `PortPosition`.

- [ ] **Step 1: Create package.json and configuration**
- [ ] **Step 2: Define diagram.ts types**
- [ ] **Step 3: Commit**

```bash
git add tools/visualization/papernn_app/frontend/
git commit -m "feat(papernn_frontend): scaffold React + TypeScript project with academic diagram contracts"
```

---

### Task 5: Diagram Store Engine with 1-Click Pattern Inserter & Undo/Redo

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/store/diagramStore.ts`
- Test: `tools/visualization/papernn_app/frontend/src/store/diagramStore.test.ts`

**Interfaces:**
- Produces:
  - `useDiagramStore`: `nodes`, `edges`, `groups`, `selection`, `mode` (`select`, `connect`, `pan`).
  - Actions:
    - `insertPattern(templateKey, targetPos)`: Translates subgraph coordinates and inserts pre-wired pattern atomically into canvas.
    - `addNode`, `updateNode`, `removeNode`, `addEdge`, `updateEdge`, `removeEdge`.
    - `alignSelected`, `distributeSelected`, `undo`, `redo`.

- [ ] **Step 1: Write store unit test for pattern insertion**
- [ ] **Step 2: Implement diagramStore with Zustand**
- [ ] **Step 3: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/store/
git commit -m "feat(papernn_frontend): implement diagram store with compound pattern insertion and undo/redo"
```

---

### Task 6: Academic Paper Canvas with 85mm / 182mm Boundary Rulers

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/Canvas.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/PaperGuides.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/GridBackground.tsx`

**Interfaces:**
- Produces:
  - Pan (Middle-Mouse / Space+Drag) & Zoom (25% to 500% centered at cursor).
  - Exact IEEE/CVPR boundaries: Single column (**85 mm**), Double column (**182 mm**).
  - Millimeter dimensions readout badge.
  - Smart grid with 8px / 16px magnetic snap.

- [ ] **Step 1: Implement Canvas, PaperGuides, and GridBackground**
- [ ] **Step 2: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Canvas/
git commit -m "feat(papernn_frontend): create paper-calibrated canvas with pan, zoom and publication column guides"
```

---

### Task 7: Node Rendering Engine with Shape Overrides & Stacked Shadows

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/NodeRenderer.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/NodeVisuals.tsx`

**Interfaces:**
- Produces:
  - Standard card for Conv / Linear / Attention.
  - Slim vertical pill (`node-slim`, 44×76 px) for Norms & Activations.
  - Polygon trapezoid for Pooling / Downsampling ($\blacktriangleright$) and Upsampling ($\blacktriangleleft$).
  - Mathematical circles for Math Ops ($\oplus, \odot, \otimes, \copyright, \sigma$).
  - Offset stacked shadow cards for blocks with `repeat > 1`.
  - Shape style override (`card`, `trapezoid`, `slim`, `circle`, `auto`).

- [ ] **Step 1: Implement NodeRenderer and shape components**
- [ ] **Step 2: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Canvas/NodeRenderer.tsx tools/visualization/papernn_app/frontend/src/components/Canvas/NodeVisuals.tsx
git commit -m "feat(papernn_frontend): implement node rendering with shape overrides and stacked shadows"
```

---

### Task 8: Interactive Magnetic Ports & Live Drag-to-Connect Wires

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/PortHandle.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/WireDraft.tsx`
- Modify: `tools/visualization/papernn_app/frontend/src/components/Canvas/NodeRenderer.tsx`

**Interfaces:**
- Produces:
  - 4 magnetic anchor ports (Top, Right, Bottom, Left) on hover.
  - Dragging from a port creates a live rubber-band wire preview (`WireDraft`).
  - Snaps into target ports automatically when released.

- [ ] **Step 1: Implement PortHandle and connection event handlers**
- [ ] **Step 2: Integrate wire creation into diagramStore**
- [ ] **Step 3: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Canvas/PortHandle.tsx tools/visualization/papernn_app/frontend/src/components/Canvas/WireDraft.tsx
git commit -m "feat(papernn_frontend): implement magnetic anchor ports and drag-to-connect wire workflow"
```

---

### Task 9: Orthogonal Wire Router, Dashed Residual Lines & Waypoint Editing

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Canvas/EdgeRenderer.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/utils/wireRouting.ts`
- Create: `tools/visualization/papernn_app/frontend/src/components/Inspector/EdgeInspector.tsx`

**Interfaces:**
- Produces:
  - Orthogonal 90° elbow path calculation avoiding intermediate blocks.
  - 1-click toggle `is_dashed` for residual skips (`stroke-dasharray="6 4"`).
  - Click-to-add elbow waypoints and drag waypoints to adjust route.
  - Standard arrowhead markers (`marker-end`).
  - Text annotations positioned along the connection line.

- [ ] **Step 1: Implement wireRouting algorithms**
- [ ] **Step 2: Implement EdgeRenderer and EdgeInspector**
- [ ] **Step 3: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Canvas/EdgeRenderer.tsx tools/visualization/papernn_app/frontend/src/utils/wireRouting.ts tools/visualization/papernn_app/frontend/src/components/Inspector/EdgeInspector.tsx
git commit -m "feat(papernn_frontend): add orthogonal router, dashed residual lines, waypoints and arrow styling"
```

---

### Task 10: Canonical Academic Pattern Drawer & Component Palette

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Palette/PatternPalette.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Palette/AtomicPalette.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Palette/ComponentPalette.tsx`

**Interfaces:**
- Produces:
  - Tab 1: **Mẫu khối kinh điển (Academic Patterns)**: 10 mẫu (ResNet Basic/Bottleneck, MobileNetV2, ConvNeXt, ViT, Swin, DiT, SE, CBAM, FPN, U-Net) với chú thích trích dẫn tác giả và năm.
  - Tab 2: **Khối nguyên tử (Atomic Primitives)**: 7 nhóm LEGO cơ bản.
  - Click vào mẫu $\to$ Tự động sinh cả cụm mạng con hoàn chỉnh trên canvas.

- [ ] **Step 1: Implement PatternPalette and AtomicPalette**
- [ ] **Step 2: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Palette/
git commit -m "feat(papernn_frontend): implement canonical academic pattern drawer with 1-click insertion"
```

---

### Task 11: Inspector Sidebar, Alignment Toolbar & End-to-End Integration

**Files:**
- Create: `tools/visualization/papernn_app/frontend/src/components/Inspector/NodeInspector.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Inspector/MultiInspector.tsx`
- Create: `tools/visualization/papernn_app/frontend/src/components/Toolbar/AppToolbar.tsx`
- Create: `tools/visualization/papernn_app/README.md`
- Test: All backend tests (`pytest`) and frontend production build (`npm run build`).

**Interfaces:**
- Produces:
  - Inspector for node dimensions, shape overrides, repeat count, colors.
  - Multi-selection alignment & distribution toolbar.
  - Full client-server integration with FastAPI export endpoints.

- [ ] **Step 1: Implement inspectors and toolbar**
- [ ] **Step 2: Run backend tests and frontend build**
- [ ] **Step 3: Write comprehensive user documentation**
- [ ] **Step 4: Commit**

```bash
git add tools/visualization/papernn_app/frontend/src/components/Inspector/ tools/visualization/papernn_app/frontend/src/components/Toolbar/ tools/visualization/papernn_app/README.md
git commit -m "feat(papernn_app): complete fullstack integration with inspectors, alignment tools and documentation"
```
