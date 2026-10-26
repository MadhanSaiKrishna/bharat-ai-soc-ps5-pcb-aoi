# AI Agent Multi-Developer Collaboration Protocol

This document defines the rules of engagement for all **3 team members and their AI coding assistants** working on this repository to eliminate git conflicts, prevent broken dependencies, and ensure zero-friction integration.

---

## 1. The Core Principle: Contract-First Architecture

The file [`schemas/inspection_event.json`](file:///home/madhan/bharat-ai-soc-ps5-pcb-aoi/schemas/inspection_event.json) is the **single source of truth** across all three subsystems:
- **Member 1 (Vision AI):** Emits defect arrays conforming to this schema.
- **Member 2 (Edge SoC & LLM):** Ingests this schema to generate DFM and Assembler advisories.
- **Member 3 (Homography & UI):** Visualizes this schema on the live PyQt6 CAD overlay.

> [!CAUTION]
> **Schema Lock Rule:**  
> **NO AI agent is permitted to alter `schemas/inspection_event.json` or `configs/` without explicit agreement from all 3 members.** If a field needs to change, all 3 developers must agree on the new spec first.

---

## 2. Strict Folder Ownership ("Territories")

To prevent multiple AI agents from overwriting the same files, each member and their AI agent has an exclusive directory boundary:

| Member & Role | Allowed Folder(s) for AI Edits | Forbidden Folders |
| :--- | :--- | :--- |
| **Member 1: Vision AI & Datasets** | `module_a_vision/`<br>`data/synthetic_generator/`<br>`data/sample_gerbers/` | `module_b_guidance/`<br>`edge_llm_storage/` |
| **Member 2: Edge SoC, LLM & Storage** | `edge_llm_storage/`<br>`data/telemetry_logs/` | `module_a_vision/`<br>`module_b_guidance/` |
| **Member 3: Gerber, UI & Homography** | `module_b_guidance/`<br>`module_b_guidance/app_ui.py` | `module_a_vision/`<br>`edge_llm_storage/` |

---

## 3. Git Branching Strategy

1. **`main` / `master` is Protected:**
   - No direct pushes to `main`.
2. **Dedicated Member Feature Branches:**
   - Member 1: `git checkout -b feat/m1-vision-obb`
   - Member 2: `git checkout -b feat/m2-edge-llama`
   - Member 3: `git checkout -b feat/m3-cad-ui`
3. **Merging Rule:**
   - Always run the contract smoke test before opening a Pull Request:
     ```bash
     pytest tests/test_contract.py
     ```
   - If tests pass, open PR & merge into `main` using **Squash & Merge** or standard merge without rebase messes.
   - Other members immediately sync:
     ```bash
     git checkout main && git pull origin main
     git checkout feat/<your-branch> && git merge main
     ```

---

## 4. Decoupled Development & Mock-First Testing

**No team member or AI agent should ever be blocked waiting for another member:**
- **Member 3 (UI)** does NOT wait for Member 1's real YOLOv8 weights to be trained; the UI loads [`schemas/sample_inspection_event.json`](file:///home/madhan/bharat-ai-soc-ps5-pcb-aoi/schemas/sample_inspection_event.json) as a mock data stream.
- **Member 3 (UI)** does NOT wait for Member 2's ExecuTorch `.pte` model; [`edge_llm_storage/dual_advisor.py`](file:///home/madhan/bharat-ai-soc-ps5-pcb-aoi/edge_llm_storage/dual_advisor.py) includes a built-in deterministic heuristic fallback that runs instantly on any laptop without needing the Arm compiler or model weights.
- **Member 1 (Vision)** can test standalone using generated synthetic frames from [`synth_defects.py`](file:///home/madhan/bharat-ai-soc-ps5-pcb-aoi/data/synthetic_generator/synth_defects.py).

---

## 5. Copy-Paste Prompt Preamble for Each Member's AI Agent

Whenever you or your team members start an AI coding session (e.g., in Claude, Cursor, ChatGPT, Antigravity, or Copilot), **paste this preamble at the beginning of the prompt**:

```markdown
You are assisting on the "Bharat AI-SoC PS5 PCB AOI" repository.
Read and adhere strictly to AGENTS.md and schemas/inspection_event.json.

YOUR CONSTRAINTS:
1. You may ONLY create or edit files in: <INSERT YOUR ASSIGNED FOLDER HERE>
2. You must NEVER modify schemas/inspection_event.json or configs/ unless explicitly commanded.
3. Any output, function, or CLI tool you create must adhere to the data types in schemas/inspection_event.json.
4. Ensure pytest tests/test_contract.py passes after your changes.
```
