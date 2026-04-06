# Multi-Agent Report Generation System: Revised Architecture v2.0

## 1. System Overview

The system comprises two complementary pipelines:

- **Pipeline A (Ingestion):** Transforms source documents into vector-indexed chunks in Milvus. Runs once per project.
- **Pipeline B (Generation):** Takes JSON plan + VectorStore, orchestrates sequential section generation via supervisor orchestration, synthesizes parent introductions, and produces the final report.

### 1.1 Key Design Principles

**Service-Oriented Hybrid Architecture:**
- **Services (Deterministic):** RAG Retriever, Quality Gate, ReferenceBuilder, Exporter
- **Agents (Cognitive/LLM):** Worker LLM, Parent Synthesis Agent
- **Orchestrator (Graph Architecture):** Defines workflow topology, routing logic, state transitions, and retry mechanisms via LangGraph

**Three Guiding Principles:**
1. **Single Responsibility:** Each node does one thing completely
2. **Data Communication:** Nodes exchange only Pydantic structures (PaquetContexte, WorkerOutput, ObjetResume) via PipelineState
3. **Separation of Concerns:**
   - Orchestrator graph: defines *when*, *what context*, and *what transitions* occur
   - Services: validate and prepare data deterministically
   - Agents: decide *what to write* cognitively (LLM-based)

---

## 2. Component Inventory

| Component | Pipeline | Type | Responsibility |
|-----------|----------|------|-----------------|
| **RAG Service** | A & B | Service | Ingestion (`.store()`) and retrieval (`.retrieve()`) from Milvus with similarity filtering (threshold: 0.45) |
| **Orchestrator** | B | Graph Architecture | Compiled LangGraph defining workflow topology, conditional routing (retry logic, coverage gating, synthesis triggering), state management via PipelineState, and node orchestration |
| **Worker LLM** | B | Agent | Generates leaf node content with citations [N]; receives PaquetContexte prepared by Orchestrator |
| **Quality Gate** | B | Service | Validates WorkerOutput on 5 criteria; sends validation details + failure reasons to Orchestrator; enables adaptive retries via Orchestrator |
| **Parent Synthesis Agent** | B | Agent | Generates introductions for containers/root from child ObjetResume summaries (triggered after all children in a branch complete) |
| **ReferenceBuilder** | B | Service | Creates unified global citations with global document registry (runs after all content generation) |
| **Exporter** | B | Service | Assembles final document (DOCX/PDF/Markdown) via pre-order traversal with global citations |

---

## 3. Data Structures

### 3.1 Input: JSON Plan

```json
{
  "metadata": {
    "title": "Strategic Analysis Report",
    "tone": "academic",
    "language": "en",
    "audience": "technical_jury"
  },
  "sections": {
    "title": "Report Title",
    "type": "root",
    "children": [
      {
        "title": "1. Context",
        "type": "container",
        "description": "Strategic context and problem statement",
        "children": [
          {
            "title": "1.1 Market Overview",
            "type": "leaf",
            "description": "Current market landscape analysis"
          },
          {
            "title": "1.2 Key Problem",
            "type": "leaf",
            "description": "Central problem addressed"
          }
        ]
      },
      {
        "title": "2. Analysis",
        "type": "container",
        "tone": "analytical",
        "instructions": "Privilege comparative tables and key metrics",
        "children": [...]
      }
    ]
  }
}
```

### 3.2 Node Internal Structure

| Field | Type | Purpose |
|-------|------|---------|
| **id** | str | 'S2.1' — unique node identifier |
| **depth** | int | Tree depth (determines heading level in export) |
| **type** | enum | `root` \| `container` \| `leaf` |
| **title** | str | Section heading |
| **description** | str | Scope and purpose guidance |
| **status** | enum | `pending` > `in_progress` > `completed` \| `failed` \| `requires_human_review` |
| **content** | str \| None | Generated text with citations [N] |
| **params_resolved** | dict | tone, audience, language (inherited/overridden) |

### 3.3 ObjetResume — Single Node Summary

**Structure:**
```python
@dataclass
class ObjetResume:
    node_id: str                    # e.g., "S1.1"
    node_title: str                 # "Market Overview"
    contenu: str                    # Contenu du noeud produit par llm
    resume_court: str               # 80-120 words summary of THIS node's content
    marqueurs_ton: list[str]        # ["analytical", "data-driven"] — tonal markers
    key_claims: list[str]           # Main factual assertions generated in this section
```

**Purpose:** Stores the *current node's* summary only. Does NOT contain previous resumes (those are accessed via `all_previous_resumes` in PaquetContexte).

---

## 4. Execution Model: Sequential Generation

### 4.1 Execution Flow

```
[Start: Receive document structure in JSON]
    ↓
[Convert JSON → TOON / Clean Markdown] 
    ├─ Removes heavy syntax (braces, quotes, trailing commas)
    └─ Saves ~40% of prompt tokens for efficient context window consumption
    ↓
[Start: Parse TOON structure → Build execution node sequence]
    ↓
[Orchestrator initializes workflow state]
    ↓
[Leaf nodes execute sequentially by tree order]
    │
    └─→ FOR each leaf node (S1.1, S1.2, S2.1, S2.2, ...):
        ├─ [Orchestrator retrieves sources via RAG Service]
        ├─ [Orchestrator calculates SourceCoverageScore (SC)]
        ├─ [Orchestrator checks SC threshold]
        │  ├─ If SC < 0.35 → BLOCK; report to user; skip to next leaf
        │  ├─ If 0.35 ≤ SC < 0.60 → FLAG for human review
        │  └─ If SC ≥ 0.60 → Proceed
        │
        ├─ [Orchestrator prepares PaquetContexte]
        │  └─ Fetches history from MemoireContexte mapped as:
        │     ├─ "All Previous Sections Resumes":
        │     │  Lists already-written sections with their distinct headers.
        │     │  (Rule for LLM: Strictly avoid repeating these facts/contents).
        │     │
        │     └─ "Immediate Last Section Resume":
        │        Specifically isolates the resume of the previous leaf.
        │        (Rule for LLM: Use only to build a fluid narrative transition).
        │
        ├─ [Worker LLM generates content]
        │  └─ Receives the newly structured PaquetContexte only
        ├─ [Quality Gate validates WorkerOutput]
        │  └─ If validation fails → Retry (up to 2x)
        ├─ [Orchestrator stores current leaf ObjetResume in MemoireContexte]
        └─ [Continue to next leaf]
    ↓
[Container synthesis (depth-first, bottom-up)]
    │
    └─→ FOR each container (starting from deepest):
        ├─ [Orchestrator waits for all leaf children to complete]
        ├─ [Orchestrator gathers child ObjetResume from MemoireContexte]
        ├─ [Parent Synthesis Agent generates introduction]
        ├─ [Quality Gate validates]
        ├─ [Orchestrator stores container resume in MemoireContexte]
        └─ [Continue to next container level]
    ↓
[Root synthesis]
    ├─ [Parent Synthesis Agent generates document introduction]
    ├─ [Quality Gate validates]
    └─ [Orchestrator stores root resume]
    ↓
[ReferenceBuilder: Global citation renumbering]
    ├─ [Traverse all sections in document order]
    ├─ [Build global document registry]
    ├─ [Map local citations to global numbers]
    └─ [Generate References section]
    ↓
[Exporter: Assemble final document]
    ├─ [Pre-order traversal of node tree]
    ├─ [Insert content + global citations]
    ├─ [Format DOCX/PDF/Markdown]
    └─ [Attach metadata: flagged sections, warnings]
    ↓
[End: Return final document + metadata]
```

### 4.2 Sequential Guarantees

- Each leaf waits for all previous leaves to complete
- All_previous_resumes in memory = full generation history
- No context bloat (resumes are ~100-120 words each, not full text)
- Deterministic ordering (no race conditions)
- Clear dependency graph (no circular waits)

---

## 5. Data Flow: PaquetContexte and MemoireContexte

### 5.1 PaquetContexte — Per-Section Context Envelope

Prepared by Orchestrator and transmitted to Worker LLM for each leaf generation:

| Field | Type | Purpose |
|-------|------|---------|
| **node** | Node | Current node with title, description, params_resolved |
| **retrieved_chunks** | list[RetrievedChunk] | Top-ranked sources from `.retrieve()` with similarity >0.45 |
| **source_coverage_score** | float | SC metric (0.0-1.0) indicating source sufficiency |
| **all_previous_resumes** | dict[str, ObjetResume] | All prior section summaries (cumulative history, not just previous sibling) |
| **title_next_section** | str \| None | Title of next section to avoid content overlap |
| **params_global** | dict | tone, audience, language, instructions |

**What Worker does NOT receive:** Full text of any generated section (only resumes from previous sections are visible via `all_previous_resumes`).

**Relationship to MemoireContexte:**
Each completed node stores its `ObjetResume` in MemoireContexte: `memoire_contexte[node_id] = ObjetResume`. When preparing PaquetContexte for the next node, the Orchestrator extracts all prior ObjetResume objects and passes them as `all_previous_resumes: dict[str, ObjetResume]`. This separation ensures:
- Nodes provide single-node summaries (ObjetResume = one summary)
- Historical context is accessed through PaquetContexte (all_previous_resumes = collection of prior summaries)
- No duplication: previous resumes live in one place (MemoireContexte) and are shared via PaquetContexte

### 5.2 MemoireContexte — Shared Resume Registry

- **Structure:** `dict[str, ObjetResume]` where each key is a `node_id` and each value is that node's *single* summary
  - Example: `{"S1.1": ObjetResume(...), "S1.2": ObjetResume(...), "S2.0": ObjetResume(...)}`
- **Lifecycle:** One write per node (after Quality Gate validation), then immutable for that node
- **Access Pattern:**
  - Orchestrator reads from MemoireContexte to build `all_previous_resumes` dict for PaquetContexte
  - Worker never reads MemoireContexte directly—only receives `all_previous_resumes` via PaquetContexte
  - Exporter reads full MemoireContexte to assemble final document (pre-order traversal)
- **Isolation Principal:** Nodes store their own summary; historical context shared via PaquetContexte, not by passing ObjetResume collections

**Distinction:**
- `ObjetResume` = single node's summary (stored in MemoireContexte)
- `all_previous_resumes` = collection of prior ObjetResume objects (built by Orchestrator, passed in PaquetContexte)

---

## 6. Source Coverage Model

### 6.1 SourceCoverageScore (SC) Calculation — Logarithmic Density Adjustment

To prevent a simple chunk count from biasing the system, the orchestrator calculates a **Coverage Score ($SC$)** normalized between $0.0$ and $1.0$. This score weights the mathematical quality of texts by their length, then adjusts the result using a logarithmic curve based on the quantity of material found relative to the writing budget requested for the node.

**Equation:**

$$SC = \left( \frac{\sum_{i=1}^{N} (sim_i \times T_i)}{\sum_{i=1}^{N} T_i} \right) \times \min\left(1.0, \frac{\ln\left(1 + \frac{\sum_{i=1}^{N} T_i}{\text{Budget}_{\text{node}}}\right)}{\ln(2)}\right)$$

**Variables:**
- $sim_i$ = semantic similarity of chunk $i$ (minimum threshold: 0.45)
- $T_i$ = token count of chunk $i$
- $N$ = number of chunks above the 0.45 threshold
- $\text{Budget}_{\text{node}}$ = token budget constraint for the node 

**Two Components:**

1. **Quality Weighting:** $\frac{\sum_{i=1}^{N} (sim_i \times T_i)}{\sum_{i=1}^{N} T_i}$ : weighted average of similarity scores by token count
2. **Quantity Factor:** $\min\left(1.0, \frac{\ln(1 + \text{total\_tokens} / \text{budget})}{\ln(2)}\right)$ : logarithmic penalty if material is insufficient relative to the node's writing budget
   - If total_tokens ≈ budget: factor ≈ 1.0 (sufficient)
   - If total_tokens ≪ budget: factor ≪ 1.0 (penalized; worker cannot hallucinate 500 tokens from 100)
   - If total_tokens ≫ budget: factor capped at 1.0 (abundance doesn't over-reward)

### 6.2 WarningLevel Decision Matrix

| Level | SC Range | Behavior |
|-------|----------|----------|
| **OK** | $SC \geq 0.60$ | Generate normally; Worker cites all claims with [N] |
| **LOW_COVERAGE** | $0.35 \leq SC < 0.60$ | **Human-in-the-Loop:** Section flagged for review. Orchestrator pauses workflow and prompts user: "Section S2.1 has scarce sources (SC=0.52). Available sources: [list]. Approve generation or provide additional documents?" After human approval, Worker generates with SC value noted in metadata. |
| **NO_SOURCE** | $SC < 0.35$ | **STRICT MODE ONLY:** Section not generated. Placeholder inserted: `[INSUFFICIENT SOURCES: S2.1 requires manual research.]` Reported in metadata. Orchestrator skips to next leaf. |

**Rationale:** 
- OK: Sufficient coverage; proceed normally
- LOW_COVERAGE: Limited but acceptable quality; require human verification before generation to confirm document sourcing completeness
- NO_SOURCE: Material too sparse or off-topic; no generation attempted; requires either additional documents or manual research

---

## 7. Source Retrieval and Citation Management

### 7.1 Retrieval Output: RetrievedChunk

```python
@dataclass
class RetrievedChunk:
    chunk_id: str  # Unique chunk in Milvus (Root)
    content: str  # Chunk text content (Root)
    similarity_score: float  # 0.0–1.0 (minimum 0.45) (Root)

    # All secondary and source-specific properties are nested here
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "doc_id": "",  # Source document ID
            "doc_title": "",  # e.g., "Annual Report 2024"
            "heading": "",  # e.g., "Chapter 3, Section 2"
            "page_no": None,  # Page number (if available)
            ...: ...,  # Placeholder for other dynamic properties
        }
    )
```

### 7.2 Citation Workflow

1. **Worker generates with local citations:**
   ```
   "Market leader controls 45% share [1], with competitors representing [2]..."
   ```

2. **WorkerOutput includes citation metadata:**
   ```python
   citations_used: [
       {"local_key": "[1]", "chunk_id": "chunk_xyz", "doc_id": "annual_2024", "page": 5},
       {"local_key": "[2]", "chunk_id": "chunk_abc", "doc_id": "competitor_2024", "page": 12}
   ]
   ```

3. **Quality Gate validates citation fidelity:**
   - Claim in "[Market leader controls 45% share]" must appear in original chunk
   - If mismatch → Reject and retry

---

## 8. ReferenceBuilder: Global Citation Management

### 8.1 Global Document Registry

Maintain persistent registry during entire report generation:

```python
global_registry = {
    ("annual_2024", 5, "chk_9921"): { # (doc_id, page_ref , chunk_id) = unique key
        "global_citation": "[1]",
        "doc_title": "Annual Report 2024",
        "text_snippet": "Revenue increased by 12% following...",
        "used_in_sections": ["S1.1", "S3.2"],
        "first_appearance": "S1.1",
    },
    ("annual_2024", 5, "chk_9922"): {
        "global_citation": "[2]",
        "doc_title": "Annual Report 2024",
        "text_snippet": "Risk factors include persistent inflation...",
        "used_in_sections": ["S2.1"],
        "first_appearance": "S2.1",
    },
    ("competitor_2024", 12, "chk_5541"): {
        "global_citation": "[3]",
        "doc_title": "Competitor Analysis 2024",
        "text_snippet": "Market share dropped to 8% in Q4...",
        "used_in_sections": ["S1.2"],
        "first_appearance": "S1.2",
    },
    # ... more granular chunk mappings
}
```

**Advantages of the Tuple Key Registry `(doc_id, page_ref, chunk_id)`**

1. **Perfect Accuracy**
   By tracking the exact text block, the system knows exactly where the AI got its facts. If a page has a chart at the top and a paragraph at the bottom, it won't mix them up.

2. **No Double Counting**
   If the AI uses the exact same paragraph in two different chapters, the system is smart enough to use the same citation number instead of making a new one.

3. **Easy to Fact-Check**
   If someone asks, *"Where did the AI find this specific number?"*, you can look at the master list and see exactly which file it came from and which chapters used it.
---

## 9. Quality Gate: Validation and Fallback

### 9.1 Five Validation Criteria

| Criterion | Check | Action if Fail |
|-----------|-------|---|
| **1. JSON Schema** | WorkerOutput matches Pydantic | Reject, retry |
| **2. Citation Fidelity** | Each [N] claim exists in source chunk (or validate placeholder structure for NO_SOURCE) | Reject, retry |
| **3. No Unallowed Patterns** | No unstructured placeholders like `##`, `[TODO]`, `[INSERT...]` (but structured placeholders like `[INSUFFICIENT SOURCES: ...]` are valid) | Reject, retry |
| **4. Non-Redundancy** | Significant overlap with `all_previous_resumes` summaries? | Minor: warn & proceed. Major: reject, retry |
| **4. Continuity** | Is the flow between the previous section n - 1 and this one n smooth? | Compares the start of section $N$ with the summary of $N-1$. If coherence is below the threshold, it rejects and retries to fix the transition. |

### 9.2 Retry Strategy with Orchestrator Feedback

| Attempt | Quality Gate Action | Feedback to Orchestrator | Orchestrator Re-generation | Notes |
|---------|----------|---|---|---|
| **1st attempt** | Validate output | Pass/Fail + failure reasons | N/A | Normal generation with full context |
| **Validation fails** | Return: `{"status": "FAIL", "failed_criteria": [...], "details": "Citation [2] not found in source"}` | Orchestrator receives structured error object with: criterion, reason, conflicting data | **Retry 1:** Modify prompt with specific feedback. Example: "Citation issue detected in [2]. Rewrite using only [1] and [3]. Focus on verified claims." | PaquetContexte adapted based on failure reason |
| **Retry 1** | Validate output | Pass/Fail + updated failure reasons | Continue if pass; else proceed to Retry 2 | Simplify prompt; focus on clarity per specific failure |
| **Retry 2** | Validate output | Pass/Fail; if fail, mark section | **Retry 2:** Reduce context window. Provide explicit instruction list. "Must cite: A, B, C. No redundancy with S1.1." | Reduce context; increase constraints |
| **Final Failure** | Reject output; return `{"status": "UNRECOVERABLE_FAIL", "criteria_failed": [...]}` | Orchestrator receives final failure signal | **Mark section for human review.** Status: `REQUIRES_HUMAN_REVIEW` | Do NOT insert placeholder in final doc; instead, user notification in metadata |
| **NO_SOURCE case** | Validate placeholder structure: `[INSUFFICIENT SOURCES: section_id requires manual research.]` | Passes if structured correctly; validates that human can clearly see the gap in sources | **Accept placeholder:** Orchestrator records in metadata that this section requires manual research. Move to next section. | Quality Gate validates placeholder validity even though no LLM content was generated |

**Important:** Quality Gate validation applies to ALL outputs, including retry attempts.

### 9.3 Quality Gate Feedback Data Structure

When validation fails, Quality Gate returns a structured feedback object to Orchestrator:

```python
@dataclass
class CriterionViolation:
    # Machine-readable code (e.g., "ERR_CITATION_NOT_FOUND")
    error_code: str
    # Human-readable message
    error_message: str
    # Clear instructions to feed straight back into the LLM prompt
    recommendation: str


@dataclass
class ValidationFailure:
    status: str  # "FAIL" | "UNRECOVERABLE_FAIL"
    attempt_number: int  # 0 (1st attempt), 1 (Retry 1), 2 (Retry 2)
    failed_criteria: List[str]  # ["Citation Fidelity", "Non-Redundancy"]

    # Fixed, predictable dictionary mapping criteria to our violation objects
    details: Dict[str, CriterionViolation]
```

**Orchestrator uses this feedback to:**
1. Identify specific failures (not generic retry)
2. Modify prompt with targeted instructions
3. Track cumulative failures (if same criterion fails 2x, escalate to human review)

---

## 10. LLM Prompts (Streamlined)

### 10.1 Worker LLM — System Prompt

```
You are a professional report writer producing a section of a sourced business document.

ABSOLUTE RULES:
- Every factual claim MUST have a citation [N] corresponding to provided sources.
- Use ONLY information from "RAG SOURCES" write the content for "CURRENT SECTION TARGET". No hallucination.
- Output ONLY valid JSON matching WorkerOutput schema. No markdown, no prose.
- Do NOT repeat or re-explain any facts or topics found in the "ALL PREVIOUS SECTION SUMMARIES" block above.
- Use the "TRANSITION POINT" summary above to create a natural transition into your opening sentences. Write the ending of this section to lead smoothly into the upcoming FLOW TARGET  section.

OUTPUT FORMAT:
{
  "contenu": "Section text with [1] citations integrated...",
  "resume_court": "80-120 word summary for next sections",
  "marqueurs_ton": ["academic", "analytical"],
  "key_claims": ["claim 1", "claim 2"],
  "citations_used": [
        {
            "citation_label": "[1]",
            "chunk_id": "chk_9921",
        },
        {
            "citation_label": "[2]",
            "chunk_id": "chk_5541",
        }
    ]
}
```

### 10.2 Worker LLM — User Prompt (Per-Section), built by the Orchestrator

```
--- CURRENT SECTION TARGET ---
Section ID: {current_node.id}
Title: {current_node.title}
Depth: {current_node.depth}
Audience: {current_node.audience}
Tone: {current_node.tone}

--- ALL PREVIOUS SECTION SUMMARIES ---
{all_previous_resumes_formatted}

--- TRANSITION POINT (IMMEDIATE PREVIOUS SECTION) ---
Header: {previous_node.title}
Resume: {previous_node.resume}

--- FLOW TARGET (NEXT SECTION) ---
Header: {next_node.title}

--- RAG SOURCES ---
{sources_formatted}

--- CRITICAL WRITING SCOPE ---
- Description: {current_node.description}

Generate the JSON payload for {current_node.id} 
```

### 10.3 Parent Synthesis Agent — System Prompt

```
You are a professional report writer producing high-level introductions and bridge summaries for container sections (chapters and branches).

ABSOLUTE RULES:
- Do NOT use bracketed citations [N]. The children nodes handle the raw sources; your job is strictly to synthesize.
- Synthesize, don't stitch: Create a cohesive narrative from the children's summaries. Connect their core themes naturally.
- Output ONLY valid JSON matching WorkerOutput schema. No markdown, no prose.
- Ensure tonal consistency by utilizing the tone markers provided by the children.
- Use the "CHILDREN SUMMARIES" provided in the user prompt to draft this chapter's introductory text.

OUTPUT FORMAT:
{
  "contenu": "High-level chapter synthesis text without citations...",
  "resume_court": "80-120 word summary of this container for future parent nodes",
  "marqueurs_ton": ["academic", "analytical"],
  "key_claims": ["Major theme 1 identified", "Major theme 2 identified"],
}
```

### 10.4 Parent Synthesis Agent — User Prompt (Per-Container), built by the Orchestrator

```
--- CURRENT CONTAINER TARGET ---
Container ID: {current_node.id}
Title: {current_node.title}
Depth: {current_node.depth}
Audience: {current_node.audience}
Tone: {current_node.tone}

--- CHILDREN SUMMARIES ---
{for child in children:}
  Header: {child.title}
  Resume: {child.resume}
{end}

Generate the JSON payload for the container {current_node.id}.
```

---

## 11. Orchestrator: Graph Architecture & Orchestration Logic

The **Orchestrator is not a single node, but rather the entire compiled LangGraph**, a state machine that coordinates workflow topology, routing logic, and state transitions across all processing nodes.

### 11.1 Orchestrator as Graph Architecture

**Conceptual Model:**
```
PipelineOrchestrator (Compiled LangGraph)
├── State Management: PipelineState
│   ├── current_node (workflow position)
│   ├── all_node_outputs (accumulated results)
│   ├── memory_context (shared history)
│   ├── validation_failure (feedback from Quality Gate)
│   ├── attempt_count (retry counter: 0, 1, 2)
│   └── metadata (flagged sections, warnings)
│
├── Orchestration Nodes:
│   ├── worker_llm (state-driven: attempt_count tracks generation_attempt, retry_1, retry_2)
│   ├── quality_gate
│   ├── parent_synthesis
│   ├── reference_builder
│   └── exporter
│
└── Conditional Routing Logic:
    ├── If validation success → next_section (or parent_synthesis if all leaves done)
    ├── If validation fail & attempt_count < 2 → re-invoke worker_llm (incremented state)
    ├── If validation fail & attempt_count ≥ 2 → mark_human_review
    ├── If all leaves complete → trigger_parent_synthesis
    ├── If all containers complete → trigger_root_synthesis
    └── If all synthesis complete → reference_builder → exporter
```

### 11.2 Graph Nodes (Individual Components)

The Orchestrator graph contains these execution nodes:

| Node | Type | Responsibility | Triggered From State | Output State | Called By |
|------|------|---|---|---|---|
| **worker_llm** | Agent | Generate or regenerate section content. Orchestrator adapts context & prompt based on ValidationFailure feedback and attempt_count. | `pending` |  `finished` | Orchestrator routing; loops back if validation fails |
| **quality_gate** | Service | Validate WorkerOutput against 5 criteria; return ValidationFailure if needed | `pending` (after worker_llm runs) | `passed` (success) or `failed` (validation error) | Immediately after worker_llm |
| **parent_synthesis** | Agent | Generate container/root intro from child resumes | `pending` (all leaves in `passed` state) | `finished` | Orchestrator after all leaves complete |
| **reference_builder** | Service | Global citation mapping (3-pass algorithm) | `pending` (all content in `passed` state) | `finished` | Orchestrator after all synthesis complete |
| **exporter** | Service | Assemble final DOCX/PDF/Markdown | `pending` (all citations remapped) | `finished` | Orchestrator final stage |

### 11.3 Orchestration Responsibilities (Graph-Level)

**The orchestrator graph is responsible for:**

1. **Retrieval & Coverage Assessment**
   - Before worker invocation: call RAG Service `.retrieve()`
   - Calculate SourceCoverageScore (SC)
   - Route: OK → invoke worker_llm (section state: `pending` → `worker_llm` processing); LOW_COVERAGE → flag + wait for human approval; NO_SOURCE → insert placeholder, mark state `finished`

2. **Context Preparation**
   - Fetch from MemoireContexte: all_previous_resumes, sibling titles, metadata
   - Construct PaquetContexte with all required fields
   - Pass via PipelineState to worker node

3. **Adaptive Retry Orchestration** *(the key graph responsibility)*
   - After Quality Gate validation failure, examine ValidationFailure object
   - Extract `failed_criteria`
   - **Increment attempt_count** in PipelineState
   - Modify PaquetContexte (reduce context, pin sources, add constraints based on attempt_count)
   - Modify worker prompt (inject  targeted instructions)
   - **Re-invoke worker_llm node** with updated state (same node, different context/prompt)
   - Track cumulative failures per criterion (if same criterion fails at attempt_count=2 → escalate)

4. **State Transitions & Sequencing**
   - Sequence leaf generation left→right (deterministic order)
   - Wait for current leaf to complete (all retries exhausted) before next leaf
   - Detect when all leaves complete → trigger parent synthesis (bottom-up)
   - Detect when all containers complete → trigger root synthesis
   - Sequential guarantees: no race conditions, clear dependency graph

5. **Metadata Management**
   - Track section status in PipelineState: pending → in_progress → completed/failed/requires_human_review
   - Accumulate validation errors per section
   - Flag LOW_COVERAGE sections for human approval
   - Collect NO_SOURCE sections for reporting
   - Prepare user-facing report at end


### 11.4 Orchestration vs. Node Responsibilities

**Orchestrator (Graph):**
- Decides *when* to invoke nodes
- Decides *what context* to provide (PaquetContexte construction)
- Decides *how to route* based on node output
- Manages *retry sequences*

**Worker/Parent/Services (Nodes):**
- Receive PipelineState → read PaquetContexte
- Execute their logic independently
- Write results to PipelineState
- Return control to orchestrator (no inter-node calls)

---

## 12. Execution Summary: Sequential Workflow

### 12.1 High-Level Sequential Flow

```
1. [INPUT] Load JSON plan → Parse nodes into sequence
2. [ORCHESTRATOR INIT] Initialize MemoireContexte, status tracking
3. [GENERATION] For each leaf (sequentially):
   a. Orchestrator retrieves chunks from Milvus
   b. Orchestrator calculates coverage SC
   c. If SC < 0.35 → BLOCK; report to user; skip
   d. If 0.35 ≤ SC < 0.60 → FLAG for human review (wait for approval)
   e. If SC ≥ 0.60 → Proceed
   f. Orchestrator creates PaquetContexte with all_previous_resumes
   g. Orchestrator calls Worker LLM
   h. Quality Gate validates (retry up to 2x if needed) [see 12.2 for feedback loop]
   i. Orchestrator stores ObjetResume in MemoireContexte
4. [SYNTHESIS] For each container (depth-first, bottom-up):
   a. Orchestrator gathers child ObjetResume from MemoireContexte
   b. Orchestrator calls Parent Synthesis Agent
   c. Quality Gate validates
   d. Orchestrator stores in MemoireContexte
5. [ROOT SYNTHESIS] Orchestrator calls Parent Synthesis for root
6. [REFERENCES] ReferenceBuilder: Global citation mapping
7. [EXPORT] Exporter: Assemble DOCX/PDF/Markdown with References
8. [OUTPUT] Return final document + metadata (flagged sections, warnings)
```
---

## 13. Known Limitations & Future Improvements

### 13.1 Current Limitations (v2.0)

- **Sequential slower than parallel:** Trade-off for simplicity and dependency handling
- **No dynamic structure:** JSON plan static (no auto-generation from documents)
- **Human-in-loop manual:** Requires UI/user interaction for LOW_COVERAGE approval

### 13.2 Future Enhancements  

- Dynamic structure planner (auto-generate outline from document analysis)
- Multi-model fallback (switch to better model if primary fails)
- Intelligent context summarization (compress all_previous_resumes if it grows too large)
