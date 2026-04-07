# MULTI-AGENT AI ARCHITECTURE FOR COMMERCIAL CONTRACT CLAUSE ANALYSIS

**DSAN 6725 - Spring 2026**
**Alison Manna, Akshay Arun, Satomi Ito**

---

## Abstract

## 1.0 Introduction/Background

## 2.0 Datasets

## 3.0 System Architecture

![Figure 1: Proposed agent architecture and data flow](img/architecture_diagram_updated.jpg)
*Figure 1: Proposed agent architecture and data flow*

### 3.1 Overview

The system implements a multi-agent pipeline for automated commercial contract clause analysis, orchestrated using LangGraph. The architecture follows a cluster pattern in which a central Orchestrator Agent receives contract documents, dispatches them through agents, and compiles their outputs into a unified risk report. Figure 1 presents the proposed agent architecture and data flow.

### 3.2 Agent Descriptions

The system comprises one central Orchestrator and four specialist agents:

**Orchestrator Agent.** Serves as a central hub of the system, coordinating the full contract analysis pipeline through LangGraph's StateGraph. The Orchestrator manages pipeline state and routes documents sequentially through the specialist agents before compiling their outputs into a risk report. The report node aggregates clause-level results and categorizes them by risk severity producing a JSON summary containing clause annotations, risk scores, risk factors, benchmark similarity scores, and source text excerpts.

**Ingestion Agent.** Serves as the parsing layer of the pipeline. It accepts raw contract text from the Orchestrator and segments it into clause-level units using regex-based pattern matching to identify structural markers such as section headers, exhibit labels, and numbered subsections. When structural markers are absent, the agent falls back to paragraph-level segmentation using double-newline delimiters. Each clause is assigned a unique identifier and tagged with its originating section label. This segmentation enables clause-level analysis and manages token budgets when interfacing with the language model in subsequent stages.

**Classification Agent.** Labels each clause according to the CUAD taxonomy, spanning provisions such as Governing Law, Non-Compete, Indemnification, and IP Ownership Assignment. The agent uses clear prompting with Anthropic's Claude Haiku model, returning structured JSON with the predicted clause type, a confidence score on a 0 to 1 scale, and a natural language reasoning trace for explainability. On parsing failure, the agent results to an "Other" classification rather than halting the pipeline.

**Risk Analysis Agent.** Evaluates each classified clause on a severity scale using the CUAD's taxonomy annotations as a structured rubric. For each clause type, the agent injects the corresponding CUAD legal review questions into the prompt, guiding the language model to assess ambiguous or vague language, missing protective provisions standard for that clause type, and deviation from standard phrasing. The agent uses zero-shot prompting with Anthropic's Claude Haiku model, returning structured JSON containing a 0 to 1 scale, an itemized list of specific risk factors, and a natural-language reasoning trace.

**Benchmark Agent.** Compares each clause against standard industry language for its classified type. The agent uses zero-shot prompting with Anthropic's Claude Haiku model to evaluate how a clause aligns with typical provisions found in commercial contracts such as NDAs, master service agreements, and licensing agreements. It returns structured JSON containing a similarity score on a 0 to 1 scale, a list of specific deviations from standard language, and a summary of what is typical for that clause type. This contextualizes the analysis against industry norms rather than evaluating the documents in isolation.

### 3.3 State Management and Communication

Agents communicate through a shared, typed state object defined using Python's TypedDict. Key fields include raw_text, clauses, classified_clauses, risk_scores, benchmark_results, and report. Agents receive the full state as input and return only the fields they update, leveraging LangGraph's partial-state merge semantics to prevent side effects.

### 3.4 Pipeline Orchestration

The data flow proceeds as follows: the front-end interface forwards an uploaded contract to the Orchestrator Agent, the Orchestrator dispatches the document through the specialist agents, processed results are returned to the Orchestrator, and the Orchestrator compiles all outputs into a structured risk report containing clause-by-clause annotations, severity scores, benchmark comparisons, and actionable insights. The pipeline is implemented using LangGraph's StateGraph, chosen for its explicit graph semantics, state propagation support, and inspectable execution traces.

### 3.5 Architecture Evolution

Between Milestone 1 and Milestone 2, the architecture underwent a key revision: the SEC EDGAR dependency was removed in favor of using the CUAD dataset exclusively for both classification taxonomy and benchmarking. The dataset's role also shifted from a training corpus to a context-engineering resource, reducing external dependencies and simplifying the data pipeline while preserving analytical capabilities.
