# MULTI-AGENT AI ARCHITECTURE FOR COMMERCIAL CONTRACT CLAUSE ANALYSIS

**DSAN 6725 - Spring 2026**
**Alison Manna, Akshay Arun, Satomi Ito**

---

## Abstract

Contract review is a time-intensive and error-prone process requiring specialized legal expertise. Organizations without dedicated legal teams often lack the resources to thoroughly analyze contracts, exposing them to regulatory risk and unfavorable terms. This paper presents a multi-agent AI system that automates end-to-end commercial contract analysis, enabling faster, more consistent clause-level evaluation and risk assessment.

Our system employs a pipeline of four specialized agents orchestrated with LangGraph: an Ingestion Agent that parses raw contract text into clause-level segments, a Classification Agent that labels clauses using the 41-type CUAD taxonomy, a Risk Analysis Agent that evaluates risk factors and severity using CUAD's legal review questions as a structured rubric, and a Benchmark Agent that compares clause language to industry standards derived from the CUAD corpus. A central Orchestrator coordinates the pipeline and compiles results into a structured JSON risk report.

The system processes real commercial contracts from the SEC EDGAR database, applying the multi-agent pipeline to extract and analyze clauses. The system successfully classifies clauses by type and generates detailed risk assessments with explanatory factors that align with legal concerns. The system provides actionable insights without requiring legal domain expertise, reducing friction for non-legal stakeholders to access contract intelligence.

The system is deployed as an interactive web application (Streamlit on HuggingFace Spaces), enabling users to upload contracts and receive clause-by-clause analysis in real time. By combining structured prompting with multi-agent orchestration, we demonstrate that automated contract analysis can provide interpretable, reliable results suitable for supporting human decision-making in contract negotiation.


## 1.0 Introduction

### 1.1 Problem Statement

Commercial contracts are fundamental to business operations, yet contract review remains labor-intensive and error-prone, requiring specialized legal expertise. Organizations without dedicated in-house counsel lack resources for thorough analysis, delaying negotiations and increasing exposure to unfavorable terms. Manual review is cognitively demanding, inconsistent across reviewers, and often misses subtle risks. Reviewers also lack visibility into how contract terms compare to industry standards. Legal review is expensive, both in cost and opportunity cost of delayed decision-making. These challenges leave many organizations unable to perform comprehensive contract analysis.

### 1.2 Motivation

The gap between the need for contract analysis and the availability of legal expertise creates an opportunity for AI-driven automation. Most existing contract analysis tools require domain knowledge or human interpretation, limiting accessibility. Non-legal professionals lack the expertise to conduct thorough contract review, yet are responsible for evaluating deals and managing contractual relationships. An automated system that provides consistent, explainable risk assessment could democratize access to contract intelligence, enabling faster decision-making and reducing reliance on expensive legal resources. This is particularly valuable for small and mid-sized organizations, startups, and entities that cannot afford dedicated legal teams.

### 1.3 Research Questions

This work investigates whether a multi-agent system can reliably classify contract clauses according to the CUAD taxonomy using structured prompting and LLM-based agents. We also evaluate whether an agent-based approach to risk analysis can detect ambiguous language, missing protections, and deviations from norms in a way that aligns with legal concerns, producing explanations actionable for non-legal stakeholders. Finally, we assess whether comparing clause language to typical industry provisions helps users understand how their contracts deviate from market norms and enables more informed negotiation decisions.


## 2.0 Related Work

[Survey of existing contract analysis tools, legal tech, and AI approaches]

## 3.0 Datasets

### 3.1 CUAD Dataset

The Contract Understanding Atticus Dataset (CUAD) is a publicly available dataset of 510 commercial contracts with 13,000+ expert-annotated clause excerpts spanning 41 distinct clause types. The dataset includes binary labels for the presence or absence of each clause type, as well as full-text annotations identifying the relevant passages. The 41 clause types include clauses (Document Name, Parties, Agreement Date, Effective Date) through complex risk-bearing provisions (Non-Compete, Indemnification, Limitation of Liability, Termination for Cause). CUAD also includes legal review questions for each clause type, which serve as structured rubrics for evaluating the quality and completeness of clauses. In our system, CUAD serves two roles: as the classification taxonomy for the Classification Agent, and as the reference corpus for the Benchmark Agent to evaluate how clauses compare to industry standard language.

### 3.2 Test Contracts

We evaluate the system on real commercial contracts from the SEC EDGAR database, specifically Exhibit 10 filings (Material Contracts) submitted to the Securities and Exchange Commission. These contracts include services agreements, NDAs, licensing agreements, and other common commercial arrangements. The SEC EDGAR corpus provides real-world contract samples that reflect actual negotiated terms and industry practices.

### 3.3 Data Preprocessing

Contracts are converted to plain-text format. Minimal preprocessing is applied: removing non-ASCII characters, normalizing whitespace, and segmenting contracts into clause-level units based on structural markers (section headers, numbered subsections) or paragraph boundaries. No additional feature engineering or data augmentation is performed; the system operates directly on raw contract text to preserve the original language.

## 4.0 System Architecture

![Figure 1: Proposed agent architecture and data flow](img/architecture_diagram_updated.jpg)
*Figure 1: Proposed agent architecture and data flow*

### 4.1 Overview

The system implements a multi-agent pipeline for automated commercial contract clause analysis, orchestrated using LangGraph. The architecture follows a cluster pattern in which a central Orchestrator Agent receives contract documents, dispatches them through agents, and compiles their outputs into a unified risk report. Figure 1 presents the proposed agent architecture and data flow.

### 4.2 Agent Descriptions

The system comprises one central Orchestrator and four specialist agents:

**Orchestrator Agent.** Serves as a central hub of the system, coordinating the full contract analysis pipeline through LangGraph's StateGraph. The Orchestrator manages pipeline state and routes documents sequentially through the specialist agents. The report node aggregates clause-level results and produces a JSON summary containing clause annotations, risk scores, risk factors, benchmark similarity scores, and source text excerpts, along with a summary count of total clauses analyzed.

**Ingestion Agent.** Serves as the parsing layer of the pipeline. It accepts raw contract text from the Orchestrator and segments it into clause-level units using regex-based pattern matching to identify structural markers such as section headers, exhibit labels, and numbered subsections. When structural markers are absent, the agent falls back to paragraph-level segmentation using double-newline delimiters. Each clause is assigned a unique identifier and tagged with its originating section label. This segmentation enables clause-level analysis and manages token budgets when interfacing with the language model in subsequent stages.

**Classification Agent.** Labels clauses according to the CUAD taxonomy, spanning 41 provision types such as Governing Law, Non-Compete, Indemnification, and IP Ownership Assignment. The agent uses clear prompting with Anthropic's Claude Haiku model, returning structured JSON with the predicted clause type, a confidence score on a 0 to 1 scale, and a natural language reasoning trace for explainability. To optimize API costs, the Classification Agent processes the first 3 clauses of each contract; remaining clauses bypass this stage. On JSON parsing failure, the agent defaults to an "Other" classification rather than halting the pipeline.

**Risk Analysis Agent.** Evaluates each classified clause on a severity scale using the CUAD's taxonomy annotations as a structured rubric. For each clause type, the agent injects the corresponding CUAD legal review questions into the prompt, guiding the language model to assess ambiguous or vague language, missing protective provisions standard for that clause type, and deviation from standard phrasing. The agent uses zero-shot prompting with Anthropic's Claude Haiku model, returning structured JSON containing a risk score (0 to 1 scale), an itemized list of specific risk factors, and a natural-language reasoning trace.

**Benchmark Agent.** Contextualizes each clause by comparing it to industry standard language for its classified type. Rather than retrieving comparable contracts, the agent uses zero-shot prompting with Anthropic's Claude Haiku model and its training knowledge of typical provisions in commercial contracts (NDAs, master service agreements, licensing agreements, etc.). The agent evaluates how closely a clause aligns with standard industry language and returns structured JSON containing a similarity score on a 0 to 1 scale (0 = unusual, 1 = standard), a list of deviations from standard language, and a summary of what is typical for that clause type. This contextualizes the analysis against industry norms without requiring external database retrieval.

### 4.3 State Management and Communication

Agents communicate through a shared, typed state object defined using Python's TypedDict. Key fields include raw_text, clauses, classified_clauses, risk_scores, benchmark_results, and report. Agents receive the full state as input and return only the fields they update, leveraging LangGraph's partial-state merge semantics to prevent side effects.

### 4.4 Pipeline Orchestration

The data flow proceeds as follows: the front-end interface forwards an uploaded contract to the Orchestrator Agent, the Orchestrator dispatches the document through the specialist agents, processed results are returned to the Orchestrator, and the Orchestrator compiles all outputs into a structured risk report containing clause-by-clause annotations, severity scores, benchmark comparisons, and actionable insights. The pipeline is implemented using LangGraph's StateGraph, chosen for its explicit graph semantics, state propagation support, and inspectable execution traces.

### 4.5 Architecture Evolution

Between Milestone 1 and Milestone 2, the architecture underwent a key revision: the SEC EDGAR dependency was removed in favor of using the CUAD dataset exclusively for both classification taxonomy and benchmarking. The dataset's role also shifted from a training corpus to a context-engineering resource, reducing external dependencies and simplifying the data pipeline while preserving analytical capabilities.

## 5.0 Data and Evaluation

### 5.1 Evaluation Methodology

### 5.2 Results

## 6.0 Models and Technologies

### 6.1 Language Models

All agents in the pipeline use Claude 3.5 Haiku through the Anthropic API. Haiku was selected for its combination of speed, capability, and cost efficiency. For structured reasoning tasks—clause classification, risk scoring, and semantic comparison—Haiku demonstrates strong performance without the overhead of larger models. Each agent is configured with token limits to balance response quality and cost: Classification Agent (256 tokens max), Risk Analysis and Benchmark Agents (512 tokens max each). This configuration enables the system to process multiple clauses per contract while maintaining reasonable API costs.

### 6.2 Frameworks and Libraries

LangGraph orchestrates the multi-agent pipeline using its StateGraph abstraction. LangGraph was chosen for its explicit graph representation of agent workflows, enabling transparent state propagation between agents, sequential node execution, and debuggable execution traces. 

LangChain provides the LLM interface, prompt management utilities, and structured JSON output parsing. LangChain's integration with Anthropic's API and its error handling for JSON parsing reduce boilerplate and improve reliability.

Streamlit powers the web interface, enabling rapid development of an interactive contract upload and analysis UI. Streamlit's native components (file uploader, text area, metric cards, expandable sections) align perfectly with the application's requirements.

### 6.3 Infrastructure and Deployment

The system is deployed on HuggingFace Spaces as a Streamlit application. HuggingFace Spaces provides free hosting for Streamlit apps with automatic GitHub integration, handles secure storage of API secrets, and auto-deploys on GitHub pushes. This enables users to access the system through a public URL without local installation.

### 6.4 Technical Design Rationale

The multi-agent architecture with LangGraph was chosen for modularity, interpretability, and flexibility. Each agent can be tested and improved independently; intermediate outputs (clause types, risk scores) are visible and debuggable; and agents can be tuned or swapped without affecting others.

Zero-shot prompting was used to avoid the overhead of labeled data collection and model training, instead leveraging Haiku's instruction-following capabilities. The CUAD dataset provides both the classification taxonomy and the implicit benchmark corpus, reducing dependency on external APIs and ensuring consistency between classification and benchmarking steps.

## 7.0 Responsible AI Considerations

### 7.1 Bias and Fairness

The CUAD dataset comprises 510 commercial contracts, predominantly from SEC filings of larger corporations. This introduces sampling bias toward large enterprises and US-based companies, potentially limiting the system's applicability to small and mid-sized businesses or non-US contract types. Additionally, the definition of "risk" embedded in CUAD's annotations reflects the perspectives of contract law experts; different jurisdictions or industries may assign different risk weights to similar clauses. To mitigate these biases, the system surfaces its reasoning (confidence scores, risk factors, benchmark deviations) so users can inspect and challenge its assessments. Users should be aware that the system's risk scores are advisory, not prescriptive, and should validate results against their own domain knowledge and legal expertise.

### 7.2 Hallucinations and Factual Accuracy

Claude Haiku can generate speculative or inaccurate risk factors. To reduce this risk, the Risk Analysis Agent's prompts are grounded in CUAD's structured legal review questions, which provide rubrics for specific clause types. Rather than reasoning freely, the model receives domain-specific guidance: "For a Non-Compete clause, assess whether the scope, geography, and duration are reasonable." This constrains outputs to relevant dimensions. The Benchmark Agent's similarity scores rely on the model's training knowledge of standard industry language rather than real-time database retrieval, which could introduce inaccuracies. The system mitigates this by displaying reasoning traces and confidence scores, enabling users to judge the reliability of each assessment.

### 7.3 Privacy and Data Handling

Users upload contracts to the Streamlit application on HuggingFace Spaces, where contract text is forwarded to the Anthropic API for analysis. Users should be aware that contract content is transmitted to external services and should not upload contracts containing highly sensitive information, trade secrets, or personally identifiable information without permission. Organizations handling regulated or confidential contracts should consider running the system in a private environment or obtaining data processing agreements. The system does not store uploaded contracts; they are processed on-demand and discarded after analysis.

### 7.4 Safety and Ethical Implications

This system is designed as an advisory tool to accelerate initial contract review, not to replace professional legal counsel. Users should treat the system's risk assessments as a starting point for human review, not as definitive legal opinions. The system has specific limitations: it analyzes only the first 3 clauses of each contract due to cost constraints; it uses zero-shot prompting without retrieval, relying on the model's training knowledge of "standard" clauses; it does not validate assessments against ground truth. For high-stakes contracts, legal review by qualified professionals remains essential. The system is most valuable for initial triage by non-legal stakeholders to identify which agreements require deeper legal scrutiny.

## 8.0 Findings and Discussion

### 8.1 Key Insights

### 8.2 Challenges and Solutions

### 8.3 Performance Trade-offs

### 8.4 Scalability Considerations

## 9.0 Conclusion and Future Work

### 9.1 Summary of Contributions

### 9.2 Limitations

### 9.3 Future Directions

## References

