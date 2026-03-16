# Data 

## CUAD (Contract Understanding Atticus Dataset)

**Contains:**

- 510 commercial contracts (NDAs, service agreements, licensing agreements, etc.)
- 13,000+ expert annotations
- 41 clause type categories (e.g., "Governing Law", "Termination for Convenience", "Limitation of Liability", "Non-Compete", "Indemnification", etc)
- Each annotation is a span within a contract + the question it answers

**Uses:**

1. **For Classification Agent taxonomy**: the 41 clause categories are the classification schema. When the classification agent receives a clause segment, it uses the 41 types as the set of all labels it can assign. Approach: put all 41 category names + brief descriptions in the system prompt as a reference list

2. **For Risk Analysis Agent**: The annotations tell  which categories expert annotators were trained to flag as legally significant, which can support the risk scoring logic (gives LLM structured guidance about what to look for)

3. **Benchmark corpus (secondary)**: the 510 CUAD contracts can be embedded and added to the vector store alongside EDGAR contracts

4. Sample input contracts for testing/demo (maybe): Since CUAD contracts are real commercial contracts, a subset (20-30?) could be used as "uploaded contracts" during testing and demo, rather than needing to find separate test documents.


**Need to collect:**

- Full taxonomy: list of 41 category names + description of each -> data/cuad/taxonomy.json
- Contract texts: save the 510 contract full texts -> data/cuad/contracts/
- Annotations (the labeled spans per contract) -> data/cuad/annotations.json