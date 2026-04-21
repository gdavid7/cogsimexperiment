# Evidence-Based Development

## Core Principle

**ALL technical decisions, implementations, and modifications MUST be justified by credible sources.**

Do not make assumptions, guesses, or rely on general knowledge. Every decision must be traceable to authoritative documentation.

## Required Source Hierarchy

When making any technical decision, consult sources in this priority order:

### 1. Primary Research (Highest Authority)
- **TRIBEv2.pdf** - The research paper in the repository root
  - Model architecture details
  - Brain encoding methodology
  - ICA network definitions
  - Validation approaches
- Always cite specific sections, figures, or equations from the paper

### 2. Project Requirements
- **`.kiro/specs/cognitive-similarity/requirements.md`** - Feature requirements
  - Functional specifications
  - API contracts
  - Validation criteria
- Quote requirement IDs when implementing features

### 3. Library Documentation
- **TRIBE v2 GitHub repository**: https://github.com/facebookresearch/tribev2
  - Model API usage
  - Input/output formats
  - Configuration options

- **NumPy documentation**: https://numpy.org/doc/stable/
  - Array operations
  - Data types
  - Mathematical functions

- **scikit-learn documentation**: https://scikit-learn.org/stable/
  - Correlation metrics
  - Statistical methods

- **nilearn documentation**: https://nilearn.github.io/stable/
  - Neuroimaging data handling
  - Brain atlas operations

- **Hypothesis documentation**: https://hypothesis.readthedocs.io/
  - Property-based testing strategies
  - Test generation

### 4. Existing Codebase
- Current implementation in `cognitive_similarity/` modules
- Test files demonstrating expected behavior
- Docstrings with requirement references

## Mandatory Practices

### Before Writing Code
1. **Identify the requirement** - Which spec requirement are you implementing?
2. **Find the source** - Where is this requirement defined? (paper, docs, requirements)
3. **Quote the source** - Copy the exact text/equation/specification
4. **Verify understanding** - Does your interpretation match the source exactly?

### When Making Design Decisions
- ❌ "I think we should use method X because it's common"
- ✅ "According to TRIBEv2.pdf Section 3.2, the model uses method X for [specific reason]"

- ❌ "This parameter should probably be 0.1"
- ✅ "TRIBEv2.pdf Figure 4 shows top_percentile=0.10 for binary masks"

- ❌ "Let's add error handling here"
- ✅ "Requirements 4.6 specifies: 'Zero-variance inputs SHALL return score 0.0 with warning'"

### When Implementing Algorithms
- **Cite equations**: "Implementing Pearson correlation as defined in TRIBEv2.pdf Equation 5"
- **Reference figures**: "Network masks derived from Figure 3 ICA component visualization"
- **Quote specifications**: "Per requirements.md 3.2: 'Continuous mode SHALL normalize weights to sum to 1.0'"

### When Debugging or Modifying
1. **Understand current behavior** - Read existing code and tests
2. **Find specification** - What does the paper/docs say this should do?
3. **Compare** - Does current behavior match specification?
4. **Justify change** - Cite the source that supports your modification

### When Uncertain
- **DO NOT GUESS** - Stop and search for documentation
- **Use web search** - Look up official library documentation
- **Read the paper** - TRIBEv2.pdf contains most architectural decisions
- **Check tests** - Property tests often encode requirements
- **Ask for clarification** - If sources conflict or are unclear

## Documentation Requirements

### Code Comments
When implementing complex logic, include source citations:

```python
# Per TRIBEv2.pdf Section 4.1: "Binary masks select top 10% of vertices
# by absolute component weight for each ICA network"
top_percentile = 0.10
mask = np.abs(component) >= np.percentile(np.abs(component), (1 - top_percentile) * 100)
```

### Commit Messages
Reference sources in commit messages:

```
Implement continuous ICA weighting (Req 3.2)

Per requirements.md 3.2 and TRIBEv2.pdf Equation 6:
- Normalize weights: w = |component| / sum(|component|)
- Apply sqrt weighting for Pearson correlation
```

### Test Docstrings
Always include requirement validation:

```python
def test_property_7_continuous_ica_weight_normalization():
    """
    Validates: Requirements 3.2
    
    Per TRIBEv2.pdf Section 3.3: "Continuous weighting mode uses
    normalized absolute component values as per-vertex weights"
    """
```

## Red Flags (Avoid These)

- ❌ "This is the standard way to do X"
- ❌ "Based on my experience..."
- ❌ "I assume this should..."
- ❌ "Typically in neuroscience..."
- ❌ "Let me try this approach..."

## Green Flags (Do These)

- ✅ "According to TRIBEv2.pdf Section X..."
- ✅ "Requirements.md 4.5 specifies..."
- ✅ "NumPy documentation states..."
- ✅ "The existing test_facade.py demonstrates..."
- ✅ "Let me search the TRIBE v2 documentation for..."

## When Sources Conflict

If you find conflicting information:

1. **Document the conflict** - Note what each source says
2. **Check hierarchy** - Research paper > Requirements > Library docs
3. **Verify with tests** - What do existing tests expect?
4. **Flag for review** - Mention the conflict and ask for clarification

## Example Workflow

**Task**: Implement whole-cortex score calculation

**Step 1 - Find Requirement**:
```
Requirements.md 4.4: "Whole-cortex score SHALL be computed as the 
vertex-count-weighted average of the 5 network scores"
```

**Step 2 - Find Algorithm**:
```
TRIBEv2.pdf Section 4.2: "The aggregate cortical similarity is computed
as: score_wc = Σ(score_i × n_i) / Σ(n_i) where n_i is the vertex count
for network i"
```

**Step 3 - Implement with Citations**:
```python
def compute_whole_cortex_score(network_scores: dict) -> float:
    """Compute vertex-count-weighted average (Req 4.4, TRIBEv2 Sec 4.2)."""
    total_vertices = sum(ns.vertex_count for ns in network_scores.values())
    return sum(
        ns.score * ns.vertex_count / total_vertices
        for ns in network_scores.values()
    )
```

**Step 4 - Test with Property**:
```python
def test_property_9_whole_cortex_score_formula():
    """Validates: Requirements 4.4
    
    Per TRIBEv2.pdf Equation 7: whole_cortex_score must equal
    the vertex-count-weighted average of network scores.
    """
```

## Summary

Every line of code, every design decision, every parameter value must be justified by:
- The TRIBEv2.pdf research paper
- The requirements.md specification
- Official library documentation
- Existing validated code/tests

**When in doubt, cite your source. If you can't find a source, stop and search for one.**
