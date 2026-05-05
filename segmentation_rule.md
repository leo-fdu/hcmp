# HCMP v1 Segmentation Rule

## 1. Purpose

This document defines the segmentation rule used in **HCMP v1** for the Stage 2 cut-bond segmentation task.

The goal of this rule is **not** to construct a complete conventional functional-group parser.  
Instead, the goal is to generate a conservative, low-noise, chemically interpretable bond-level segmentation target.

The segmentation rule is designed to identify chemically salient motifs and mark the bonds that separate them from the surrounding molecular graph.

The final supervision target is a bond-level binary label:

\[
y_b^{\mathrm{cut}} =
\mathbb{1}[b \text{ is a boundary between chemically salient segments}]
\]

The segmentation rule follows a priority-based design:

\[
\text{ring system}
>
\text{non-ring unsaturated/conjugated motif}
>
\text{heteroatom connected cluster}
>
\text{terminal heteroatom}
\]

Higher-priority segments are assigned first.  
Once an atom or bond has been assigned to a higher-priority segment, it is masked from lower-priority detection.

---

## 2. Design Philosophy

The segmentation rule is based on the following principle:

> Preserve chemically coherent motifs and cut at stable, interpretable motif boundaries.

This rule intentionally emphasizes:

- ring systems;
- non-ring unsaturated and conjugated motifs;
- heteroatom connected clusters;
- terminal heteroatom substituents.

It intentionally does **not** attempt to segment every traditional functional group.

In particular, simple non-terminal saturated heteroatom linkers such as ordinary ethers, thioethers, and amines are not necessarily segmented as independent motifs in HCMP v1.

This is a deliberate conservative choice.  
The rule should avoid noisy or ambiguous segmentation labels and should prioritize motifs that are graph-theoretically stable and chemically salient.

---

## 3. Preprocessing

Before segmentation:

1. Parse the molecule with RDKit.
2. Sanitize the molecule.
3. Remove explicit hydrogens for segmentation.
4. Keep heavy-atom graph information:
   - atomic number;
   - aromaticity;
   - formal charge;
   - hybridization;
   - ring membership;
   - bond order;
   - aromaticity;
   - conjugation flag;
   - ring membership.

Hydrogens are removed from the segmentation graph, but implicit hydrogen information may still be retained as atom features for the model.

---

## 4. Segment Priority System

Each atom and bond is assigned to at most one segment.

If multiple candidate motifs contain the same atom or bond, the motif with the higher priority wins.

Priority order:

1. Ring-system segment.
2. Non-ring unsaturated/conjugated segment.
3. Heteroatom connected cluster.
4. Terminal heteroatom segment.

This priority rule is intentional.  
For example:

- heteroatoms inside aromatic rings belong to the ring segment;
- carbonyls inside ring systems belong to the ring segment;
- amide or ester heteroatoms captured by the conjugated carbonyl motif are not separately processed as terminal heteroatoms;
- terminal heteroatoms already assigned to higher-priority motifs are ignored by the terminal heteroatom rule.

---

## 5. Priority 1: Ring-System Segments

### 5.1 Goal

Ring systems are assigned first because they are stable, chemically salient graph motifs.

The rule preserves ring integrity and avoids cutting inside ring systems.

### 5.2 Ring-system construction

Identify all rings in the molecule.

Merge rings into the same ring-system segment if they are structurally coupled, including:

- fused rings;
- bridged rings;
- spiro rings;
- rings sharing atoms;
- rings sharing bonds;
- polycyclic ring systems.

These structures are treated as one segment.

### 5.3 Ring-ring conjugation

Separate ring systems may be merged only under explicitly allowed **ring-ring conjugation** rules.

Important restriction:

> Only ring-ring conjugation can merge ring segments.

Ring-functional-group conjugation does not merge the ring with the non-ring motif.

Therefore, the following should not cause ring and non-ring motif merging:

- phenyl-carbonyl conjugation;
- phenyl-nitro conjugation;
- phenyl-amine conjugation;
- phenyl-alkene conjugation;
- phenyl-phenyl connection through a non-ring unsaturated linker.

For example, in a structure like:

\[
\mathrm{Ph{-}CH=CH{-}Ph}
\]

the preferred segmentation is:

- phenyl ring segment;
- non-ring \( \mathrm{CH=CH} \) unsaturated segment;
- phenyl ring segment.

The rings are not merged through the non-ring double bond linker.

### 5.4 Ring-internal atoms

Atoms inside a ring system remain part of the ring segment even if they are heteroatoms or part of an unsaturated motif.

Examples:

- pyridine nitrogen belongs to the ring segment;
- thiophene sulfur belongs to the ring segment;
- indole nitrogen belongs to the fused ring segment;
- lactone/lactam carbonyl inside a ring belongs to the ring segment under the ring-first priority rule.

### 5.5 Bond labels inside ring segments

All bonds inside a ring-system segment are protected and labeled as non-cut.

---

## 6. Priority 2: Non-Ring Unsaturated / Conjugated Motifs

### 6.1 Goal

This rule identifies non-ring unsaturated and conjugated motifs.

It is designed to capture:

- isolated double bonds;
- isolated triple bonds;
- carbonyls;
- nitriles;
- imines;
- alkenes;
- alkynes;
- dienes;
- enones;
- allenes and cumulenes;
- amide-like and ester-like conjugated carbonyl systems when RDKit perceives the relevant bonds as conjugated.

### 6.2 Seed bonds

After masking all Priority 1 ring-system atoms and bonds, identify all remaining non-ring multiple bonds.

A bond is a multiple-bond seed if:

\[
\text{bond order} > 1
\]

In RDKit-style implementation, this may correspond to:

```python
bond.GetBondTypeAsDouble() > 1
```

or checking bond types such as:

- `DOUBLE`;
- `TRIPLE`.

Multiple-bond seeds are included even if RDKit does not mark them as conjugated.

This is important because isolated double bonds or isolated carbonyls may not be marked as conjugated by RDKit, but they should still form unsaturated motifs.

### 6.3 RDKit conjugation expansion

Starting from non-ring multiple-bond seeds, expand through RDKit-perceived conjugated bonds.

In RDKit, the relevant bond flag is:

```python
bond.GetIsConjugated()
```

The resulting motif should be described as an RDKit-perceived graph-conjugated motif, not as a physically exact electronic conjugation system.

Recommended wording:

> HCMP v1 identifies non-ring multiple-bond-seeded, RDKit-conjugation-expanded motifs.

This means:

1. start from non-ring multiple bonds;
2. add non-ring bonds whose `GetIsConjugated()` value is true;
3. find connected components in this auxiliary bond graph;
4. keep only components containing at least one multiple bond;
5. do not merge these components into ring-system segments.

### 6.4 Ring exclusion

Priority 2 must not cross into Priority 1 ring segments.

Even if RDKit marks a ring-functional-group attachment bond as conjugated, the ring segment and the non-ring motif remain separate.

For example:

- in aniline, the ring and amino group are not merged;
- in phenol, the ring and hydroxyl group are not merged;
- in nitrobenzene, the ring and nitro group are not merged;
- in styrene, the ring and vinyl group are not merged;
- in benzamide, the ring and amide motif are not merged.

The attachment bond between a ring segment and a non-ring motif may become a cut bond.

### 6.5 Allene and cumulene behavior

RDKit may mark allene-like adjacent multiple bonds as conjugated for graph perception purposes.

For HCMP v1, this is acceptable and useful.

Allene and cumulene structures should be preserved as unsaturated motifs even if their electronic structure is not equivalent to ordinary diene conjugation in a strict physical sense.

The rule should therefore preserve adjacent multiple-bond systems such as:

\[
\mathrm{C=C=C}
\]

as one non-ring unsaturated motif.

### 6.6 Bond labels inside unsaturated/conjugated motifs

All bonds inside a Priority 2 motif are protected and labeled as non-cut.

Attachment bonds connecting this motif to a different segment or to unsegmented background may be labeled as cut in the final label-generation step.

---

## 7. Priority 3: Heteroatom Connected Clusters

### 7.1 Goal

This rule identifies connected heteroatom clusters that remain after higher-priority segments have been assigned and masked.

The rule is intended to capture motifs such as:

- nitro-like heteroatom groups;
- peroxide-like groups;
- azide-like groups;
- azo/diazo-like groups;
- sulfur-oxygen clusters;
- phosphorus-oxygen clusters;
- other connected heteroatom-rich motifs.

### 7.2 Definition of heteroatom

A heteroatom is any atom whose atomic number is neither hydrogen nor carbon:

\[
Z \notin \{1, 6\}
\]

Since explicit hydrogens have already been removed, in practice this means:

\[
Z \neq 6
\]

for heavy atoms.

### 7.3 Connected-cluster construction

After removing atoms and bonds already assigned to Priority 1 and Priority 2 segments:

1. collect all remaining heteroatoms;
2. build the heteroatom-induced subgraph;
3. find connected components in this subgraph;
4. each connected component is a candidate heteroatom-cluster segment.

This is a standard graph connected-component problem and can be implemented using DFS, BFS, NetworkX, or simple RDKit neighbor traversal.

### 7.4 Priority masking

If a heteroatom already belongs to a ring segment or a non-ring conjugated motif, it is not reconsidered here.

Examples:

- heteroaromatic atoms are already part of ring segments;
- amide nitrogen may already be part of a Priority 2 conjugated carbonyl motif;
- ester oxygen may already be part of a Priority 2 conjugated carbonyl motif;
- nitro atoms may already be captured by Priority 2 if represented through multiple bonds and conjugation.

### 7.5 Bond labels inside heteroatom clusters

All bonds inside a heteroatom connected cluster are protected and labeled as non-cut.

Attachment bonds connecting the cluster to other segments or to unsegmented background may be labeled as cut.

---

## 8. Priority 4: Terminal Heteroatom Segments

### 8.1 Goal

This final rule identifies remaining terminal heteroatom substituents.

It is intentionally conservative.

It does not force segmentation of ordinary non-terminal saturated heteroatom linkers such as:

- simple ethers;
- simple thioethers;
- simple secondary/tertiary amines.

### 8.2 Definition

After masking all higher-priority segments, a remaining atom is a terminal heteroatom candidate if:

1. it is a heteroatom;
2. it has exactly one heavy-atom neighbor in the remaining segmentation graph;
3. it has not already been assigned to a higher-priority segment.

Examples may include remaining terminal:

- hydroxyl oxygen;
- thiol sulfur;
- amino nitrogen;
- halogen substituent;
- other terminal heteroatom substituents.

### 8.3 Bond labels

The bond connecting a terminal heteroatom segment to the rest of the graph is a candidate cut bond.

---

## 9. Final Bond Label Generation

After all segment candidates have been assigned, generate the final cut-bond label for each bond.

For each bond \(b = (u, v)\):

### 9.1 Non-cut bonds

Label the bond as non-cut if:

1. both atoms belong to the same segment;
2. the bond is internal to a protected segment;
3. both atoms are unassigned background atoms.

### 9.2 Cut bonds

Label the bond as cut if:

1. the two atoms belong to different assigned segments;
2. one atom belongs to an assigned segment and the other belongs to unassigned background skeleton.

In other words:

\[
y_b^{\mathrm{cut}} =
\begin{cases}
1, & \text{if } b \text{ connects different segments or segment-background} \\
0, & \text{otherwise}
\end{cases}
\]

This matches the training objective: the model predicts whether a bond is a chemically meaningful boundary.

---

## 10. Background Skeleton

Atoms and bonds not assigned to any of the four priority motif classes are treated as background skeleton.

The background skeleton is not necessarily segmented further.

Examples may include:

- saturated carbon chains;
- ordinary alkyl linkers;
- ordinary saturated C-C regions;
- ordinary non-terminal saturated heteroatom linkers not captured by the rule.

This is intentional.

HCMP v1 does not require complete atom partition into conventional functional groups.  
It only requires stable cut-bond labels around salient chemical motifs.

---

## 11. Algorithm Summary

A concise implementation workflow:

```text
Input: RDKit molecule

0. Sanitize molecule and remove explicit hydrogens.

1. Initialize:
   assigned_atoms = empty
   assigned_bonds = empty
   segments = empty

2. Priority 1: ring systems
   - find ring systems;
   - merge fused / bridged / spiro / shared-atom systems;
   - merge only explicitly allowed ring-ring conjugated systems;
   - do not merge ring with non-ring functional groups;
   - assign ring-system segments;
   - mark internal bonds as protected.

3. Priority 2: non-ring unsaturated/conjugated motifs
   - in the unassigned non-ring region, find multiple-bond seeds;
   - expand through RDKit-conjugated bonds;
   - take connected components;
   - keep components containing at least one multiple bond;
   - assign these as segments;
   - mark internal bonds as protected.

4. Priority 3: heteroatom connected clusters
   - in the remaining unassigned region, collect heteroatoms;
   - build heteroatom-induced subgraph;
   - find connected components;
   - assign each component as a segment;
   - mark internal bonds as protected.

5. Priority 4: terminal heteroatoms
   - in the remaining unassigned region, find terminal heteroatoms;
   - assign each terminal heteroatom as a segment.

6. Final cut-label generation:
   For every bond:
   - same segment -> non-cut;
   - different segments -> cut;
   - assigned segment to background -> cut;
   - background to background -> non-cut.

Output:
   - segment assignments;
   - bond-level cut labels.
```

---

## 12. Notes on RDKit Conjugation

RDKit's `GetIsConjugated()` should be treated as a graph-chemical heuristic.

It is useful for HCMP v1 because it is:

- deterministic;
- mature;
- chemically plausible;
- easy to implement;
- consistent across molecules.

However, it should not be described as a physically exact conjugation detector.

Recommended wording:

> RDKit-perceived conjugated bonds.

or:

> graph-conjugated motifs as perceived by RDKit's sanitization and conjugation perception.

Important practical observations:

1. An isolated double bond or isolated carbonyl may not be marked as conjugated by RDKit.
2. Therefore, Priority 2 must start from all non-ring multiple bonds, not only from RDKit-conjugated bonds.
3. Amide-like and ester-like carbonyl systems are often captured naturally because RDKit may mark the relevant single bonds as conjugated.
4. Allene-like adjacent multiple-bond systems may be marked as conjugated by RDKit and should be preserved as unsaturated motifs in HCMP v1.
5. Ring-functional-group conjugation perceived by RDKit should not merge ring and non-ring segments, because HCMP v1 intentionally separates ring systems from non-ring functional motifs.

---

## 13. Sanity-Check Molecule Set

Before using the rule for large-scale pretraining, run visualization sanity checks on representative molecules.

Suggested examples:

```python
tests = {
    "ethylene": "C=C",
    "butadiene": "C=CC=C",
    "allene": "C=C=C",
    "propyne": "CC#C",
    "acetone": "CC(=O)C",
    "acetamide": "CC(=O)N",
    "methyl_acetate": "CC(=O)OC",
    "enone": "CC=CC(=O)C",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "aniline": "Nc1ccccc1",
    "phenol": "Oc1ccccc1",
    "styrene": "C=Cc1ccccc1",
    "stilbene": "c1ccccc1C=Cc2ccccc2",
    "biphenyl": "c1ccccc1-c2ccccc2",
    "pyridine": "c1ccncc1",
    "thiophene": "c1ccsc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "decalin": "C1CCC2CCCCC2C1",
}
```

For each molecule, visualize:

- Priority 1 ring segments;
- Priority 2 unsaturated/conjugated motifs;
- Priority 3 heteroatom clusters;
- Priority 4 terminal heteroatoms;
- final cut bonds.

The rule is acceptable for HCMP v1 if the generated labels are stable, interpretable, and mostly aligned with the intended conservative motif-boundary philosophy.

---

## 14. Final One-Sentence Definition

HCMP v1 segmentation is a priority-based conservative salient-motif segmentation rule that first preserves ring systems, then extracts non-ring multiple-bond-seeded RDKit-conjugated motifs, then identifies heteroatom connected clusters and terminal heteroatoms, and finally labels bonds connecting these motifs to each other or to background skeleton as cut bonds.
