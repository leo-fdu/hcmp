# HCMP v1 Project Design

## Hierarchical Conservative Molecular Pretraining

**Project name:** HCMP v1  
**Full name:** Hierarchical Conservative Molecular Pretraining  
**Subtitle:** From local chemical syntax to global physicochemical directions and scaffold geometry

---

## 1. Core Motivation

This project aims to design a conservative molecular pretraining framework for molecular graph representation learning.

The central goal is not to inject as much auxiliary information as possible during pretraining. Instead, the goal is to construct a set of robust, chemically meaningful, low-noise, interpretable weak supervision signals that help a molecular graph model learn useful representations without relying on noisy downstream labels, experimental property labels, LLM-generated labels, DFT labels, bioactivity proxies, or drug-likeness priors.

The framework is based on the following belief:

> Molecular pretraining should first learn reliable chemical structure and structure-derived physicochemical regularities, rather than absorb strong but noisy property priors.

The current design is a hierarchical curriculum:

\[
\text{atom/bond syntax}
\rightarrow
\text{functional-group-level segmentation}
\rightarrow
\text{global physicochemical ranking + scaffold geometry}
\]

---

## 2. Final High-Level Design

HCMP v1 contains four major losses activated progressively during training:

\[
\mathcal{L}
=
\lambda_1\mathcal{L}_{\text{BERT}}
+
\lambda_2\mathcal{L}_{\text{cut-seg}}
+
\lambda_3\mathcal{L}_{\text{prop-rank}}
+
\lambda_4\mathcal{L}_{\text{scaf-triplet}}
\]

The four components have distinct responsibilities:

| Stage | Level | Loss | Main role |
|---|---:|---|---|
| Stage 1 | atom/bond | atom/bond BERT | Learn local chemical syntax |
| Stage 2 | subgraph | cut-bond segmentation | Learn functional-group-level chemical boundaries |
| Stage 3A | molecule | global property threshold ranking | Learn global physicochemical directions |
| Stage 3B | molecule | scaffold triplet ranking | Learn scaffold-level structural geometry |

The final conceptual summary is:

> HCMP learns local chemical syntax with atom/bond BERT, functional-group-level chemical boundaries with cut-bond segmentation, global physicochemical directions with thresholded scalar descriptor ranking, and scaffold-level structural geometry with expanded-scaffold triplet ranking.

---

## 3. Stage 1: Atom/Bond-Level BERT

### 3.1 Goal

Stage 1 teaches the model local chemical syntax.

This includes:

- atom type;
- formal charge;
- aromaticity;
- hybridization;
- local valence pattern;
- bond type;
- bond aromaticity;
- bond conjugation;
- ring membership;
- local graph context.

This stage is not the core novelty of HCMP. It serves as the stable local representation basis for later stages.

### 3.2 Task

Randomly mask a subset of atoms and bonds, then ask the model to reconstruct their original labels.

Possible losses:

\[
\mathcal{L}_{\text{BERT}}
=
\mathcal{L}_{\text{atom-mask}}
+
\mathcal{L}_{\text{bond-mask}}
\]

### 3.3 Atom masking targets

Possible atom-level prediction targets:

- atomic number;
- formal charge;
- aromaticity;
- hybridization;
- chirality;
- degree;
- implicit/explicit hydrogen count.

### 3.4 Bond masking targets

Possible bond-level prediction targets:

- bond type;
- aromaticity;
- conjugation;
- stereo;
- ring membership.

---

## 4. Stage 2: Functional-Group-Level Chemical Segmentation

### 4.1 Goal

Stage 2 teaches the model mesoscopic chemical organization.

The current design intentionally makes this stage **functional-group-centric** rather than scaffold-centric. This is because scaffold-level structure will be handled separately by the scaffold triplet task in Stage 3B.

The segmentation task should help the model learn:

- which bonds separate functional chemical units;
- which atoms/bonds belong to functional groups;
- where chemical boundaries occur;
- how functional groups attach to carbon skeletons, rings, linkers, and substituents.

### 4.2 Current preferred formulation: cut-bond prediction

Instead of forcing a hard atom partition, HCMP v1 uses a bond-level boundary prediction task.

For each bond \(b\):

\[
y_b^{\text{cut}}
=
\mathbb{1}[b \text{ is a functional segment boundary}]
\]

The model predicts:

\[
\hat{p}_b^{\text{cut}}
=
\sigma(h_b(e_b))
\]

The loss is binary cross entropy:

\[
\mathcal{L}_{\text{cut-seg}}
=
-
\sum_b
\left[
\alpha y_b^{\text{cut}}\log \hat{p}_b^{\text{cut}}
+
(1-y_b^{\text{cut}})\log(1-\hat{p}_b^{\text{cut}})
\right]
\]

where \(\alpha>1\) can be used to handle class imbalance because cut bonds are usually much rarer than non-cut bonds.

### 4.3 Segmentation philosophy

The segmentation rule should not be designed merely to produce visually neat fragments. It should be designed to teach the model functional chemical units.

The key principle is:

> Preserve functional group integrity; cut at chemically meaningful boundaries between functional units.

### 4.4 Bonds that should usually be cut

Candidate cut bonds include:

- bond between an aromatic/alkyl scaffold and a heteroatom-containing substituent;
- bond between an alkyl linker and an aromatic ring;
- bond between a carbon skeleton and a terminal functional substituent;
- bond connecting a side chain to a core region;
- bond separating two chemically distinct functional units.

### 4.5 Bonds that should usually not be cut

The segmentation should avoid cutting inside chemically coherent functional groups.

Usually do not cut:

- carbonyl C=O;
- amide C-N;
- ester C-O inside the ester group;
- carboxyl C-O / C=O inside the carboxyl group;
- nitro N-O;
- sulfonamide S-N / S=O inside the sulfonamide group;
- aromatic ring bonds;
- conjugated bonds inside one functional motif.

### 4.6 Relation to functional group distance

Earlier versions considered a functional-group count distance in the molecule-level triplet task. HCMP v1 removes this component.

Reason:

> Functional-group-level semantics should be learned by Stage 2 segmentation rather than duplicated in the molecule-level distance.

This makes the division of responsibilities cleaner:

- Stage 2 learns functional group boundaries and local functional units.
- Stage 3A learns global physicochemical directions.
- Stage 3B learns scaffold geometry.

---

## 5. Stage 3A: Global Property Threshold Pairwise Ranking

### 5.1 Goal

Stage 3A teaches the model global molecular physicochemical directions.

This task is needed because scaffold triplet distance alone mainly teaches structural geometry. It does not directly teach molecular-level tendencies such as:

- more hydrophobic vs less hydrophobic;
- more polar vs less polar;
- stronger charge separation vs weaker charge separation;
- more conjugated/aromatic vs less conjugated/aromatic;
- more flexible vs more rigid;
- higher refractivity/polarizability-related tendency.

The descriptor task should use scalar whole-molecule descriptors rather than VSA descriptor bins.

### 5.2 Why avoid VSA descriptors in HCMP v1?

VSA descriptors are useful, but they describe distributions of atomic surface areas across local property bins. That makes them partly local/regional rather than purely global.

Since Stage 2 already handles functional group and local subgraph semantics, Stage 3A should focus on scalar global properties.

Therefore, HCMP v1 avoids VSA descriptors as core property-ranking targets.

### 5.3 Descriptor ranking rather than regression

HCMP v1 does not directly regress descriptor values.

Instead, for descriptor \(q_k\), sample two molecules \(G_i\) and \(G_j\):

\[
\Delta q_k = q_k(G_i)-q_k(G_j)
\]

If the difference is too small, the pair is ignored for this descriptor.

If the difference is large enough:

\[
|\Delta q_k| \ge \tau_k
\]

then define ranking label:

\[
y_{ij}^{(k)} = \operatorname{sign}(q_k(G_i)-q_k(G_j))
\]

The model produces a descriptor-specific score:

\[
s_k(G)=h_k(f(G))
\]

and uses margin ranking loss:

\[
\mathcal{L}_{k}
=
\max
\left(
0,
 m_k
-
y_{ij}^{(k)}[s_k(G_i)-s_k(G_j)]
\right)
\]

The full descriptor ranking loss is:

\[
\mathcal{L}_{\text{prop-rank}}
=
\sum_k \alpha_k \mathcal{L}_k
\]

### 5.4 Threshold strategy

A descriptor-specific threshold is used to reduce noise.

The preferred first version is top-30% hard thresholding.

For each descriptor \(q_k\), randomly sample many molecule pairs and compute:

\[
|q_k(G_i)-q_k(G_j)|
\]

Then choose:

\[
\tau_k = Q_{0.70}(|\Delta q_k|)
\]

Only pairs with descriptor difference in the largest 30% are used.

This matches the design philosophy:

> Do not trust the precise descriptor value; trust only the direction when the difference is sufficiently large.

### 5.5 Optional soft threshold

A soft threshold can be used later:

\[
w_{ij}^{(k)}
=
\sigma
\left(
\frac{|\Delta q_k|-\tau_k}{T_k}
\right)
\]

Then:

\[
\mathcal{L}_{k}
=
w_{ij}^{(k)}
\max
\left(
0,
 m_k
-
y_{ij}^{(k)}[s_k(G_i)-s_k(G_j)]
\right)
\]

For HCMP v1, the simpler top-30% hard threshold is preferred.

---

## 6. Recommended Global Descriptor Set

HCMP v1 avoids trivial size descriptors such as atom count, heavy atom count, molecular weight, and exact molecular weight.

The preferred descriptors are scalar whole-molecule or normalized global summary descriptors.

### 6.1 Hydrophobicity / lipophilicity

Recommended:

- MolLogP;
- MeanCrippenLogP;
- HydrophobicContributionFraction.

These describe global hydrophobic/lipophilic tendency.

### 6.2 Refractivity / polarizability-related tendency

Recommended:

- MolMR;
- MeanCrippenMR;
- HighMRContributionFraction.

These describe molar refractivity and atom-level refractivity contribution tendencies.

### 6.3 Polarity / hydrogen bonding

Recommended:

- TPSA;
- TPSA_over_LabuteASA;
- HBA_fraction;
- HBD_fraction.

These describe polar surface, hydrogen-bonding capacity, and normalized polarity.

### 6.4 Charge / electronic distribution

Recommended:

- MeanAbsGasteigerCharge;
- GasteigerChargeVariance;
- NegativeChargeFraction;
- PositiveChargeFraction;
- ChargeQ90MinusQ10.

These describe charge separation, charge heterogeneity, and positive/negative charge tendency.

### 6.5 Electrotopological state

Recommended:

- MeanAbsEState;
- EStateVariance;
- HighEStateFraction;
- LowEStateFraction.

These describe electronic/topological environment heterogeneity.

### 6.6 Conjugation / aromaticity / rigidity

Recommended:

- ConjugatedBondFraction;
- AromaticAtomFraction;
- AromaticBondFraction;
- RotatableBondFraction.

These describe conjugation, aromaticity, planarity-related structure, and flexibility.

### 6.7 Descriptors excluded from HCMP v1

Excluded from core property ranking:

- MolWt;
- ExactMolWt;
- HeavyAtomCount;
- AtomCount;
- raw RingCount;
- raw RotatableBondCount;
- raw HBA/HBD count without normalization;
- pKa;
- logS;
- QED;
- synthetic accessibility score;
- HOMO/LUMO or DFT-derived labels.

Reason:

- some are too weak and become counting shortcuts;
- some are noisy model-derived properties;
- some inject drug-likeness or task-specific bias;
- some are closer to powerful pretraining than conservative pretraining.

---

## 7. Stage 3B: Scaffold Triplet Distance

### 7.1 Goal

Stage 3B teaches scaffold-level structural geometry.

This task is different from segmentation. Segmentation teaches the model where functional chemical units are inside one molecule. Scaffold triplet ranking teaches the model how molecules relate to each other in scaffold space.

### 7.2 Why remove functional group distance?

Earlier versions used a combined scaffold-functional-group distance:

\[
D_{\text{total}}
=
\lambda D_{\text{scaffold}}+(1-\lambda)D_{\text{FG}}
\]

HCMP v1 removes \(D_{\text{FG}}\) from triplet supervision.

Reason:

> Functional-group-level semantics should be learned by segmentation. Molecule-level triplet distance should focus on scaffold geometry.

Thus:

\[
D_{\text{triplet}} = D_{\text{scaffold}}
\]

### 7.3 Expanded scaffold extraction

The scaffold extraction rule keeps:

- all carbon atoms;
- all ring atoms;
- all ring bonds;
- all multiple bonds and their endpoint atoms;
- deterministic shortest paths connecting retained components.

This produces an expanded scaffold that captures carbon skeletons, ring systems, multiple-bond systems, and the connecting paths between them.

### 7.4 Scaffold similarity

Given two expanded scaffold graphs, HCMP computes iterative bond-based MCS similarity.

Each round:

1. runs exact atom-type and exact bond-order MCS on the currently unmasked scaffold subgraphs;
2. selects a deterministic match;
3. masks matched atoms and bonds;
4. repeats for a fixed number of rounds.

The scaffold similarity is:

\[
S_{\text{scaffold}}
=
\frac{\text{matched bond total}}{\max(\#\text{bonds}_a,\#\text{bonds}_b)}
\]

The scaffold distance is:

\[
D_{\text{scaffold}} = 1-S_{\text{scaffold}}
\]

### 7.5 Triplet construction

For anchor molecule \(G_a\), positive candidate \(G_p\), and negative candidate \(G_n\), keep the triplet only when:

\[
D_{\text{scaffold}}(G_a,G_p)+\tau_D
<
D_{\text{scaffold}}(G_a,G_n)
\]

This threshold removes ambiguous triplets.

The model should satisfy:

\[
d_\theta(G_a,G_p)<d_\theta(G_a,G_n)
\]

Loss:

\[
\mathcal{L}_{\text{scaf-triplet}}
=
\max
\left(
0,
 m_D
+d_\theta(G_a,G_p)
-d_\theta(G_a,G_n)
\right)
\]

### 7.6 Semantic role of scaffold distance

The scaffold distance should be described carefully.

It is not:

- true molecular distance;
- physicochemical distance;
- electronic distance;
- biological similarity;
- experimental property distance.

It is:

> an interpretable, representation-independent, operational scaffold-level structural distance based on expanded scaffold graph overlap.

---

## 8. Training Curriculum

HCMP v1 uses progressive loss activation.

| Phase | Active losses | Purpose |
|---|---|---|
| Phase 1 | BERT only | stabilize local chemical syntax |
| Phase 2 | BERT + cut segmentation | add functional-group-level organization |
| Phase 3 | BERT + cut segmentation + property ranking | add global physicochemical directions |
| Phase 4 | full loss with scaffold triplet | add scaffold-level structural geometry |

The curriculum prevents molecule-level losses from dominating before the model learns stable local/subgraph-level chemistry.

A possible schedule:

\[
w_1(t)>0 \quad \text{from the beginning}
\]

\[
w_2(t), w_3(t), w_4(t)
\quad \text{activate progressively}
\]

---

## 9. Minimal Ablation Plan

The key ablation should determine whether each stage adds useful information.

| Model | Losses |
|---|---|
| A | no pretraining |
| B | BERT only |
| C | BERT + cut segmentation |
| D | BERT + cut segmentation + property ranking |
| E | BERT + cut segmentation + scaffold triplet |
| F | full HCMP |

Main questions:

1. Does BERT help over no pretraining?
2. Does segmentation add value beyond BERT?
3. Does global descriptor ranking add physicochemical information beyond segmentation?
4. Does scaffold triplet improve OOD/scaffold-split robustness?
5. Does the full model outperform each partial version?

---

## 10. Evaluation Strategy

HCMP should not be evaluated only by average benchmark performance.

Important evaluation settings:

- random split;
- scaffold split;
- Butina split;
- expanded-scaffold-distance split;
- possibly low-data, medium-data, and larger-data regimes.

Important evaluation questions:

1. Does HCMP improve robustness under OOD-like splits?
2. Does it improve scaffold/generalization-sensitive tasks more than random split tasks?
3. Does it reduce validation-test instability?
4. Does it improve intermediate-data regimes more than extremely tiny or very large regimes?
5. Does scaffold triplet help specifically under scaffold-shifted evaluation?
6. Does property ranking improve physicochemical property prediction tasks?

---

## 11. Current Design Position

The current HCMP v1 design is:

1. Stage 1 learns atom/bond chemical syntax through BERT.
2. Stage 2 learns functional-group-level boundaries through cut-bond segmentation.
3. Stage 3A learns scalar global physicochemical directions through thresholded descriptor ranking.
4. Stage 3B learns scaffold-level structural geometry through expanded-scaffold triplet ranking.
5. Functional group distance is removed from the triplet task because functional group semantics are assigned to segmentation.
6. VSA descriptors are removed from the core descriptor set because Stage 3A should focus on scalar global properties.
7. Trivial size descriptors are removed to avoid weak counting shortcuts.

---

## 12. One-Sentence Summary

> HCMP v1 is a hierarchical conservative molecular pretraining framework that learns local atom/bond syntax, functional-group-level chemical boundaries, global scalar physicochemical directions, and scaffold-level structural geometry through progressively activated weak supervision tasks.
