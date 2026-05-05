# HCMP v1 Descriptor Summary

## Purpose

This file summarizes every descriptor discussed for the **global property threshold pairwise ranking** task in HCMP v1.

The design goal is to use scalar whole-molecule or normalized global descriptors that are:

- chemically meaningful;
- structure-derived;
- relatively mature or interpretable;
- lower-noise than experimental labels or learned property predictors;
- not trivial size/counting shortcuts;
- suitable for thresholded pairwise ranking rather than direct regression.

The descriptor task should teach global physicochemical directions, not local fragment recognition. Functional-group-level semantics are handled by Stage 2 segmentation.

---

## 1. Descriptor Ranking Setup

For descriptor \(q_k\), sample two molecules \(G_i\) and \(G_j\):

\[
\Delta q_k = q_k(G_i)-q_k(G_j)
\]

If:

\[
|\Delta q_k| < \tau_k
\]

the pair is ignored.

If:

\[
|\Delta q_k| \ge \tau_k
\]

then define:

\[
y_{ij}^{(k)}
=
\operatorname{sign}(q_k(G_i)-q_k(G_j))
\]

The model predicts descriptor-specific scalar scores:

\[
s_k(G)=h_k(f(G))
\]

and uses ranking loss:

\[
\mathcal{L}_k
=
\max
\left(
0,
m_k
-
y_{ij}^{(k)}[s_k(G_i)-s_k(G_j)]
\right)
\]

Recommended threshold:

\[
\tau_k=Q_{0.70}(|\Delta q_k|)
\]

That means only the top 30% most separated molecule pairs for each descriptor are used.

---

# 2. Hydrophobicity / Lipophilicity Descriptors

## 2.1 MolLogP

### What it describes

MolLogP describes whole-molecule hydrophobicity / lipophilicity.

Higher MolLogP usually indicates:

- more hydrophobic character;
- stronger preference for nonpolar/lipid-like environments;
- lower polarity tendency;
- potentially higher membrane permeability, although this is not guaranteed.

### How it is calculated

RDKit computes MolLogP using the Wildman–Crippen atom contribution scheme.

Each atom is assigned an atom type based on its element and local chemical environment. Each atom type has an empirical logP contribution.

The molecule-level value is the sum of atomic contributions:

\[
\operatorname{MolLogP}(G)
=
\sum_{i \in atoms} c_i^{\log P}
\]

where \(c_i^{\log P}\) is the Crippen logP contribution of atom \(i\).

### Why it is useful for HCMP

MolLogP is a mature scalar global descriptor. It is suitable for property threshold ranking because HCMP does not force the model to reproduce the exact empirical value; it only asks for the correct direction when the difference is large.

### Risk

MolLogP is still an empirical estimate, not an experimental logP. It can be affected by protonation state, tautomerism, and domain limitations.

### Recommendation

**Use as a core descriptor.**

---

## 2.2 MeanCrippenLogP

### What it describes

MeanCrippenLogP describes average atomic hydrophobic contribution.

It is a normalized version of MolLogP that reduces size dependence.

### How it is calculated

First compute Crippen logP atom contributions \(c_i^{\log P}\), then average over atoms:

\[
\operatorname{MeanCrippenLogP}(G)
=
\frac{1}{N}
\sum_{i=1}^{N}
c_i^{\log P}
\]

where \(N\) is the number of atoms included in the calculation, usually heavy atoms or all atoms depending on implementation.

### Why it is useful for HCMP

MolLogP is a sum, so larger molecules can naturally have larger absolute values. MeanCrippenLogP asks instead:

> On average, how hydrophobic are the atoms in this molecule?

This better reflects global hydrophobic tendency rather than molecular size.

### Risk

Averaging may hide strong local hydrophobic regions if the rest of the molecule is polar.

### Recommendation

**Use as a core normalized hydrophobicity descriptor.**

---

## 2.3 HydrophobicContributionFraction

### What it describes

HydrophobicContributionFraction describes the fraction of atoms with positive or sufficiently high hydrophobic contribution.

It captures whether a large proportion of the molecule is hydrophobic.

### How it is calculated

Using Crippen logP atom contributions:

\[
\operatorname{HydrophobicContributionFraction}(G)
=
\frac{
\#\{i:c_i^{\log P}>\epsilon_{\log P}\}
}{
N
}
\]

A weighted version can also be used:

\[
\operatorname{HydrophobicContributionFraction}_{w}(G)
=
\frac{
\sum_i w_i \mathbb{1}[c_i^{\log P}>\epsilon_{\log P}]
}{
\sum_i w_i
}
\]

where \(w_i\) may be an atomic surface contribution if available.

### Threshold choice

A simple first choice is \(\epsilon_{\log P}=0\). A more data-adaptive version uses a global quantile of atomic Crippen logP contributions.

### Why it is useful for HCMP

It converts local atomic hydrophobic contributions into one scalar global summary.

### Risk

It discards magnitude information and keeps only sign/threshold membership.

### Recommendation

**Use as a core hydrophobic surface/composition descriptor.**

---

# 3. Refractivity / Polarizability-Related Descriptors

## 3.1 MolMR

### What it describes

MolMR describes whole-molecule molar refractivity.

Molar refractivity is related to:

- molecular volume;
- electronic polarizability;
- heavy atoms;
- aromatic systems;
- halogens and sulfur-containing fragments.

### How it is calculated

RDKit computes MolMR using Wildman–Crippen atom contributions.

Each atom receives an empirical MR contribution \(c_i^{MR}\), and the whole-molecule value is:

\[
\operatorname{MolMR}(G)
=
\sum_i c_i^{MR}
\]

### Why it is useful for HCMP

MolMR gives a scalar global property related to polarizability and refractivity, which are not captured well by purely local BERT or cut-bond segmentation.

### Risk

MolMR is strongly correlated with molecular size because it is additive.

### Recommendation

**Use, but pair it with normalized MR descriptors.**

---

## 3.2 MeanCrippenMR

### What it describes

MeanCrippenMR describes average atomic molar refractivity contribution.

### How it is calculated

\[
\operatorname{MeanCrippenMR}(G)
=
\frac{1}{N}
\sum_{i=1}^{N}
c_i^{MR}
\]

where \(c_i^{MR}\) is the Crippen MR contribution of atom \(i\).

### Why it is useful for HCMP

It reduces the size shortcut in raw MolMR and asks whether the molecule is made of atoms/environments with higher refractivity contribution on average.

### Risk

It may underrepresent the effect of large polarizable substructures if averaged over many atoms.

### Recommendation

**Use as a normalized companion to MolMR.**

---

## 3.3 HighMRContributionFraction

### What it describes

HighMRContributionFraction describes the fraction of atoms with high molar refractivity contribution.

It can capture the presence of polarizable atom environments.

### How it is calculated

\[
\operatorname{HighMRContributionFraction}(G)
=
\frac{
\#\{i:c_i^{MR}>\epsilon_{MR}\}
}{
N
}
\]

A robust threshold can be chosen as a quantile over atom-level MR contributions in the pretraining dataset.

For example:

\[
\epsilon_{MR}=Q_{0.75}(c^{MR})
\]

### Why it is useful for HCMP

It compresses refractivity-related local information into a scalar global proportion.

### Risk

Threshold choice matters.

### Recommendation

**Optional but useful.**

---

# 4. Polarity / Hydrogen-Bonding Descriptors

## 4.1 TPSA

### What it describes

TPSA means topological polar surface area.

It describes the polar surface contribution of a molecule based on topological fragments, especially polar heteroatom environments.

High TPSA usually indicates:

- stronger polarity;
- more polar surface;
- stronger hydrogen-bonding potential;
- lower passive membrane permeability tendency.

### How it is calculated

TPSA is calculated from 2D molecular topology rather than a 3D conformer.

The general form is:

\[
\operatorname{TPSA}(G)
=
\sum_{j \in polar\ fragments}
a_j^{polar}
\]

where \(a_j^{polar}\) is a tabulated polar surface area contribution for a polar atom or polar fragment type.

### Why it is useful for HCMP

TPSA is a scalar global descriptor with strong physicochemical meaning. It is more informative than simply counting heteroatoms.

### Risk

TPSA is still topological and fragment-based. It is not actual solvent-accessible polar surface area from 3D simulation.

### Recommendation

**Use as a core descriptor.**

---

## 4.2 TPSA_over_LabuteASA

### What it describes

TPSA_over_LabuteASA describes the fraction of approximate molecular surface that is polar.

It is a normalized polarity descriptor.

### How it is calculated

\[
\operatorname{TPSA\_over\_LabuteASA}(G)
=
\frac{
\operatorname{TPSA}(G)
}{
\operatorname{LabuteASA}(G)+\epsilon
}
\]

where \(\epsilon\) is a small constant for numerical stability.

### Why it is useful for HCMP

Raw TPSA increases with molecule size. This normalized descriptor asks:

> What fraction of the molecular surface is polar?

This is more useful than raw polar area alone.

### Risk

LabuteASA is itself approximate.

### Recommendation

**Use as a core normalized polarity descriptor.**

---

## 4.3 HBA_fraction

### What it describes

HBA_fraction describes the density of hydrogen-bond acceptors.

### How it is calculated

First compute the number of hydrogen-bond acceptors:

\[
\operatorname{HBA}(G)
\]

Then normalize by heavy atom count:

\[
\operatorname{HBA\_fraction}(G)
=
\frac{
\operatorname{HBA}(G)
}{
N_{\text{heavy}}+\epsilon
}
\]

### What counts as HBA?

HBA is rule-based. It usually includes heteroatoms such as oxygen and nitrogen in environments where they can accept hydrogen bonds, while excluding environments where acceptor ability is suppressed, such as some positively charged or amide-like cases.

### Why it is useful for HCMP

It gives a stable scalar signal for hydrogen-bond acceptor density.

### Risk

The raw HBA count is too count-like, so the fraction is preferred.

### Recommendation

**Use normalized HBA_fraction, not raw HBA count as a core descriptor.**

---

## 4.4 HBD_fraction

### What it describes

HBD_fraction describes the density of hydrogen-bond donors.

### How it is calculated

First compute hydrogen-bond donor count:

\[
\operatorname{HBD}(G)
\]

Then normalize:

\[
\operatorname{HBD\_fraction}(G)
=
\frac{
\operatorname{HBD}(G)
}{
N_{\text{heavy}}+\epsilon
}
\]

### What counts as HBD?

HBD is rule-based. It usually includes N-H and O-H donor environments and excludes atoms that cannot donate hydrogen bonds.

### Why it is useful for HCMP

It gives a stable scalar signal for hydrogen-bond donor density.

### Risk

The signal is relatively simple and may overlap with segmentation.

### Recommendation

**Use with moderate or low weight.**

---

# 5. Surface / Size-Normalization Descriptor

## 5.1 LabuteASA

### What it describes

LabuteASA is an approximate molecular surface area descriptor.

It describes molecular surface/size more geometrically than molecular weight.

### How it is calculated

LabuteASA estimates atomic surface area contributions from molecular topology and atom types, then sums approximate surface contributions:

\[
\operatorname{LabuteASA}(G)
=
\sum_i ASA_i^{approx}
\]

It is not a full 3D solvent-accessible surface simulation.

### Why it is useful for HCMP

Raw LabuteASA itself may be size-related, but it is useful as a normalization denominator, especially for TPSA_over_LabuteASA.

### Risk

As a standalone descriptor, it may behave partly like a size signal.

### Recommendation

**Use primarily as a normalization denominator. Use raw LabuteASA cautiously.**

---

# 6. Charge / Electronic Distribution Descriptors

These descriptors are based on atom-level partial charges, such as Gasteiger/PEOE-style partial charges.

They are not quantum-mechanical charges, but they provide useful low-cost, structure-derived electronic tendency signals.

Let \(q_i\) be the partial charge assigned to atom \(i\).

## 6.1 MeanAbsGasteigerCharge

### What it describes

MeanAbsGasteigerCharge describes average charge separation strength.

It captures polarity/electrostatic heterogeneity better than mean charge, because positive and negative charges do not cancel.

### How it is calculated

\[
\operatorname{MeanAbsGasteigerCharge}(G)
=
\frac{1}{N}
\sum_i |q_i|
\]

### Why it is useful for HCMP

This is one of the best scalar descriptors for teaching a weak notion of electronic polarity.

### Risk

Gasteiger charges are approximate and can fail or be unreliable for unusual molecules.

### Recommendation

**Use as a core electronic descriptor.**

---

## 6.2 GasteigerChargeVariance

### What it describes

GasteigerChargeVariance describes how spread out the partial charges are.

### How it is calculated

\[
\operatorname{GasteigerChargeVariance}(G)
=
\frac{1}{N}
\sum_i(q_i-\bar{q})^2
\]

where:

\[
\bar{q}=\frac{1}{N}\sum_i q_i
\]

### Why it is useful for HCMP

It captures intramolecular electrostatic heterogeneity.

### Risk

Sensitive to charge estimation quality.

### Recommendation

**Use as a core electronic descriptor.**

---

## 6.3 NegativeChargeFraction

### What it describes

NegativeChargeFraction describes the fraction of atoms with significantly negative partial charge.

### How it is calculated

\[
\operatorname{NegativeChargeFraction}(G)
=
\frac{
\#\{i:q_i<-\epsilon_q\}
}{
N
}
\]

A surface-weighted version is possible:

\[
\operatorname{NegativeChargeFraction}_w(G)
=
\frac{
\sum_i w_i\mathbb{1}[q_i<-\epsilon_q]
}{
\sum_i w_i
}
\]

### Why it is useful for HCMP

It teaches the model whether a molecule contains a substantial amount of electron-rich / negatively polarized atom environments.

### Risk

The threshold \(\epsilon_q\) must be chosen carefully. A very small threshold may classify noise as signal.

### Recommendation

**Use with a conservative threshold.**

---

## 6.4 PositiveChargeFraction

### What it describes

PositiveChargeFraction describes the fraction of atoms with significantly positive partial charge.

### How it is calculated

\[
\operatorname{PositiveChargeFraction}(G)
=
\frac{
\#\{i:q_i>\epsilon_q\}
}{
N
}
\]

### Why it is useful for HCMP

It complements NegativeChargeFraction and captures electron-poor / positively polarized regions.

### Risk

Same as NegativeChargeFraction.

### Recommendation

**Use with a conservative threshold.**

---

## 6.5 ChargeQ90MinusQ10

### What it describes

ChargeQ90MinusQ10 describes robust charge distribution width.

It is a robust alternative to max-min charge range.

### How it is calculated

\[
\operatorname{ChargeQ90MinusQ10}(G)
=
Q_{0.90}(q_i)-Q_{0.10}(q_i)
\]

### Why it is useful for HCMP

It captures charge heterogeneity while reducing sensitivity to one extreme atom.

### Risk

For very small molecules, quantiles may be unstable.

### Recommendation

**Use as a robust charge-spread descriptor.**

---

# 7. Electrotopological State Descriptors

EState means electrotopological state.

It combines atom-level electronic character with molecular topology. Let \(E_i\) be the EState index of atom \(i\).

## 7.1 MeanAbsEState

### What it describes

MeanAbsEState describes the average strength of atom-level electrotopological states.

### How it is calculated

\[
\operatorname{MeanAbsEState}(G)
=
\frac{1}{N}
\sum_i |E_i|
\]

### Why it is useful for HCMP

It provides a scalar summary of electronic/topological environment intensity.

### Risk

EState is an empirical topological-electronic index, not a direct physical observable.

### Recommendation

**Use as a core electronic/topological descriptor.**

---

## 7.2 EStateVariance

### What it describes

EStateVariance describes heterogeneity of electrotopological environments across the molecule.

### How it is calculated

\[
\operatorname{EStateVariance}(G)
=
\frac{1}{N}
\sum_i(E_i-\bar{E})^2
\]

where:

\[
\bar{E}=\frac{1}{N}\sum_i E_i
\]

### Why it is useful for HCMP

It captures whether a molecule contains diverse electronic/topological atom environments.

### Risk

May overlap with functional group segmentation, but the global variance form still provides molecule-level information.

### Recommendation

**Use as a core descriptor.**

---

## 7.3 HighEStateFraction

### What it describes

HighEStateFraction describes the fraction of atoms with high EState values.

### How it is calculated

\[
\operatorname{HighEStateFraction}(G)
=
\frac{
\#\{i:E_i>\epsilon_E^{high}\}
}{
N
}
\]

A robust threshold can be selected from the atom-level EState distribution in the pretraining dataset, for example:

\[
\epsilon_E^{high}=Q_{0.75}(E)
\]

### Why it is useful for HCMP

It captures whether high-EState atom environments are common in the molecule.

### Risk

Threshold choice matters.

### Recommendation

**Optional but useful.**

---

## 7.4 LowEStateFraction

### What it describes

LowEStateFraction describes the fraction of atoms with low EState values.

### How it is calculated

\[
\operatorname{LowEStateFraction}(G)
=
\frac{
\#\{i:E_i<\epsilon_E^{low}\}
}{
N
}
\]

where a data-driven threshold may be:

\[
\epsilon_E^{low}=Q_{0.25}(E)
\]

### Why it is useful for HCMP

It complements HighEStateFraction and captures the presence of low-EState environments.

### Risk

Threshold choice matters.

### Recommendation

**Optional but useful.**

---

# 8. Conjugation / Aromaticity / Rigidity Descriptors

These descriptors are graph-derived scalar ratios rather than raw counts. They are intended to avoid trivial size shortcuts.

## 8.1 ConjugatedBondFraction

### What it describes

ConjugatedBondFraction describes the fraction of bonds participating in conjugation.

It is directly related to:

- conjugated systems;
- electron delocalization;
- planar unsaturated structure;
- aromatic and non-aromatic conjugated motifs.

### How it is calculated

For each bond \(b\), RDKit can mark whether the bond is conjugated.

\[
\operatorname{ConjugatedBondFraction}(G)
=
\frac{
\#\{b:b\text{ is conjugated}\}
}{
\#\text{bonds}+\epsilon
}
\]

### Why it is useful for HCMP

This is one of the most directly useful descriptors for teaching the model whether a molecule has significant conjugated structure.

### Risk

It may overlap with atom/bond BERT and segmentation, but the molecule-level ratio gives a global direction.

### Recommendation

**Use as a core conjugation descriptor.**

---

## 8.2 AromaticAtomFraction

### What it describes

AromaticAtomFraction describes the fraction of atoms that are aromatic.

### How it is calculated

\[
\operatorname{AromaticAtomFraction}(G)
=
\frac{
\#\{i:i\text{ is aromatic}\}
}{
N+\epsilon
}
\]

### Why it is useful for HCMP

It captures global aromatic character without using raw aromatic ring count.

### Risk

It is still a relatively simple graph-derived property.

### Recommendation

**Use.**

---

## 8.3 AromaticBondFraction

### What it describes

AromaticBondFraction describes the fraction of bonds that are aromatic.

### How it is calculated

\[
\operatorname{AromaticBondFraction}(G)
=
\frac{
\#\{b:b\text{ is aromatic}\}
}{
\#\text{bonds}+\epsilon
}
\]

### Why it is useful for HCMP

It complements AromaticAtomFraction and describes aromatic bonding density.

### Risk

Highly correlated with AromaticAtomFraction.

### Recommendation

**Use either AromaticAtomFraction or AromaticBondFraction first; include both only if ablation shows value.**

---

## 8.4 RotatableBondFraction

### What it describes

RotatableBondFraction describes molecular flexibility normalized by bond count.

### How it is calculated

First compute the number of rotatable bonds:

\[
\operatorname{RotBonds}(G)
\]

Then normalize:

\[
\operatorname{RotatableBondFraction}(G)
=
\frac{
\operatorname{RotBonds}(G)
}{
\#\text{bonds}+\epsilon
}
\]

A rotatable bond is usually a non-ring single bond between non-terminal atoms, with special exclusions such as amide-like bonds depending on the rule implementation.

### Why it is useful for HCMP

It teaches global flexibility/rigidity without simply counting the number of rotatable bonds.

### Risk

It is not an electronic descriptor.

### Recommendation

**Use if flexibility is considered important; otherwise keep optional.**

---

# 9. Descriptors Mentioned but Not Recommended for HCMP v1 Core

## 9.1 MolWt / ExactMolWt

### How calculated

Molecular weight is the sum of atomic masses:

\[
\operatorname{MolWt}(G)
=
\sum_i m_i
\]

ExactMolWt uses exact isotope masses.

### Why excluded

Too close to atom-count/element-count shortcut. It does not directly teach the desired physicochemical semantics.

---

## 9.2 HeavyAtomCount / AtomCount

### How calculated

\[
\operatorname{HeavyAtomCount}(G)
=
\#\{i: Z_i>1\}
\]

### Why excluded

Too trivial. The model can learn this by counting atoms.

---

## 9.3 Raw RingCount

### How calculated

Graph ring perception gives number of rings.

### Why excluded

It is a simple structural count. AromaticAtomFraction or scaffold triplet distance is more aligned with HCMP v1.

---

## 9.4 Raw HBA / HBD counts

### Why excluded as core raw descriptors

Raw counts are size-dependent. Use HBA_fraction and HBD_fraction instead.

---

## 9.5 pKa

### Why excluded

pKa is chemically important but difficult as a conservative weak signal.

Problems:

- acid vs base definition;
- site-specific micro-pKa vs molecule-level macro-pKa;
- protonation state;
- tautomerism;
- solvent and condition dependence;
- predictor/model dependence;
- not a simple stable RDKit scalar descriptor like MolLogP.

### Recommendation

Do not use in HCMP v1 core. Consider as a future optional experiment.

---

## 9.6 logS

### Why excluded

Aqueous solubility usually requires learned or empirical models and is strongly affected by ionization, crystal packing, and experimental conditions.

It is too noisy for the conservative core.

---

## 9.7 QED / SA score

### Why excluded

These inject drug-likeness or synthetic-accessibility priors. They are closer to task-biased or medicinal-chemistry-biased pretraining.

---

## 9.8 HOMO / LUMO / DFT labels

### Why excluded

They require quantum chemistry or ML predictors and represent high-information powerful pretraining, not conservative structure-derived weak supervision.

---

# 10. Final Recommended Descriptor Set for HCMP v1

## Core set

```python
GLOBAL_PROPERTY_DESCRIPTORS = [
    # hydrophobicity / lipophilicity
    "MolLogP",
    "MeanCrippenLogP",
    "HydrophobicContributionFraction",

    # refractivity / polarizability-related tendency
    "MolMR",
    "MeanCrippenMR",
    "HighMRContributionFraction",

    # polarity / hydrogen bonding
    "TPSA",
    "TPSA_over_LabuteASA",
    "HBA_fraction",
    "HBD_fraction",

    # charge / electronic distribution
    "MeanAbsGasteigerCharge",
    "GasteigerChargeVariance",
    "NegativeChargeFraction",
    "PositiveChargeFraction",
    "ChargeQ90MinusQ10",

    # electrotopological state
    "MeanAbsEState",
    "EStateVariance",
    "HighEStateFraction",
    "LowEStateFraction",

    # conjugation / aromaticity / rigidity
    "ConjugatedBondFraction",
    "AromaticAtomFraction",
    "AromaticBondFraction",
    "RotatableBondFraction",
]
```

## Conservative reduced set

If the first implementation should be smaller:

```python
REDUCED_GLOBAL_PROPERTY_DESCRIPTORS = [
    "MolLogP",
    "MeanCrippenLogP",
    "MolMR",
    "MeanCrippenMR",
    "TPSA",
    "TPSA_over_LabuteASA",
    "MeanAbsGasteigerCharge",
    "GasteigerChargeVariance",
    "ChargeQ90MinusQ10",
    "MeanAbsEState",
    "EStateVariance",
    "ConjugatedBondFraction",
    "AromaticAtomFraction",
    "RotatableBondFraction",
]
```

## Excluded for v1

```python
DROP_FOR_NOW = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "AtomCount",
    "RawRingCount",
    "RawRotatableBondCount",
    "RawHBA",
    "RawHBD",
    "pKa",
    "logS",
    "QED",
    "SA_score",
    "HOMO",
    "LUMO",
]
```

---

# 11. Practical Implementation Notes

## 11.1 Avoid direct regression first

The preferred task is pairwise ranking, not descriptor regression.

Reason:

> The exact descriptor value may be noisy or approximate, but the direction between two sufficiently different molecules is more reliable.

## 11.2 Use descriptor-specific thresholds

Do not use one absolute threshold for all descriptors.

Use:

\[
\tau_k=Q_{0.70}(|\Delta q_k|)
\]

for each descriptor.

## 11.3 Standardize descriptor heads

Each descriptor can have its own projection head:

\[
s_k(G)=h_k(f(G))
\]

This avoids forcing one scalar score to represent all physicochemical directions.

## 11.4 Consider group weighting

Descriptor groups can be weighted differently:

\[
\mathcal{L}_{\text{prop-rank}}
=
\lambda_{\text{hydro}}\mathcal{L}_{\text{hydro}}
+
\lambda_{\text{polar}}\mathcal{L}_{\text{polar}}
+
\lambda_{\text{charge}}\mathcal{L}_{\text{charge}}
+
\lambda_{\text{estate}}\mathcal{L}_{\text{estate}}
+
\lambda_{\text{conj}}\mathcal{L}_{\text{conj}}
\]

Possible first weights:

```text
hydrophobicity: 1.0
polarity/H-bonding: 1.0
charge/electronic: 1.0
EState: 1.0
conjugation/aromaticity/flexibility: 0.5
```

## 11.5 Beware shortcut descriptors

If a descriptor can be solved by simply counting atoms, bonds, or rings, it should either be excluded or normalized into a ratio/fraction.

---

# 12. Conceptual Role in HCMP

The descriptor ranking task teaches global physicochemical directions.

It does not replace:

- atom/bond BERT, which learns local syntax;
- segmentation, which learns functional chemical boundaries;
- scaffold triplet distance, which learns scaffold-level geometry.

Instead, it fills the missing molecule-level physicochemical axis:

\[
\text{local syntax}
\rightarrow
\text{functional group boundaries}
\rightarrow
\text{global property directions}
+
\text{scaffold geometry}
\]

This is why property threshold pairwise remains important even when scaffold triplet distance exists.
