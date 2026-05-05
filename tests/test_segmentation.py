from rdkit import Chem

from hcmp.chemistry.segmentation import SegmentationResult, segment_molecule
from hcmp.visualization.visualize_segmentation import get_cut_bond_indices


def _segments_of_type(result: SegmentationResult, segment_type: str):
    return [segment for segment in result.segments if segment.segment_type == segment_type]


def _labels_between_types(result: SegmentationResult, type_a: str, type_b: str):
    target = {type_a, type_b}
    return [
        label
        for label in result.bond_labels
        if {label.begin_segment_type, label.end_segment_type} == target
    ]


def _multiple_bond_to_oxygen_labels(result: SegmentationResult):
    mol = result.mol
    labels_by_idx = {label.bond_idx: label for label in result.bond_labels}
    labels = []
    for bond in mol.GetBonds():
        atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
        has_oxygen = any(atom.GetAtomicNum() == 8 for atom in atoms)
        if has_oxygen and bond.GetBondTypeAsDouble() > 1.0:
            labels.append(labels_by_idx[bond.GetIdx()])
    return labels


def _labels_by_bond_idx(result: SegmentationResult):
    return {label.bond_idx: label for label in result.bond_labels}


def test_ethylene_has_one_unsaturated_segment_and_no_internal_cut():
    result = segment_molecule("C=C")

    unsaturated = _segments_of_type(result, "unsaturated_conjugated")
    assert len(unsaturated) == 1
    assert len(unsaturated[0].atom_indices) == 2
    assert len(result.bond_labels) == 1
    assert result.bond_labels[0].cut_label == 0
    assert result.bond_labels[0].reason.startswith("same_segment")


def test_acetone_carbonyl_segment_has_cut_cc_boundary_bonds():
    result = segment_molecule("CC(=O)C")
    labels_by_idx = _labels_by_bond_idx(result)

    unsaturated = _segments_of_type(result, "unsaturated_conjugated")
    assert len(unsaturated) == 1
    carbonyl_segment = unsaturated[0]
    assert len(carbonyl_segment.atom_indices) == 2

    internal = [
        label
        for label in result.bond_labels
        if label.begin_segment_id == carbonyl_segment.segment_id
        and label.end_segment_id == carbonyl_segment.segment_id
    ]
    assert len(internal) == 1
    assert internal[0].cut_label == 0

    boundary = [
        label
        for label in result.bond_labels
        if label.reason.startswith("assigned_segment_to_background")
    ]
    assert len(boundary) == 2
    assert all(label.cut_label == 1 for label in boundary)

    carbonyl_bonds = [
        bond
        for bond in result.mol.GetBonds()
        if bond.GetBondTypeAsDouble() > 1.0
        and {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {6, 8}
    ]
    assert len(carbonyl_bonds) == 1
    assert labels_by_idx[carbonyl_bonds[0].GetIdx()].cut_label == 0

    cut_bonds = [result.mol.GetBondWithIdx(label.bond_idx) for label in boundary]
    assert all(bond.GetBondType() == Chem.BondType.SINGLE for bond in cut_bonds)
    assert all(
        bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in cut_bonds
    )


def test_benzene_has_one_ring_system_and_no_ring_internal_cut():
    result = segment_molecule("c1ccccc1")

    rings = _segments_of_type(result, "ring_system")
    assert len(rings) == 1
    assert len(rings[0].atom_indices) == 6
    assert all(label.cut_label == 0 for label in result.bond_labels)


def test_biphenyl_ring_ring_conjugation_behavior_is_deterministic_and_documented():
    first = segment_molecule("c1ccccc1-c2ccccc2")
    second = segment_molecule("c1ccccc1-c2ccccc2")

    first_signature = [
        (segment.segment_type, sorted(segment.atom_indices), sorted(segment.bond_indices), segment.reason)
        for segment in first.segments
    ]
    second_signature = [
        (segment.segment_type, sorted(segment.atom_indices), sorted(segment.bond_indices), segment.reason)
        for segment in second.segments
    ]
    assert first_signature == second_signature
    assert [
        (label.bond_idx, label.cut_label, label.reason)
        for label in first.bond_labels
    ] == [
        (label.bond_idx, label.cut_label, label.reason)
        for label in second.bond_labels
    ]

    rings = _segments_of_type(first, "ring_system")
    assert len(rings) in {1, 2}
    assert len(rings) == 1
    assert all("aromatic-compatible ring-ring bonds" in segment.reason for segment in rings)

    mol = first.mol
    ring_bond_indices = {bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()}
    labels_by_idx = {label.bond_idx: label for label in first.bond_labels}
    assert ring_bond_indices
    assert all(labels_by_idx[bond_idx].cut_label == 0 for bond_idx in ring_bond_indices)


def test_biphenyl_merges_direct_ring_ring_conjugation_without_cutting_inter_ring_bond():
    result = segment_molecule("c1ccccc1-c2ccccc2")
    labels_by_idx = _labels_by_bond_idx(result)
    rings = _segments_of_type(result, "ring_system")
    ring_bond_indices = {bond.GetIdx() for bond in result.mol.GetBonds() if bond.IsInRing()}
    inter_ring_bond_indices = {
        bond.GetIdx()
        for bond in result.mol.GetBonds()
        if not bond.IsInRing()
        and bond.GetBeginAtom().IsInRing()
        and bond.GetEndAtom().IsInRing()
    }
    cut_bond_indices = {label.bond_idx for label in result.bond_labels if label.cut_label == 1}

    assert len(rings) == 1
    assert rings[0].atom_indices == {atom.GetIdx() for atom in result.mol.GetAtoms()}
    assert ring_bond_indices
    assert inter_ring_bond_indices
    assert all(labels_by_idx[bond_idx].cut_label == 0 for bond_idx in ring_bond_indices)
    assert all(labels_by_idx[bond_idx].cut_label == 0 for bond_idx in inter_ring_bond_indices)
    assert cut_bond_indices == set()


def test_styrene_has_separate_ring_and_vinyl_segments_with_cut_attachment():
    result = segment_molecule("C=Cc1ccccc1")

    assert len(_segments_of_type(result, "ring_system")) == 1
    assert len(_segments_of_type(result, "unsaturated_conjugated")) == 1
    attachment_labels = _labels_between_types(result, "ring_system", "unsaturated_conjugated")
    assert len(attachment_labels) == 1
    assert attachment_labels[0].cut_label == 1


def test_pyridine_nitrogen_belongs_to_ring_segment():
    result = segment_molecule("c1ccncc1")
    mol = result.mol

    rings = _segments_of_type(result, "ring_system")
    assert len(rings) == 1
    nitrogen_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    assert len(nitrogen_indices) == 1
    assert nitrogen_indices[0] in rings[0].atom_indices


def test_chlorobenzene_has_ring_and_terminal_cl_with_cut_attachment():
    result = segment_molecule("Clc1ccccc1")

    assert len(_segments_of_type(result, "ring_system")) == 1
    terminal = _segments_of_type(result, "terminal_heteroatom")
    assert len(terminal) == 1
    attachment = _labels_between_types(result, "ring_system", "terminal_heteroatom")
    assert len(attachment) == 1
    assert attachment[0].cut_label == 1


def test_diethyl_ether_saturated_nonterminal_oxygen_stays_background():
    result = segment_molecule("CCOCC")

    assert not _segments_of_type(result, "terminal_heteroatom")
    assert not _segments_of_type(result, "heteroatom_cluster")
    assert all(label.cut_label == 0 for label in result.bond_labels)


def test_allene_forms_one_unsaturated_conjugated_segment():
    result = segment_molecule("C=C=C")

    unsaturated = _segments_of_type(result, "unsaturated_conjugated")
    assert len(unsaturated) == 1
    assert len(unsaturated[0].atom_indices) == 3
    assert len(unsaturated[0].bond_indices) == 2
    assert all(label.cut_label == 0 for label in result.bond_labels)


def test_propyne_cuts_only_methyl_to_alkyne_not_internal_triple_bond():
    result = segment_molecule("CC#C")
    labels_by_idx = _labels_by_bond_idx(result)

    triple_bonds = [
        bond for bond in result.mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE
    ]
    assert len(triple_bonds) == 1
    assert labels_by_idx[triple_bonds[0].GetIdx()].cut_label == 0

    cut_bond_indices = [label.bond_idx for label in result.bond_labels if label.cut_label == 1]
    assert len(cut_bond_indices) == 1
    cut_bond = result.mol.GetBondWithIdx(cut_bond_indices[0])
    assert cut_bond.GetBondType() == Chem.BondType.SINGLE


def test_stilbene_cuts_ring_vinyl_attachments_but_no_ring_internal_bonds():
    result = segment_molecule("c1ccccc1C=Cc2ccccc2")
    labels_by_idx = _labels_by_bond_idx(result)
    rings = _segments_of_type(result, "ring_system")
    unsaturated = _segments_of_type(result, "unsaturated_conjugated")

    ring_bond_indices = {bond.GetIdx() for bond in result.mol.GetBonds() if bond.IsInRing()}
    assert len(rings) == 2
    assert len(unsaturated) == 1
    assert ring_bond_indices
    assert all(labels_by_idx[bond_idx].cut_label == 0 for bond_idx in ring_bond_indices)

    attachment_labels = _labels_between_types(result, "ring_system", "unsaturated_conjugated")
    assert len(attachment_labels) == 2
    assert all(label.cut_label == 1 for label in attachment_labels)
    assert {label.bond_idx for label in result.bond_labels if label.cut_label == 1} == {
        label.bond_idx for label in attachment_labels
    }


def test_naphthalene_fused_rings_form_one_ring_system():
    result = segment_molecule("c1ccc2ccccc2c1")

    rings = _segments_of_type(result, "ring_system")
    assert len(rings) == 1
    assert len(rings[0].atom_indices) == result.mol.GetNumAtoms()
    assert all(label.cut_label == 0 for label in result.bond_labels)


def test_indole_fused_heteroaromatic_system_is_one_ring_system():
    result = segment_molecule("c1ccc2[nH]ccc2c1")

    rings = _segments_of_type(result, "ring_system")
    assert len(rings) == 1
    nitrogen_indices = [
        atom.GetIdx() for atom in result.mol.GetAtoms() if atom.GetAtomicNum() == 7
    ]
    assert len(nitrogen_indices) == 1
    assert nitrogen_indices[0] in rings[0].atom_indices
    assert all(label.cut_label == 0 for label in result.bond_labels)


def test_visualization_cut_bond_indices_match_cut_labels_exactly():
    result = segment_molecule("c1ccccc1C=Cc2ccccc2")
    expected = sorted(label.bond_idx for label in result.bond_labels if label.cut_label == 1)

    assert get_cut_bond_indices(result.mol, result.bond_labels) == expected


def test_cyclohexanone_carbonyl_oxygen_is_in_ring_system_and_non_cut():
    result = segment_molecule("O=C1CCCCC1")

    rings = _segments_of_type(result, "ring_system")
    assert len(rings) == 1

    mol = result.mol
    oxygen_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    assert len(oxygen_indices) == 1
    assert oxygen_indices[0] in rings[0].atom_indices

    carbonyl_labels = _multiple_bond_to_oxygen_labels(result)
    assert len(carbonyl_labels) == 1
    assert carbonyl_labels[0].cut_label == 0
    assert carbonyl_labels[0].begin_segment_type == "ring_system"
    assert carbonyl_labels[0].end_segment_type == "ring_system"


def test_gamma_butyrolactone_does_not_cut_carbonyl_bond():
    result = segment_molecule("O=C1OCCC1")

    carbonyl_labels = _multiple_bond_to_oxygen_labels(result)
    assert len(carbonyl_labels) == 1
    assert carbonyl_labels[0].cut_label == 0
    assert carbonyl_labels[0].begin_segment_type == "ring_system"
    assert carbonyl_labels[0].end_segment_type == "ring_system"


def test_benzaldehyde_keeps_ring_and_carbonyl_separate_with_cut_attachment():
    result = segment_molecule("O=Cc1ccccc1")

    assert len(_segments_of_type(result, "ring_system")) == 1
    assert len(_segments_of_type(result, "unsaturated_conjugated")) == 1

    carbonyl_labels = _multiple_bond_to_oxygen_labels(result)
    assert len(carbonyl_labels) == 1
    assert carbonyl_labels[0].cut_label == 0
    assert carbonyl_labels[0].begin_segment_type == "unsaturated_conjugated"
    assert carbonyl_labels[0].end_segment_type == "unsaturated_conjugated"

    attachment = _labels_between_types(result, "ring_system", "unsaturated_conjugated")
    assert len(attachment) == 1
    assert attachment[0].cut_label == 1
