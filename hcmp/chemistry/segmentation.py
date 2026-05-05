"""Conservative priority-based molecular segmentation for HCMP v1."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem


@dataclass
class Segment:
    """A chemically salient motif segment assigned by the HCMP v1 rule."""

    segment_id: int
    segment_type: str
    priority: int
    atom_indices: set[int]
    bond_indices: set[int]
    reason: str


@dataclass
class BondLabel:
    """A final cut/non-cut label for one RDKit bond."""

    bond_idx: int
    begin_atom_idx: int
    end_atom_idx: int
    cut_label: int
    reason: str
    begin_segment_id: Optional[int]
    end_segment_id: Optional[int]
    begin_segment_type: Optional[str]
    end_segment_type: Optional[str]


@dataclass
class SegmentationResult:
    """Complete HCMP v1 segmentation output for a molecule."""

    smiles: str
    canonical_smiles: str
    mol: Chem.Mol = field(repr=False)
    segments: list[Segment]
    atom_to_segment: dict[int, int]
    bond_labels: list[BondLabel]


class _UnionFind:
    def __init__(self, n_items: int) -> None:
        self.parent = list(range(n_items))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def segment_molecule(smiles: str) -> SegmentationResult:
    """Parse a SMILES string and generate HCMP v1 segment and cut-bond labels.

    Explicit hydrogens are removed before segmentation. Invalid SMILES strings
    raise ``ValueError`` with a concise diagnostic.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    return segment_mol(mol, smiles=smiles)


def segment_mol(mol: Chem.Mol, smiles: Optional[str] = None) -> SegmentationResult:
    """Generate HCMP v1 segment and cut-bond labels from an RDKit molecule."""

    if mol is None:
        raise ValueError("mol must be an RDKit Mol, got None.")

    try:
        work_mol = Chem.Mol(mol)
        Chem.SanitizeMol(work_mol)
        work_mol = Chem.RemoveHs(work_mol, sanitize=True)
    except Exception as exc:  # pragma: no cover - RDKit exception types vary.
        raise ValueError(f"Could not sanitize molecule: {exc}") from exc

    canonical_smiles = Chem.MolToSmiles(work_mol, canonical=True)
    input_smiles = smiles if smiles is not None else canonical_smiles

    segments: list[Segment] = []
    atom_to_segment: dict[int, int] = {}
    bond_to_segment: dict[int, int] = {}

    _assign_ring_systems(work_mol, segments, atom_to_segment, bond_to_segment)
    _assign_unsaturated_conjugated(work_mol, segments, atom_to_segment, bond_to_segment)
    _assign_heteroatom_clusters(work_mol, segments, atom_to_segment, bond_to_segment)
    _assign_terminal_heteroatoms(work_mol, segments, atom_to_segment, bond_to_segment)

    segment_by_id = {segment.segment_id: segment for segment in segments}
    bond_labels = _generate_bond_labels(work_mol, atom_to_segment, segment_by_id)

    return SegmentationResult(
        smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        mol=work_mol,
        segments=segments,
        atom_to_segment=atom_to_segment,
        bond_labels=bond_labels,
    )


def _next_segment_id(segments: list[Segment]) -> int:
    return len(segments)


def _add_segment(
    *,
    segments: list[Segment],
    atom_to_segment: dict[int, int],
    bond_to_segment: dict[int, int],
    segment_type: str,
    priority: int,
    atom_indices: set[int],
    bond_indices: set[int],
    reason: str,
) -> None:
    atom_indices = set(atom_indices) - set(atom_to_segment)
    bond_indices = set(bond_indices) - set(bond_to_segment)
    if not atom_indices:
        return

    segment_id = _next_segment_id(segments)
    segment = Segment(
        segment_id=segment_id,
        segment_type=segment_type,
        priority=priority,
        atom_indices=atom_indices,
        bond_indices=bond_indices,
        reason=reason,
    )
    segments.append(segment)

    for atom_idx in sorted(atom_indices):
        atom_to_segment[atom_idx] = segment_id
    for bond_idx in sorted(bond_indices):
        bond_to_segment[bond_idx] = segment_id


def _assign_ring_systems(
    mol: Chem.Mol,
    segments: list[Segment],
    atom_to_segment: dict[int, int],
    bond_to_segment: dict[int, int],
) -> None:
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not atom_rings:
        return

    ring_systems = _merge_overlapping_rings(atom_rings)
    ring_systems = _merge_direct_ring_ring_conjugation(mol, ring_systems)

    for atom_indices in sorted(ring_systems, key=lambda atoms: (min(atoms), len(atoms))):
        atom_indices, exocyclic_bond_indices = _include_exocyclic_multiple_heteroatoms(
            mol,
            atom_indices,
        )
        bond_indices = _bonds_with_both_atoms_in(mol, atom_indices)
        bond_indices.update(exocyclic_bond_indices)
        _add_segment(
            segments=segments,
            atom_to_segment=atom_to_segment,
            bond_to_segment=bond_to_segment,
            segment_type="ring_system",
            priority=1,
            atom_indices=atom_indices,
            bond_indices=bond_indices,
            reason=(
                "Ring-system segment from rings merged by shared atoms/bonds; "
                "separate rings merge across direct RDKit-conjugated or "
                "aromatic-compatible ring-ring bonds; "
                "exocyclic multiple bonds from ring atoms to heteroatoms are preserved."
            ),
        )


def _include_exocyclic_multiple_heteroatoms(
    mol: Chem.Mol,
    ring_atom_indices: set[int],
) -> tuple[set[int], set[int]]:
    """Absorb cyclic carbonyl-like exocyclic heteroatoms into a ring segment."""

    expanded_atom_indices = set(ring_atom_indices)
    exocyclic_bond_indices: set[int] = set()
    for atom_idx in sorted(ring_atom_indices):
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            if bond.IsInRing() or bond.GetBondTypeAsDouble() <= 1.0:
                continue
            neighbor = bond.GetOtherAtom(atom)
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in ring_atom_indices or not _is_heteroatom(neighbor):
                continue

            expanded_atom_indices.add(neighbor_idx)
            exocyclic_bond_indices.add(bond.GetIdx())

    return expanded_atom_indices, exocyclic_bond_indices


def _merge_overlapping_rings(atom_rings: list[set[int]]) -> list[set[int]]:
    union_find = _UnionFind(len(atom_rings))
    for i, atoms_i in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if atoms_i & atom_rings[j]:
                union_find.union(i, j)

    grouped: dict[int, set[int]] = defaultdict(set)
    for idx, atom_set in enumerate(atom_rings):
        grouped[union_find.find(idx)].update(atom_set)

    return [grouped[key] for key in sorted(grouped)]


def _merge_direct_ring_ring_conjugation(
    mol: Chem.Mol,
    ring_systems: list[set[int]],
) -> list[set[int]]:
    """Merge separate ring systems only when directly ring-ring conjugated.

    This deliberately does not merge through non-ring unsaturated linkers, so
    Ph-CH=CH-Ph remains ring / alkene / ring.
    """

    if len(ring_systems) <= 1:
        return ring_systems

    atom_to_ring_system: dict[int, int] = {}
    for system_idx, atoms in enumerate(ring_systems):
        for atom_idx in atoms:
            atom_to_ring_system[atom_idx] = system_idx

    union_find = _UnionFind(len(ring_systems))
    for bond in mol.GetBonds():
        if _should_merge_ring_systems_by_direct_conjugation(
            bond,
            atom_to_ring_system,
        ):
            union_find.union(
                atom_to_ring_system[bond.GetBeginAtomIdx()],
                atom_to_ring_system[bond.GetEndAtomIdx()],
            )

    grouped: dict[int, set[int]] = defaultdict(set)
    for idx, atom_set in enumerate(ring_systems):
        grouped[union_find.find(idx)].update(atom_set)

    return [grouped[key] for key in sorted(grouped)]


def _should_merge_ring_systems_by_direct_conjugation(
    bond: Chem.Bond,
    atom_to_ring_system: dict[int, int],
) -> bool:
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_idx = begin_atom.GetIdx()
    end_idx = end_atom.GetIdx()

    begin_system = atom_to_ring_system.get(begin_idx)
    end_system = atom_to_ring_system.get(end_idx)
    if begin_system is None or end_system is None or begin_system == end_system:
        return False
    if not begin_atom.IsInRing() or not end_atom.IsInRing():
        return False
    if bond.IsInRing():
        return False

    return bond.GetIsConjugated() or (
        begin_atom.GetIsAromatic() and end_atom.GetIsAromatic()
    )


def _assign_unsaturated_conjugated(
    mol: Chem.Mol,
    segments: list[Segment],
    atom_to_segment: dict[int, int],
    bond_to_segment: dict[int, int],
) -> None:
    seed_bonds: set[int] = set()
    auxiliary_bonds: set[int] = set()

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if bond_idx in bond_to_segment or begin in atom_to_segment or end in atom_to_segment:
            continue
        if bond.IsInRing():
            continue

        is_seed = bond.GetBondTypeAsDouble() > 1.0
        if is_seed:
            seed_bonds.add(bond_idx)
            auxiliary_bonds.add(bond_idx)
        elif bond.GetIsConjugated():
            auxiliary_bonds.add(bond_idx)

    for component_bonds in _connected_bond_components(mol, auxiliary_bonds):
        if not component_bonds & seed_bonds:
            continue
        atom_indices = _atoms_from_bonds(mol, component_bonds)
        _add_segment(
            segments=segments,
            atom_to_segment=atom_to_segment,
            bond_to_segment=bond_to_segment,
            segment_type="unsaturated_conjugated",
            priority=2,
            atom_indices=atom_indices,
            bond_indices=component_bonds,
            reason=(
                "Non-ring multiple-bond seed expanded through unassigned "
                "RDKit-perceived conjugated bonds."
            ),
        )


def _connected_bond_components(mol: Chem.Mol, bond_indices: set[int]) -> list[set[int]]:
    if not bond_indices:
        return []

    atom_to_bonds: dict[int, set[int]] = defaultdict(set)
    for bond_idx in bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_to_bonds[bond.GetBeginAtomIdx()].add(bond_idx)
        atom_to_bonds[bond.GetEndAtomIdx()].add(bond_idx)

    components: list[set[int]] = []
    seen: set[int] = set()
    for start_bond in sorted(bond_indices):
        if start_bond in seen:
            continue
        component: set[int] = set()
        queue: deque[int] = deque([start_bond])
        seen.add(start_bond)
        while queue:
            bond_idx = queue.popleft()
            component.add(bond_idx)
            bond = mol.GetBondWithIdx(bond_idx)
            for atom_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                for neighbor_bond_idx in sorted(atom_to_bonds[atom_idx]):
                    if neighbor_bond_idx not in seen:
                        seen.add(neighbor_bond_idx)
                        queue.append(neighbor_bond_idx)
        components.append(component)

    return sorted(components, key=lambda bonds: (min(_atoms_from_bonds(mol, bonds)), min(bonds)))


def _assign_heteroatom_clusters(
    mol: Chem.Mol,
    segments: list[Segment],
    atom_to_segment: dict[int, int],
    bond_to_segment: dict[int, int],
) -> None:
    candidate_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if _is_heteroatom(atom) and atom.GetIdx() not in atom_to_segment
    }
    if not candidate_atoms:
        return

    adjacency: dict[int, set[int]] = {atom_idx: set() for atom_idx in candidate_atoms}
    internal_bond_by_pair: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if bond_idx in bond_to_segment:
            continue
        if begin in candidate_atoms and end in candidate_atoms:
            adjacency[begin].add(end)
            adjacency[end].add(begin)
            internal_bond_by_pair[tuple(sorted((begin, end)))] = bond_idx

    for component_atoms in _connected_atom_components(adjacency):
        if len(component_atoms) < 2:
            continue
        bond_indices = {
            bond_idx
            for pair, bond_idx in internal_bond_by_pair.items()
            if pair[0] in component_atoms and pair[1] in component_atoms
        }
        _add_segment(
            segments=segments,
            atom_to_segment=atom_to_segment,
            bond_to_segment=bond_to_segment,
            segment_type="heteroatom_cluster",
            priority=3,
            atom_indices=component_atoms,
            bond_indices=bond_indices,
            reason="Connected unassigned heteroatom cluster with at least two heteroatoms.",
        )


def _assign_terminal_heteroatoms(
    mol: Chem.Mol,
    segments: list[Segment],
    atom_to_segment: dict[int, int],
    bond_to_segment: dict[int, int],
) -> None:
    for atom in sorted(mol.GetAtoms(), key=lambda rd_atom: rd_atom.GetIdx()):
        atom_idx = atom.GetIdx()
        if atom_idx in atom_to_segment or not _is_heteroatom(atom):
            continue
        heavy_neighbors = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() > 1
        ]
        if len(heavy_neighbors) != 1:
            continue

        _add_segment(
            segments=segments,
            atom_to_segment=atom_to_segment,
            bond_to_segment=bond_to_segment,
            segment_type="terminal_heteroatom",
            priority=4,
            atom_indices={atom_idx},
            bond_indices=set(),
            reason=(
                "Remaining heteroatom with exactly one heavy-atom neighbor; "
                "attachment bond is labeled during final boundary generation."
            ),
        )


def _generate_bond_labels(
    mol: Chem.Mol,
    atom_to_segment: dict[int, int],
    segment_by_id: dict[int, Segment],
) -> list[BondLabel]:
    labels: list[BondLabel] = []
    for bond in sorted(mol.GetBonds(), key=lambda rd_bond: rd_bond.GetIdx()):
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_segment_id = atom_to_segment.get(begin)
        end_segment_id = atom_to_segment.get(end)
        begin_segment_type = (
            segment_by_id[begin_segment_id].segment_type
            if begin_segment_id is not None
            else None
        )
        end_segment_type = (
            segment_by_id[end_segment_id].segment_type
            if end_segment_id is not None
            else None
        )

        if begin_segment_id is None and end_segment_id is None:
            cut_label = 0
            reason = "both_atoms_unassigned_background"
        elif begin_segment_id is not None and begin_segment_id == end_segment_id:
            cut_label = 0
            reason = f"same_segment:{begin_segment_type}"
        elif begin_segment_id is None or end_segment_id is None:
            cut_label = 1
            assigned_type = begin_segment_type or end_segment_type
            reason = f"assigned_segment_to_background:{assigned_type}"
        else:
            cut_label = 1
            reason = f"different_segments:{begin_segment_type}|{end_segment_type}"

        labels.append(
            BondLabel(
                bond_idx=bond.GetIdx(),
                begin_atom_idx=begin,
                end_atom_idx=end,
                cut_label=cut_label,
                reason=reason,
                begin_segment_id=begin_segment_id,
                end_segment_id=end_segment_id,
                begin_segment_type=begin_segment_type,
                end_segment_type=end_segment_type,
            )
        )

    return labels


def _connected_atom_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        component: set[int] = set()
        while queue:
            atom_idx = queue.popleft()
            component.add(atom_idx)
            for neighbor in sorted(adjacency[atom_idx]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _bonds_with_both_atoms_in(mol: Chem.Mol, atom_indices: set[int]) -> set[int]:
    return {
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_indices and bond.GetEndAtomIdx() in atom_indices
    }


def _atoms_from_bonds(mol: Chem.Mol, bond_indices: set[int]) -> set[int]:
    atom_indices: set[int] = set()
    for bond_idx in bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())
    return atom_indices


def _is_heteroatom(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() != 6
