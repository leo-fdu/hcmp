"""Minimal molecular graph construction for HCMP encoders."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rdkit import Chem

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment.
    raise ModuleNotFoundError("hcmp.data.graph_builder requires torch.") from exc


DEFAULT_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
CHIRALITY_VALUES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_VALUES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


@dataclass(frozen=True)
class FeatureSpec:
    """Feature dimensions and slices for HCMP's minimal graph input."""

    atomic_numbers: list[int]
    formal_charge_clip: int = 3
    rich_molecular_features: bool = False
    feature_mode: str = "hcmp"

    @property
    def atom_type_dim(self) -> int:
        return len(self.atomic_numbers) + 2  # known atoms + unknown + mask

    @property
    def atom_type_mask_index(self) -> int:
        return self.atom_type_dim - 1

    @property
    def formal_charge_dim(self) -> int:
        return 2 * self.formal_charge_clip + 2  # clipped range + mask

    @property
    def formal_charge_mask_index(self) -> int:
        return self.formal_charge_dim - 1

    @property
    def chirality_dim(self) -> int:
        return len(CHIRALITY_VALUES) + 1

    @property
    def chirality_mask_index(self) -> int:
        return self.chirality_dim - 1

    @property
    def bond_type_dim(self) -> int:
        return len(BOND_TYPES) + 2

    @property
    def bond_type_mask_index(self) -> int:
        return self.bond_type_dim - 1

    @property
    def bond_stereo_dim(self) -> int:
        return len(BOND_STEREO_VALUES) + 1

    @property
    def bond_stereo_mask_index(self) -> int:
        return self.bond_stereo_dim - 1

    @property
    def degree_dim(self) -> int:
        return 7  # clipped 0..5 plus mask

    @property
    def degree_mask_index(self) -> int:
        return self.degree_dim - 1

    @property
    def hybridization_dim(self) -> int:
        return 7  # five common values plus unknown plus mask

    @property
    def hybridization_mask_index(self) -> int:
        return self.hybridization_dim - 1

    @property
    def boolean_dim(self) -> int:
        return 3  # false, true, mask

    @property
    def boolean_mask_index(self) -> int:
        return 2

    @property
    def num_hydrogens_dim(self) -> int:
        return 6  # clipped 0..4 plus mask

    @property
    def num_hydrogens_mask_index(self) -> int:
        return self.num_hydrogens_dim - 1

    @property
    def node_feature_dim(self) -> int:
        base = self.atom_type_dim + self.formal_charge_dim + self.chirality_dim
        if self.feature_mode != "graph_bert":
            return base
        return (
            base
            + self.degree_dim
            + self.hybridization_dim
            + self.boolean_dim
            + self.num_hydrogens_dim
            + self.boolean_dim
        )

    @property
    def edge_feature_dim(self) -> int:
        base = self.bond_type_dim + self.bond_stereo_dim
        if self.feature_mode != "graph_bert":
            return base
        return base + self.boolean_dim + self.boolean_dim + self.boolean_dim

    @property
    def atom_type_slice(self) -> slice:
        return slice(0, self.atom_type_dim)

    @property
    def formal_charge_slice(self) -> slice:
        start = self.atom_type_dim
        return slice(start, start + self.formal_charge_dim)

    @property
    def chirality_slice(self) -> slice:
        start = self.atom_type_dim + self.formal_charge_dim
        return slice(start, start + self.chirality_dim)

    @property
    def bond_type_slice(self) -> slice:
        return slice(0, self.bond_type_dim)

    @property
    def bond_stereo_slice(self) -> slice:
        start = self.bond_type_dim
        return slice(start, start + self.bond_stereo_dim)

    @property
    def degree_slice(self) -> slice:
        start = self.chirality_slice.stop
        return slice(start, start + self.degree_dim)

    @property
    def hybridization_slice(self) -> slice:
        start = self.degree_slice.stop
        return slice(start, start + self.hybridization_dim)

    @property
    def atom_aromaticity_slice(self) -> slice:
        start = self.hybridization_slice.stop
        return slice(start, start + self.boolean_dim)

    @property
    def num_hydrogens_slice(self) -> slice:
        start = self.atom_aromaticity_slice.stop
        return slice(start, start + self.num_hydrogens_dim)

    @property
    def atom_ring_slice(self) -> slice:
        start = self.num_hydrogens_slice.stop
        return slice(start, start + self.boolean_dim)

    @property
    def bond_conjugation_slice(self) -> slice:
        start = self.bond_stereo_slice.stop
        return slice(start, start + self.boolean_dim)

    @property
    def bond_aromaticity_slice(self) -> slice:
        start = self.bond_conjugation_slice.stop
        return slice(start, start + self.boolean_dim)

    @property
    def bond_ring_slice(self) -> slice:
        start = self.bond_aromaticity_slice.stop
        return slice(start, start + self.boolean_dim)

    @property
    def atom_target_fields(self) -> list[str]:
        if self.feature_mode == "graph_bert":
            return [
                "atomic_number",
                "formal_charge",
                "degree",
                "hybridization",
                "aromaticity",
                "num_hydrogens",
                "ring_membership",
                "chirality",
            ]
        return ["atomic_number"]

    @property
    def bond_target_fields(self) -> list[str]:
        if self.feature_mode == "graph_bert":
            return [
                "bond_type",
                "conjugation",
                "aromaticity",
                "ring_membership",
                "stereo",
            ]
        return ["bond_type"]


@dataclass
class GraphData:
    """Single-molecule graph tensors with one edge per RDKit bond."""

    mol_id: Any
    input_smiles: str
    canonical_smiles: str
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    atom_type_targets: torch.Tensor
    bond_type_targets: torch.Tensor
    graph_bert_atom_targets: dict[str, torch.Tensor] | None = None
    graph_bert_bond_targets: dict[str, torch.Tensor] | None = None
    cut_labels: torch.Tensor | None = None
    descriptor_values: torch.Tensor | None = None
    global_idx: int | None = None
    source_row_index: int | None = None


@dataclass
class MaskedGraphData:
    """BERT-masked graph and the minimal prediction targets."""

    graph: GraphData
    atom_mask_indices: torch.Tensor
    bond_mask_indices: torch.Tensor
    atom_type_targets: torch.Tensor
    bond_type_targets: torch.Tensor


@dataclass
class GraphBatch:
    """A batch of HCMP graph tensors."""

    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    atom_type_targets: torch.Tensor
    bond_type_targets: torch.Tensor
    graph_bert_atom_targets: dict[str, torch.Tensor] | None
    graph_bert_bond_targets: dict[str, torch.Tensor] | None
    cut_labels: torch.Tensor | None
    descriptor_values: torch.Tensor | None
    global_indices: torch.Tensor
    source_row_indices: torch.Tensor
    atom_mask_indices: torch.Tensor | None
    bond_mask_indices: torch.Tensor | None
    mol_ids: list[Any]
    canonical_smiles: list[str]


def default_feature_spec(config: dict[str, Any] | None = None) -> FeatureSpec:
    config = config or {}
    feature_mode = str(config.get("feature_mode", "graph_bert" if config.get("rich_molecular_features", False) else "hcmp"))
    if feature_mode not in {"hcmp", "graph_bert"}:
        raise ValueError("features.feature_mode must be either 'hcmp' or 'graph_bert'.")
    return FeatureSpec(
        atomic_numbers=list(config.get("atomic_numbers", DEFAULT_ATOMIC_NUMBERS)),
        formal_charge_clip=int(config.get("formal_charge_clip", 3)),
        rich_molecular_features=(feature_mode == "graph_bert"),
        feature_mode=feature_mode,
    )


def mol_to_graph(
    mol: Chem.Mol,
    mol_id: Any = None,
    input_smiles: str | None = None,
    feature_spec: FeatureSpec | None = None,
    cut_labels: list[int] | None = None,
    descriptor_values: list[float] | torch.Tensor | None = None,
    global_idx: int | None = None,
    source_row_index: int | None = None,
) -> GraphData:
    """Convert an RDKit molecule to HCMP's minimal graph tensor schema."""

    feature_spec = feature_spec or default_feature_spec()
    work_mol = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    canonical_smiles = Chem.MolToSmiles(work_mol, canonical=True)
    node_features, atom_targets, graph_bert_atom_targets = _build_node_features(work_mol, feature_spec)
    edge_features, edge_index, bond_targets, graph_bert_bond_targets = _build_edge_features(work_mol, feature_spec)
    cut_tensor = None
    if cut_labels is not None:
        cut_tensor = torch.tensor(cut_labels, dtype=torch.float32)
        if cut_tensor.numel() != work_mol.GetNumBonds():
            raise ValueError("cut_labels length must match the number of RDKit bonds.")
    descriptor_tensor = None
    if descriptor_values is not None:
        descriptor_tensor = torch.as_tensor(descriptor_values, dtype=torch.float32)

    return GraphData(
        mol_id=mol_id,
        input_smiles=input_smiles or canonical_smiles,
        canonical_smiles=canonical_smiles,
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        atom_type_targets=atom_targets,
        bond_type_targets=bond_targets,
        graph_bert_atom_targets=graph_bert_atom_targets,
        graph_bert_bond_targets=graph_bert_bond_targets,
        cut_labels=cut_tensor,
        descriptor_values=descriptor_tensor,
        global_idx=global_idx,
        source_row_index=source_row_index,
    )


def smiles_to_graph(
    smiles: str,
    mol_id: Any = None,
    feature_spec: FeatureSpec | None = None,
) -> GraphData:
    """Parse SMILES and build an HCMP graph."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol_to_graph(mol, mol_id=mol_id, input_smiles=smiles, feature_spec=feature_spec)


def collate_graphs(graphs: list[GraphData]) -> GraphBatch:
    """Collate single-molecule graphs without duplicating directed edges."""

    if not graphs:
        raise ValueError("collate_graphs requires at least one graph.")
    node_features = []
    edge_features = []
    edge_indices = []
    atom_targets = []
    bond_targets = []
    graph_bert_atom_targets: dict[str, list[torch.Tensor]] = {}
    graph_bert_bond_targets: dict[str, list[torch.Tensor]] = {}
    cut_labels = []
    descriptor_values = []
    global_indices = []
    source_row_indices = []
    node_batch = []
    edge_batch = []
    mol_ids = []
    canonical_smiles = []
    node_offset = 0
    has_cut_labels = all(graph.cut_labels is not None for graph in graphs)
    has_descriptor_values = all(graph.descriptor_values is not None for graph in graphs)
    for batch_idx, graph in enumerate(graphs):
        node_features.append(graph.node_features)
        edge_features.append(graph.edge_features)
        edge_indices.append(graph.edge_index + node_offset)
        atom_targets.append(graph.atom_type_targets)
        bond_targets.append(graph.bond_type_targets)
        if graph.graph_bert_atom_targets is not None:
            for field, target in graph.graph_bert_atom_targets.items():
                graph_bert_atom_targets.setdefault(field, []).append(target)
        if graph.graph_bert_bond_targets is not None:
            for field, target in graph.graph_bert_bond_targets.items():
                graph_bert_bond_targets.setdefault(field, []).append(target)
        if has_cut_labels and graph.cut_labels is not None:
            cut_labels.append(graph.cut_labels)
        if has_descriptor_values and graph.descriptor_values is not None:
            descriptor_values.append(graph.descriptor_values)
        global_indices.append(-1 if graph.global_idx is None else int(graph.global_idx))
        source_row_indices.append(-1 if graph.source_row_index is None else int(graph.source_row_index))
        node_batch.append(torch.full((graph.node_features.shape[0],), batch_idx, dtype=torch.long))
        edge_batch.append(torch.full((graph.edge_features.shape[0],), batch_idx, dtype=torch.long))
        mol_ids.append(graph.mol_id)
        canonical_smiles.append(graph.canonical_smiles)
        node_offset += graph.node_features.shape[0]
    return GraphBatch(
        node_features=torch.cat(node_features, dim=0),
        edge_features=torch.cat(edge_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        node_batch=torch.cat(node_batch, dim=0),
        edge_batch=torch.cat(edge_batch, dim=0),
        atom_type_targets=torch.cat(atom_targets, dim=0),
        bond_type_targets=torch.cat(bond_targets, dim=0),
        graph_bert_atom_targets=(
            {field: torch.cat(values, dim=0) for field, values in graph_bert_atom_targets.items()}
            if graph_bert_atom_targets and all(len(values) == len(graphs) for values in graph_bert_atom_targets.values())
            else None
        ),
        graph_bert_bond_targets=(
            {field: torch.cat(values, dim=0) for field, values in graph_bert_bond_targets.items()}
            if graph_bert_bond_targets and all(len(values) == len(graphs) for values in graph_bert_bond_targets.values())
            else None
        ),
        cut_labels=torch.cat(cut_labels, dim=0) if has_cut_labels else None,
        descriptor_values=torch.stack(descriptor_values, dim=0) if has_descriptor_values else None,
        global_indices=torch.tensor(global_indices, dtype=torch.long),
        source_row_indices=torch.tensor(source_row_indices, dtype=torch.long),
        atom_mask_indices=None,
        bond_mask_indices=None,
        mol_ids=mol_ids,
        canonical_smiles=canonical_smiles,
    )


def apply_bert_masking(
    graph: GraphData,
    feature_spec: FeatureSpec,
    atom_mask_ratio: float = 0.15,
    bond_mask_ratio: float = 0.15,
    generator: torch.Generator | None = None,
) -> MaskedGraphData:
    """Mask atom fields and bond fields while targeting only atom/bond type."""

    node_features = graph.node_features.clone()
    edge_features = graph.edge_features.clone()
    atom_mask_indices = _sample_mask_indices(
        graph.node_features.shape[0],
        atom_mask_ratio,
        generator,
    )
    bond_mask_indices = _sample_mask_indices(
        graph.edge_features.shape[0],
        bond_mask_ratio,
        generator,
    )

    if atom_mask_indices.numel() > 0:
        _set_one_hot_mask(
            node_features,
            atom_mask_indices,
            feature_spec.atom_type_slice,
            feature_spec.atom_type_mask_index,
        )
        _set_one_hot_mask(
            node_features,
            atom_mask_indices,
            feature_spec.formal_charge_slice,
            feature_spec.formal_charge_mask_index,
        )
        _set_one_hot_mask(
            node_features,
            atom_mask_indices,
            feature_spec.chirality_slice,
            feature_spec.chirality_mask_index,
        )

    if bond_mask_indices.numel() > 0:
        _set_one_hot_mask(
            edge_features,
            bond_mask_indices,
            feature_spec.bond_type_slice,
            feature_spec.bond_type_mask_index,
        )
        _set_one_hot_mask(
            edge_features,
            bond_mask_indices,
            feature_spec.bond_stereo_slice,
            feature_spec.bond_stereo_mask_index,
        )

    masked_graph = replace(
        graph,
        node_features=node_features,
        edge_features=edge_features,
    )
    return MaskedGraphData(
        graph=masked_graph,
        atom_mask_indices=atom_mask_indices,
        bond_mask_indices=bond_mask_indices,
        atom_type_targets=graph.atom_type_targets[atom_mask_indices],
        bond_type_targets=graph.bond_type_targets[bond_mask_indices],
    )


def _build_node_features(
    mol: Chem.Mol,
    feature_spec: FeatureSpec,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    features = torch.zeros((mol.GetNumAtoms(), feature_spec.node_feature_dim), dtype=torch.float32)
    targets = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    graph_bert_targets: dict[str, torch.Tensor] | None = None
    if feature_spec.feature_mode == "graph_bert":
        graph_bert_targets = {
            "atomic_number": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "formal_charge": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "degree": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "hybridization": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "aromaticity": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "num_hydrogens": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "ring_membership": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
            "chirality": torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
        }
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_class = _index_or_unknown(atom.GetAtomicNum(), feature_spec.atomic_numbers)
        charge_class = _formal_charge_class(atom.GetFormalCharge(), feature_spec)
        chirality_class = _index_or_unknown(atom.GetChiralTag(), CHIRALITY_VALUES, unknown_at_end=False)
        targets[idx] = atom_class
        features[idx, feature_spec.atom_type_slice.start + atom_class] = 1.0
        features[idx, feature_spec.formal_charge_slice.start + charge_class] = 1.0
        features[idx, feature_spec.chirality_slice.start + chirality_class] = 1.0
        if feature_spec.feature_mode == "graph_bert":
            degree = min(int(atom.GetDegree()), 5)
            hybridization_values = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
            hybridization = _index_or_unknown(atom.GetHybridization(), hybridization_values)
            hydrogens = min(int(atom.GetTotalNumHs()), 4)
            aromaticity = int(atom.GetIsAromatic())
            ring = int(atom.IsInRing())
            features[idx, feature_spec.degree_slice.start + degree] = 1.0
            features[idx, feature_spec.hybridization_slice.start + hybridization] = 1.0
            features[idx, feature_spec.atom_aromaticity_slice.start + aromaticity] = 1.0
            features[idx, feature_spec.num_hydrogens_slice.start + hydrogens] = 1.0
            features[idx, feature_spec.atom_ring_slice.start + ring] = 1.0
            assert graph_bert_targets is not None
            graph_bert_targets["atomic_number"][idx] = atom_class
            graph_bert_targets["formal_charge"][idx] = charge_class
            graph_bert_targets["degree"][idx] = degree
            graph_bert_targets["hybridization"][idx] = hybridization
            graph_bert_targets["aromaticity"][idx] = aromaticity
            graph_bert_targets["num_hydrogens"][idx] = hydrogens
            graph_bert_targets["ring_membership"][idx] = ring
            graph_bert_targets["chirality"][idx] = chirality_class
    return features, targets, graph_bert_targets


def _build_edge_features(
    mol: Chem.Mol,
    feature_spec: FeatureSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    features = torch.zeros((mol.GetNumBonds(), feature_spec.edge_feature_dim), dtype=torch.float32)
    edge_index = torch.zeros((2, mol.GetNumBonds()), dtype=torch.long)
    targets = torch.zeros(mol.GetNumBonds(), dtype=torch.long)
    graph_bert_targets: dict[str, torch.Tensor] | None = None
    if feature_spec.feature_mode == "graph_bert":
        graph_bert_targets = {
            "bond_type": torch.zeros(mol.GetNumBonds(), dtype=torch.long),
            "conjugation": torch.zeros(mol.GetNumBonds(), dtype=torch.long),
            "aromaticity": torch.zeros(mol.GetNumBonds(), dtype=torch.long),
            "ring_membership": torch.zeros(mol.GetNumBonds(), dtype=torch.long),
            "stereo": torch.zeros(mol.GetNumBonds(), dtype=torch.long),
        }
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        bond_type_class = _index_or_unknown(bond.GetBondType(), BOND_TYPES)
        stereo_class = _index_or_unknown(bond.GetStereo(), BOND_STEREO_VALUES, unknown_at_end=False)
        targets[idx] = bond_type_class
        edge_index[0, idx] = bond.GetBeginAtomIdx()
        edge_index[1, idx] = bond.GetEndAtomIdx()
        features[idx, feature_spec.bond_type_slice.start + bond_type_class] = 1.0
        features[idx, feature_spec.bond_stereo_slice.start + stereo_class] = 1.0
        if feature_spec.feature_mode == "graph_bert":
            conjugation = int(bond.GetIsConjugated())
            aromaticity = int(bond.GetIsAromatic())
            ring = int(bond.IsInRing())
            features[idx, feature_spec.bond_conjugation_slice.start + conjugation] = 1.0
            features[idx, feature_spec.bond_aromaticity_slice.start + aromaticity] = 1.0
            features[idx, feature_spec.bond_ring_slice.start + ring] = 1.0
            assert graph_bert_targets is not None
            graph_bert_targets["bond_type"][idx] = bond_type_class
            graph_bert_targets["conjugation"][idx] = conjugation
            graph_bert_targets["aromaticity"][idx] = aromaticity
            graph_bert_targets["ring_membership"][idx] = ring
            graph_bert_targets["stereo"][idx] = stereo_class
    return features, edge_index, targets, graph_bert_targets


def _index_or_unknown(value: Any, values: list[Any], unknown_at_end: bool = True) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values) if unknown_at_end else 0


def _formal_charge_class(charge: int, feature_spec: FeatureSpec) -> int:
    clipped = max(-feature_spec.formal_charge_clip, min(feature_spec.formal_charge_clip, charge))
    return clipped + feature_spec.formal_charge_clip


def _sample_mask_indices(
    num_items: int,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if num_items == 0 or ratio <= 0:
        return torch.empty(0, dtype=torch.long)
    num_mask = max(1, int(round(num_items * ratio)))
    num_mask = min(num_items, num_mask)
    return torch.randperm(num_items, generator=generator)[:num_mask].sort().values


def _set_one_hot_mask(
    features: torch.Tensor,
    indices: torch.Tensor,
    feature_slice: slice,
    mask_index: int,
) -> None:
    features[indices, feature_slice] = 0.0
    features[indices, feature_slice.start + mask_index] = 1.0
