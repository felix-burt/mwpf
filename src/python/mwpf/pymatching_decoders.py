"""
PyMatching MWPM decoder integration for the Heralded Detector Error Model.

The :class:`HeraldedDetectorErrorModel` builds:

* a *skeleton* DEM in which every heralded error is replaced by a tiny
  ``false_positive_rate`` noise instruction, so the decoding graph contains
  every edge that may ever be needed; and
* a per-herald ``herald_fault_map`` describing — for each herald detector —
  which skeleton edges should be reweighted (and which observables they flip)
  when that herald fires.

PyMatching has no native herald awareness, but it does support per-edge
mutation through ``Matching.add_edge(..., merge_strategy='replace')``.  We
therefore build one base :class:`pymatching.Matching` object from the
skeleton DEM and, for each shot, locally swap the affected edges in,
decode, and swap them back out.

Requirements:

* the skeleton DEM must be a graph (every error touches ≤ 2 detectors); and
* every herald sub-DEM must already be present in the skeleton DEM as an
  edge with the same detector set (this is asserted at build time by
  :meth:`HeraldedDetectorErrorModel.heralded_dems`, so it is guaranteed
  when the heralded DEM was constructed successfully).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import math
import numpy as np
import stim

from .heralded_dem import HeraldedDetectorErrorModel
from .ref_circuit import RefCircuit, probability_to_weight


def _fault_ids_set(fault_mask: int) -> set:
    """Convert a bitmask of observable ids to a set of indices (PyMatching API)."""
    s: set = set()
    i = 0
    while fault_mask:
        if fault_mask & 1:
            s.add(i)
        fault_mask >>= 1
        i += 1
    return s


@dataclass
class _BaseEdge:
    """Snapshot of an edge in the skeleton matching graph (used to restore)."""

    endpoints: Tuple[int, ...]  # length 1 (boundary) or 2
    fault_ids: set
    weight: float


@dataclass
class PyMatchingHeraldedCompiledDecoder:
    """
    Sinter-compatible compiled decoder that performs PyMatching MWPM with
    per-shot edge reweighting driven by heralded detector bits.
    """

    matching: Any  # pymatching.Matching
    base_edges: Tuple[_BaseEdge, ...]
    herald_fault_map: Tuple[Any, ...]  # tuple[frozendict[int, (p, fault_mask)]]
    num_dets: int
    num_obs: int
    herald_mask: np.ndarray  # bool, len=num_dets
    det_to_herald_id: np.ndarray  # intp, len=num_dets

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]
        num_obs_bytes = (self.num_obs + 7) // 8
        predictions = np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)
        # working syndrome buffer with herald bits zeroed out (matching only sees defects)
        for shot in range(num_shots):
            dets_bit_packed = bit_packed_detection_event_data[shot]
            unpacked = np.unpackbits(
                dets_bit_packed, count=self.num_dets, bitorder="little"
            ).astype(np.uint8, copy=False)
            active = np.flatnonzero(unpacked)
            is_herald = self.herald_mask[active]
            fired_herald_ids = self.det_to_herald_id[active[is_herald]].tolist()

            # zero out herald detector bits — matching graph has no nodes there
            defect_syndrome = unpacked.copy()
            defect_syndrome[active[is_herald]] = 0

            # apply per-herald edge reweighting
            touched: list[int] = []
            for herald_id in fired_herald_ids:
                for edge_index, (p, fault_mask) in self.herald_fault_map[
                    herald_id
                ].items():
                    self._set_edge(edge_index, fault_mask, probability_to_weight(p))
                    touched.append(edge_index)

            try:
                obs_pred = self.matching.decode(defect_syndrome)
            finally:
                # restore baseline edges
                for edge_index in touched:
                    base = self.base_edges[edge_index]
                    self._set_edge_raw(edge_index, base.fault_ids, base.weight)

            obs_int = 0
            for i, bit in enumerate(obs_pred):
                if bit:
                    obs_int |= 1 << i
            predictions[shot] = np.frombuffer(
                int(obs_int).to_bytes(num_obs_bytes, byteorder="little"),
                dtype=np.uint8,
            )
        return predictions

    def _set_edge(self, edge_index: int, fault_mask: int, weight: float) -> None:
        self._set_edge_raw(edge_index, _fault_ids_set(fault_mask), weight)

    def _set_edge_raw(self, edge_index: int, fault_ids: set, weight: float) -> None:
        endpoints = self.base_edges[edge_index].endpoints
        if len(endpoints) == 1:
            self.matching.add_boundary_edge(
                endpoints[0],
                fault_ids=fault_ids,
                weight=weight,
                merge_strategy="replace",
            )
        else:
            self.matching.add_edge(
                endpoints[0],
                endpoints[1],
                fault_ids=fault_ids,
                weight=weight,
                merge_strategy="replace",
            )


@dataclass
class SinterPyMatchingHeraldedDecoder:
    """
    Sinter decoder that runs PyMatching MWPM on the skeleton of a heralded
    detector error model, reweighting affected edges per shot based on the
    fired herald bits.

    Usage with sinter mirrors :class:`SinterMWPFDecoder`:

        decoder = SinterPyMatchingHeraldedDecoder().with_circuit(circuit)
        sinter.collect(..., custom_decoders={"pymatching_heralded": decoder})

    The ``circuit`` is required (the heralded DEM is built from it); calling
    ``compile_decoder_for_dem`` without first attaching a circuit will raise.
    """

    circuit: Optional[stim.Circuit] = None
    pass_circuit: bool = True
    false_positive_rate: float = 1e-15
    with_progress: bool = False

    @property
    def config(self) -> dict[str, Any]:
        return dict(false_positive_rate=self.false_positive_rate)

    def with_circuit(
        self, circuit: stim.Circuit | None
    ) -> "SinterPyMatchingHeraldedDecoder":
        if circuit is None:
            self.circuit = None
            return self
        assert isinstance(circuit, stim.Circuit)
        self.circuit = circuit.copy()
        return self

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> PyMatchingHeraldedCompiledDecoder:
        try:
            import pymatching
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pymatching is required for SinterPyMatchingHeraldedDecoder; "
                "install it via `pip install pymatching`."
            ) from exc

        assert (
            self.circuit is not None
        ), "SinterPyMatchingHeraldedDecoder requires a circuit; call .with_circuit(...)"

        heralded_dem = HeraldedDetectorErrorModel.of(
            self.circuit, false_positive_rate=self.false_positive_rate
        )
        skeleton_dem = heralded_dem.skeleton_dem
        num_dets = skeleton_dem._dem.num_detectors
        num_obs = skeleton_dem._dem.num_observables

        assert dem.num_detectors == num_dets, (
            "Mismatched detector count between the supplied DEM and the heralded "
            "DEM derived from the circuit; ensure the same circuit is used."
        )
        assert dem.num_observables == num_obs, (
            "Mismatched observable count; ensure the same circuit is used."
        )

        # Build matching from the skeleton stim.DetectorErrorModel so that
        # pymatching's internal node count matches num_detectors exactly
        # (even for detectors that have no edges, e.g., herald detectors).
        skeleton_stim_dem = skeleton_dem.dem()
        matching = pymatching.Matching.from_detector_error_model(skeleton_stim_dem)

        # We also need a 1:1 lookup: edge_index (in skeleton_dem.hyperedges) →
        # endpoint tuple, so we can reweight via add_edge(merge_strategy='replace').
        base_edges_list: list[_BaseEdge] = []
        for hyperedge in skeleton_dem.hyperedges:
            detectors = sorted(hyperedge.detectors)
            assert 1 <= len(detectors) <= 2, (
                f"PyMatching requires graph DEMs (≤2 detectors per error); "
                f"got hyperedge with detectors={detectors}. The circuit is not "
                f"matchable; use the mwpf decoder instead."
            )
            fault_mask = sum(1 << k for k in hyperedge.observables)
            base_edges_list.append(
                _BaseEdge(
                    endpoints=tuple(detectors),
                    fault_ids=_fault_ids_set(fault_mask),
                    weight=float(probability_to_weight(hyperedge.probability)),
                )
            )

        # precompute herald lookup arrays (mirrors HeraldedDemPredictor)
        herald_mask = np.zeros(num_dets, dtype=bool)
        det_to_herald_id = np.zeros(num_dets, dtype=np.intp)
        for det_id, herald_id in heralded_dem.detector_id_to_herald_id.items():
            herald_mask[det_id] = True
            det_to_herald_id[det_id] = herald_id

        return PyMatchingHeraldedCompiledDecoder(
            matching=matching,
            base_edges=tuple(base_edges_list),
            herald_fault_map=tuple(heralded_dem.herald_fault_map),
            num_dets=num_dets,
            num_obs=num_obs,
            herald_mask=herald_mask,
            det_to_herald_id=det_to_herald_id,
        )

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path,
        dets_b8_in_path,
        obs_predictions_b8_out_path,
        tmp_dir,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        compiled = self.compile_decoder_for_dem(dem=dem)
        num_det_bytes = math.ceil(num_dets / 8)
        with open(dets_b8_in_path, "rb") as dets_in_f, open(
            obs_predictions_b8_out_path, "wb"
        ) as obs_out_f:
            for _ in range(num_shots):
                buf = np.fromfile(dets_in_f, dtype=np.uint8, count=num_det_bytes)
                if buf.shape != (num_det_bytes,):
                    raise IOError("Missing dets data.")
                pred = compiled.decode_shots_bit_packed(
                    bit_packed_detection_event_data=buf[np.newaxis, :]
                )
                obs_out_f.write(pred[0].tobytes())
