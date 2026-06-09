"""Tests for the PyMatching + Heralded DEM integration."""
import pytest

pymatching = pytest.importorskip("pymatching")

# Try the canonical package layout first, then fall back to the dev one
# (the maturin extension typically lives in `mwpf_dev` in this repo).
try:
    import mwpf  # type: ignore
    from mwpf.pymatching_decoders import SinterPyMatchingHeraldedDecoder
    from mwpf.heralded_dem import HeraldedDetectorErrorModel
except (ImportError, AttributeError):
    import mwpf_dev as mwpf  # type: ignore
    from mwpf_dev.pymatching_decoders import SinterPyMatchingHeraldedDecoder
    from mwpf_dev.heralded_dem import HeraldedDetectorErrorModel

import stim
import numpy as np


def _surface_repcode_with_herald() -> stim.Circuit:
    """Small repetition-code-style circuit that has a heralded erasure."""
    # Same shape as test_heralded_dem_simple in test_heralded_dem.py — verified
    # to satisfy HeraldedDetectorErrorModel sanity checks.
    return stim.Circuit("""\
R 0 1 2 3 4
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]
DEPOLARIZE1(0.02) 2
HERALDED_ERASE(0.05) 0
DETECTOR rec[-1]
M 0 2 4
DETECTOR(1, 1) rec[-2] rec[-3] rec[-6]
DETECTOR(3, 1) rec[-1] rec[-2] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1]
""")


def test_pymatching_heralded_decoder_basic():
    circuit = _surface_repcode_with_herald()
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)

    decoder = SinterPyMatchingHeraldedDecoder().with_circuit(circuit)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    assert compiled.num_dets == dem.num_detectors
    assert compiled.num_obs == dem.num_observables

    # sample shots and decode; just check the API runs and shapes match.
    sampler = circuit.compile_detector_sampler(seed=1234)
    dets, obs = sampler.sample(256, separate_observables=True, bit_packed=True)
    preds_packed = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )
    assert preds_packed.shape == obs.shape

    # logical error rate should be small (<50%) for this tiny benign example
    preds_unpacked = np.unpackbits(
        preds_packed, axis=1, count=dem.num_observables, bitorder="little"
    )
    obs_unpacked = np.unpackbits(
        obs, axis=1, count=dem.num_observables, bitorder="little"
    )
    err_rate = np.mean(np.any(preds_unpacked != obs_unpacked, axis=1))
    assert err_rate < 0.5, f"unexpectedly high logical error rate {err_rate}"


def test_pymatching_heralded_matches_no_herald_baseline():
    """With no heralded errors at all, the decoder must agree with a plain
    pymatching.Matching built directly from the DEM."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=5,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    decoder = SinterPyMatchingHeraldedDecoder().with_circuit(circuit)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    sampler = circuit.compile_detector_sampler(seed=2025)
    dets, obs = sampler.sample(512, separate_observables=True, bit_packed=True)

    ours = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)

    reference = pymatching.Matching.from_detector_error_model(dem)
    dets_unpacked = np.unpackbits(
        dets, axis=1, count=dem.num_detectors, bitorder="little"
    ).astype(np.uint8)
    ref_preds = np.zeros_like(ours)
    for i in range(dets_unpacked.shape[0]):
        pred = reference.decode(dets_unpacked[i])
        v = 0
        for k, b in enumerate(pred):
            if b:
                v |= 1 << k
        ref_preds[i] = np.frombuffer(
            int(v).to_bytes(ours.shape[1], byteorder="little"), dtype=np.uint8
        )

    # Logical-error agreement, not necessarily byte-for-byte equality
    # (skeleton has tiny extra weights), should be very high.
    ours_u = np.unpackbits(ours, axis=1, count=dem.num_observables, bitorder="little")
    ref_u = np.unpackbits(
        ref_preds, axis=1, count=dem.num_observables, bitorder="little"
    )
    obs_u = np.unpackbits(obs, axis=1, count=dem.num_observables, bitorder="little")
    ours_err = np.mean(np.any(ours_u != obs_u, axis=1))
    ref_err = np.mean(np.any(ref_u != obs_u, axis=1))
    assert abs(ours_err - ref_err) < 0.02, (ours_err, ref_err)


def test_pymatching_heralded_rejects_hyperedges():
    """If the skeleton DEM has a >2-detector hyperedge, the decoder must reject it."""
    # surface-code style circuit produces 4-body Y errors that decompose to hyperedges
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=2,
        distance=3,
        after_clifford_depolarization=0.005,
    )
    skel = HeraldedDetectorErrorModel.of(circuit).skeleton_dem
    if not any(len(h.detectors) > 2 for h in skel.hyperedges):
        pytest.skip("constructed circuit happened to be matchable")
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    decoder = SinterPyMatchingHeraldedDecoder().with_circuit(circuit)
    with pytest.raises(AssertionError, match="graph DEM"):
        decoder.compile_decoder_for_dem(dem=dem)
