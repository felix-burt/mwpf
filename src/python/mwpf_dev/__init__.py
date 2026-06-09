from . import mwpf as mwpf
from .mwpf import *


__doc__ = mwpf.__doc__
if hasattr(mwpf, "__all__"):
    __all__ = mwpf.__all__  # type: ignore

try:
    from .sinter_decoders import *
    from . import heralded_dem
    from . import ref_circuit
    try:
        from .pymatching_decoders import (
            SinterPyMatchingHeraldedDecoder,
            PyMatchingHeraldedCompiledDecoder,
        )
        from . import pymatching_decoders
    except ImportError:
        pass
except BaseException as e:
    raise e
