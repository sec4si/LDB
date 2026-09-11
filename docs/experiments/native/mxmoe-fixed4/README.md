# Corrected mixed runtime source snapshot

This snapshot preserves the exact fixed4 header overlay and generated sources.
The library hash is in `manifest.json`; weight packing is unchanged.

`build.py` records the original local build, using the existing CUDA 13.0 compiler,
configured MxMoE CMake include/flags files, and compiled Python bridge object.
Its output/source path is `/tmp/ldb-mixed-activation-fixed4-20260911`; recreate
that tree from this snapshot before replaying it. This is a reproducibility
record for this environment, not a portable installer.

Changes include the earlier INT8 scale-column and fragment-count fixes,
split-K shared-memory lifetime barriers, and persistent tile boundaries.
See adjacent racecheck evidence before drawing correctness/performance conclusions.
