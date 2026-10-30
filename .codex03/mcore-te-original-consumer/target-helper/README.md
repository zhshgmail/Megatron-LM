# Original MCore D/E dataset-helper preparation

Use the same Python executable as the selected target D/E consumer. The original Makefile compiles helpers.cpp using Python and pybind11 headers; it has no Torch C++ linkage or NPU calls. The Python extension still requires the correct Python minor ABI, CPU architecture and target C++ runtime. Existing headers/compiler availability is a real target preflight condition, not established by local success.

The CP311 x86 local helper is not a CP312 AArch64 supply. Do not copy it to the target. Do not reuse the historical K1 wheel helper solely because its filename matches.

```bash
"$PYTHON" prepare_helpers.py --source "$MCORE_SOURCE_ROOT" --output "$HELPER_BUILD_ROOT"
"$PYTHON" prepare_helpers.py --source "$MCORE_SOURCE_ROOT" --output "$HELPER_BUILD_ROOT" --build
```

Both commands require target CP312/AArch64 by default and validate the original source hashes in SOURCE_FILES.json. Build uses the unchanged Makefile in an isolated output directory with explicit selected-Python include and extension suffix. No pip/install, Torch import, TorchNPU import or device initialization occurs. Only the second command compiles and imports the genuine extension. Root must then bind/copy that exact verified helper into the authoritative dataset module directory before the original D/E consumers, and record the final import path/hash. It must not shadow the selected MCore source with an old wheel.

Required inputs: selected source39ea3df52d520f102e2ffe30afc7c45b22a9df0e (R1), original complete WikiText103 indexed + GPT2 tokenizer/cache, unchanged original observer/strict reader and Adaptor f707. Target TE Python byte cohort/source Git identity and actual native ABI remain separately qualified or UNKNOWN. This helper build cannot qualify TE, optimizer restore, model numerics or whole software.

After placing the exact helper binary in the selected source directory, use the unchanged D/E training arguments with source first in PYTHONPATH, then verify without importing Torch:

```bash
PYTHONPATH="$MCORE_SOURCE_ROOT:$CANDIDATE_SITE_ROOT:$ADAPTOR_ROOT:$CANN_VENDOR_PYTHONPATH" "$PYTHON" check_selected_source.py --source "$MCORE_SOURCE_ROOT" --helper-receipt "$HELPER_BUILD_ROOT/BUILD_RESULT.json" --output "$HELPER_BUILD_ROOT/SELECTED_SOURCE_CHECK.json"
```

The gate verifies selected source body hashes, Python extension suffix, exact authoritative core/helper paths and copied helper hash. It does not replace module import/target D/E validation. The prepared caller puts source first; metadata/source parity and helper build leave R1 optimizer restore NOT_RUN until real original D/E/strict.
