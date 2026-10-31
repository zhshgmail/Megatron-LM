# Source from RUN_D.sh or RUN_E.sh. No installation, backend substitution or import bypass.
: "${MCORE_SOURCE_ROOT:?UNBOUND_SELECTED_SOURCE_533cf}"
: "${CANDIDATE_SITE_ROOT:?UNBOUND_SELECTED_TE_SITE}"
: "${ADAPTOR_ROOT:?UNBOUND_ORIGINAL_ADAPTOR_f707}"
: "${RUNTIME_DEPENDENCY_ROOT:?UNBOUND_READONLY_PYDANTIC_WHITELIST}"
: "${CANN_VENDOR_PYTHONPATH:?UNBOUND_ACTUAL_VENDOR_PATHS}"
# The provider is a filtered mount, not the complete user site. A complete user site
# also contains legacy megatron/transformer_engine and must not shadow this cohort.
for package in pydantic pydantic_core annotated_types typing_inspection; do
    test -f "$RUNTIME_DEPENDENCY_ROOT/$package/__init__.py" || {
        printf '%s\n' "Missing selected runtime dependency: $package" >&2; return 2;
    }
done
for forbidden in megatron transformer_engine torch torch_npu sitecustomize.py usercustomize.py; do
    test ! -e "$RUNTIME_DEPENDENCY_ROOT/$forbidden" || {
        printf '%s\n' "Unfiltered runtime dependency root contains: $forbidden" >&2; return 2;
    }
done
export PYTHONPATH="$MCORE_SOURCE_ROOT:$CANDIDATE_SITE_ROOT:$ADAPTOR_ROOT:$RUNTIME_DEPENDENCY_ROOT:$CANN_VENDOR_PYTHONPATH"
