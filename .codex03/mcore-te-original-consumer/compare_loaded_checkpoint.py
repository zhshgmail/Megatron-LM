"""Physical full-checkpoint value measurement; no training/repair/retry routing.

Run later in the supplied image with two readonly checkpoint directories.
Trajectory use: both sides are actual final iteration1 checkpoints of the two matched-horizon arms.
This is final-state comparison, not immediate post-load state observation.
Uses the original MCore DCP reader; compares one global key at a time on CPU.
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--loaded-snapshot", required=True)
    parser.add_argument("--mcore-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-iteration", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, args.mcore_root)
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core.dist_checkpointing import serialization as ser
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.strategies import torch as torch_strategy

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events = (output / "events.jsonl").open("x", encoding="utf-8")

    def emit(**row):
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        events.write(line + "\n")
        events.flush()

    # These are measured import identities, not assumed from a directory or image label.
    for module in (torch, ser, torch_strategy, sys.modules[ShardedTensor.__module__]):
        path = Path(inspect.getfile(module)).resolve()
        emit(kind="loaded_source", module=module.__name__, path=str(path),
             bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    emit(kind="runtime", python=sys.version, torch=torch.__version__,
         left=args.input, right=args.loaded_snapshot)
    dist.init_process_group("gloo", init_method=(output / "gloo-init").resolve().as_uri(),
                            rank=0, world_size=1)

    def tensor_equal(left, right):
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        a, b = left.reshape(-1), right.reshape(-1)
        return all(torch.equal(a[i:i + 262144], b[i:i + 262144])
                   for i in range(0, a.numel(), 262144))

    def class_identity(value):
        if not isinstance(value, type):
            return {"is_class": False,
                    "value_type": f"{type(value).__module__}.{type(value).__qualname__}"}
        module_name = value.__module__
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None) if module is not None else None
        identity = {"is_class": True, "module": module_name,
                    "qualname": value.__qualname__, "symbol": f"{module_name}.{value.__qualname__}",
                    "module_loaded": module is not None, "module_file": module_file,
                    "module_source_status": "NO_MODULE_FILE" if module_file is None else "NOT_READ"}
        if module_file is not None:
            try:
                path = Path(module_file).resolve()
                body = path.read_bytes()
                identity.update(module_file=str(path), module_file_bytes=len(body),
                                module_file_sha256=hashlib.sha256(body).hexdigest(),
                                module_source_status="READ")
            except (OSError, ValueError, TypeError) as error:
                identity.update(module_source_status="READ_ERROR",
                                module_source_error=f"{type(error).__name__}: {error}")
        return identity

    def equal(left, right, *, key, path=()):
        # Pickle may restore class-valued optimizer defaults. Compare the actual
        # objects in this process, never merely their qualified names.
        if isinstance(left, type) or isinstance(right, type):
            matches = left is right
            emit(kind="class_state_leaf", key=key, nested_path=list(path),
                 input=class_identity(left), loaded_snapshot=class_identity(right),
                 same_identity=matches, comparison="left is right")
            return matches
        if type(left) is not type(right):
            return False
        if isinstance(left, torch.Tensor):
            return tensor_equal(left, right)
        if isinstance(left, np.ndarray):
            return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                equal(left[k], right[k], key=key,
                      path=path + ({"dict_key": repr(k),
                                    "key_type": f"{type(k).__module__}.{type(k).__qualname__}"},))
                for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                equal(a, b, key=key, path=path + ({"sequence_index": i},))
                for i, (a, b) in enumerate(zip(left, right)))
        if isinstance(left, (str, int, float, complex, bool, bytes, np.generic)) or left is None:
            return bool(left == right)
        leaf_type = f"{type(left).__module__}.{type(left).__qualname__}"
        emit(kind="unsupported_state_leaf", key=key, nested_path=list(path),
             input_type=leaf_type,
             loaded_snapshot_type=f"{type(right).__module__}.{type(right).__qualname__}")
        raise TypeError(f"Unmeasured state leaf type: {leaf_type}; "
                        f"object key={key!r}; nested path={list(path)!r}")

    def tensor_sha(value):
        flat = value.detach().contiguous().reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for start in range(0, flat.numel(), 1048576):
            digest.update(flat[start:start + 1048576].numpy().tobytes())
        return digest.hexdigest()

    def tensor_numerics(left, right, key):
        # Measurement only. Float64 reductions per bounded chunk; compensated
        # Python binary64 accumulation across chunks. No tolerance verdict.
        def unavailable(reason):
            emit(kind="tensor_numerics_unmeasured", key=key, reason=reason)
            raise TypeError(reason)
        if left.dtype != right.dtype or left.shape != right.shape:
            unavailable("unaligned dtype/shape")
        if left.device.type != "cpu" or right.device.type != "cpu" or not left.is_contiguous() or not right.is_contiguous():
            unavailable("requires contiguous CPU original global loads")
        if left.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            unavailable("unsupported numerical dtype: " + str(left.dtype))
        counts = dict(left_finite=0, right_finite=0, left_nan=0, right_nan=0,
                      left_inf=0, right_inf=0, finite_paired=0, unequal_elements=0)
        totals = {k: [0.0, 0.0] for k in ("sum_abs_diff", "sum_squared_diff",
                  "left_squared_norm", "right_squared_norm",
                  "paired_left_squared_norm", "paired_right_squared_norm")}
        extrema = {k: None for k in ("left_min", "left_max", "right_min", "right_max", "diff_min", "diff_max", "max_abs_diff")}
        def add(name, value):
            if not math.isfinite(value):
                unavailable("FP64 reduction overflow: " + name)
            total, correction = totals[name]
            new = total + value
            correction += ((total - new) + value) if abs(total) >= abs(value) else ((value - new) + total)
            if not math.isfinite(new) or not math.isfinite(correction):
                unavailable("FP64 accumulator overflow: " + name)
            totals[name] = [new, correction]
        def bounds(prefix, x):
            if x.numel():
                lo, hi = float(x.min()), float(x.max())
                extrema[prefix + "_min"] = lo if extrema[prefix + "_min"] is None else min(extrema[prefix + "_min"], lo)
                extrema[prefix + "_max"] = hi if extrema[prefix + "_max"] is None else max(extrema[prefix + "_max"], hi)
        a, b = left.view(-1), right.view(-1)
        for start in range(0, a.numel(), 262144):
            x, y = a[start:start + 262144].to(torch.float64), b[start:start + 262144].to(torch.float64)
            fx, fy = torch.isfinite(x), torch.isfinite(y)
            counts["unequal_elements"] += int(torch.count_nonzero(x != y))
            for name, z, mask in (("left", x, fx), ("right", y, fy)):
                counts[name + "_finite"] += int(mask.sum())
                counts[name + "_nan"] += int(torch.isnan(z).sum())
                counts[name + "_inf"] += int(torch.isinf(z).sum())
                finite = z[mask]
                bounds(name, finite)
                add(name + "_squared_norm", float(torch.sum(finite * finite)))
            paired = fx & fy
            counts["finite_paired"] += int(paired.sum())
            px, py = x[paired], y[paired]
            delta = px - py
            if not bool(torch.isfinite(delta).all()):
                unavailable("FP64 finite-pair subtraction overflow")
            bounds("diff", delta)
            if delta.numel():
                maximum = float(delta.abs().max())
                extrema["max_abs_diff"] = maximum if extrema["max_abs_diff"] is None else max(extrema["max_abs_diff"], maximum)
            add("sum_abs_diff", float(delta.abs().sum()))
            add("sum_squared_diff", float((delta * delta).sum()))
            add("paired_left_squared_norm", float((px * px).sum()))
            add("paired_right_squared_norm", float((py * py).sum()))
        sums = {k: math.fsum(v) for k, v in totals.items()}
        n = counts["finite_paired"]
        numerator = math.sqrt(sums["sum_squared_diff"])
        denominator = math.sqrt(sums["paired_left_squared_norm"])
        emit(kind="tensor_numerics", key=key, shape=list(left.shape), dtype=str(left.dtype),
             numel=left.numel(), chunk_elements=262144, **counts, **extrema, **sums,
             mean_abs_diff=sums["sum_abs_diff"] / n if n else None,
             rms_diff=math.sqrt(sums["sum_squared_diff"] / n) if n else None,
             relative_l2_numerator=numerator, relative_l2_denominator=denominator,
             relative_l2=numerator / denominator if n and denominator else None,
             difference_domain="both operands finite; delta=left-right",
             norm_domain="per-side finite; paired norms use both-finite intersection",
             unequal_rule="all elements; NaN unequal, signed zeros equal",
             accumulation="FP64 chunk reductions plus compensated Python binary64 totals",
             numerical_acceptance=None)

    def load_one(metadata, key, root):
        spec = copy.deepcopy(metadata[key])
        if isinstance(spec, ShardedTensor):
            # load_sharded_metadata builds from the saved global tp.size with no
            # rank offsets. Measure the whole saved tensor, not an original shard.
            emit(kind="tensor_load_geometry", checkpoint=root, key=key,
                 local_shape=list(spec.local_shape), global_shape=list(spec.global_shape),
                 global_offset=list(spec.global_offset),
                 axis_fragmentations=list(spec.axis_fragmentations),
                 flattened_range=repr(spec.flattened_range), prepend_axis_num=spec.prepend_axis_num)
            assert tuple(spec.local_shape) == tuple(spec.global_shape), "Partial tensor load spec"
            assert not any(spec.global_offset) and spec.flattened_range is None
            assert spec.prepend_axis_num == 0 and all(x == 1 for x in spec.axis_fragmentations)
            spec.init_data("cpu")
        loaded = ser.load({key: spec}, root, validate_access_integrity=False)
        value = loaded[key]
        if isinstance(spec, ShardedTensor):
            assert tuple(value.shape) == tuple(spec.global_shape), "Loaded tensor is not full global shape"
        return value

    try:
        left_meta = ser.load_sharded_metadata(args.input)
        right_meta = ser.load_sharded_metadata(args.loaded_snapshot)
        assert left_meta.keys() == right_meta.keys(), "Saved tensor/object key sets differ"
        tensor_keys = [k for k in sorted(left_meta) if isinstance(left_meta[k], ShardedTensor)]
        object_keys = [k for k in sorted(left_meta) if isinstance(left_meta[k], ShardedObject)]
        assert len(tensor_keys) + len(object_keys) == len(left_meta), "Unsupported metadata kind"
        assert tensor_keys, "No saved model/optimizer tensors were exposed"
        # Original dp_reshardable publisher stores bucket-centric main/master
        # param and moments under optimizer.distributed...bucket_idx_N.STATE.
        buckets = [k for k in tensor_keys
                   if k.startswith("optimizer.distributed.dp_group_idx_")
                   and ".gbuf_idx_" in k and ".bucket_idx_" in k]
        moments = [k for k in buckets if k.endswith(".exp_avg")]
        second_moments = [k for k in buckets if k.endswith(".exp_avg_sq")]
        masters = [k for k in buckets if k.endswith(".param")]
        rng_keys = [k for k in object_keys if left_meta[k].key == "rng_state"]
        emit(kind="inventory", tensor_keys=tensor_keys, object_keys=object_keys,
             first_moment_keys=moments, second_moment_keys=second_moments,
             master_keys=masters, training_rng_keys=rng_keys)
        assert moments and second_moments and masters, "Required optimizer state was not exposed; not a fidelity pass"
        assert rng_keys, "Training RNG objects absent; not a fidelity pass"
        negative_key = min(moments, key=lambda k: left_meta[k].global_shape.numel()
                           if hasattr(left_meta[k].global_shape, "numel")
                           else np.prod(left_meta[k].global_shape))
        total_numel = total_bytes = 0
        all_equal = True
        all_storage_equal = True
        negative_detected = False
        for key in tensor_keys:
            left = load_one(left_meta, key, args.input)
            right = load_one(right_meta, key, args.loaded_snapshot)
            assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), key
            matches = tensor_equal(left, right)
            left_sha, right_sha = tensor_sha(left), tensor_sha(right)
            total_numel += left.numel()
            total_bytes += left.numel() * left.element_size()
            all_equal = all_equal and matches
            all_storage_equal = all_storage_equal and left_sha == right_sha
            emit(kind="tensor", key=key, shape=list(left.shape), dtype=str(left.dtype),
                 numel=left.numel(), bytes=left.numel() * left.element_size(),
                 equal=matches, input_sha256=left_sha, loaded_sha256=right_sha)
            tensor_numerics(left, right, key)
            if key == negative_key:
                original = left.reshape(-1)[:262144]
                wrong = original.clone()
                assert wrong.numel(), "Negative-control moment tensor is empty"
                assert torch.isfinite(wrong[0]), "Negative-control moment value is not finite"
                wrong[0] = -wrong[0] if wrong[0] != 0 else 1
                assert wrong.dtype == original.dtype and wrong.shape == original.shape
                negative_detected = not tensor_equal(original, wrong)
                emit(kind="negative_wrong_moment", key=key, same_dtype=True, same_shape=True,
                     semantic_comparator_rejected=negative_detected)
                del wrong, original
            del left, right
            gc.collect()
        object_equal = True
        for key in object_keys:
            left = load_one(left_meta, key, args.input)
            right = load_one(right_meta, key, args.loaded_snapshot)
            matches = equal(left, right, key=key)
            object_equal = object_equal and matches
            emit(kind="sharded_object", key=key, equal=matches,
                 is_training_rng=key in rng_keys)
            del left, right

        left_common = ser.load_common_state_dict(args.input)
        right_common = ser.load_common_state_dict(args.loaded_snapshot)
        assert left_common.keys() == right_common.keys(), "Common-state key sets differ"
        assert "optimizer" in left_common and "opt_param_scheduler" in left_common
        assert left_common["iteration"] == right_common["iteration"] == args.expected_iteration, \
            "Common iteration differs from the expected comparison iteration"
        common_equal = True
        for key in sorted(left_common):
            if key == "args":
                continue
            matches = equal(left_common[key], right_common[key], key=key)
            common_equal = common_equal and matches
            emit(kind="common_state", key=key, equal=matches)
        a, b = vars(left_common["args"]), vars(right_common["args"])
        # Argument metadata is observational. Agent interpretation, not a Python
        # whitelist, distinguishes intentional path/horizon changes from concerns.
        for key in sorted(a.keys() | b.keys()):
            left_present, right_present = key in a, key in b
            left_repr = repr(a[key]) if left_present else None
            right_repr = repr(b[key]) if right_present else None
            if left_present != right_present or left_repr != right_repr:
                emit(kind="argument_difference", key=key, input_present=left_present,
                     loaded_snapshot_present=right_present, input=left_repr,
                     loaded_snapshot=right_repr, comparison="repr display only")
        emit(kind="summary", tensor_count=len(tensor_keys), object_count=len(object_keys),
             tensor_numel=total_numel, logical_tensor_bytes=total_bytes,
             all_tensor_values_equal=all_equal, all_tensor_storage_equal=all_storage_equal,
             all_sharded_objects_equal=object_equal, all_common_state_equal=common_equal,
             metadata_requires_agent_interpretation=True,
             wrong_moment_discriminator=negative_detected,
             trajectory="NOT_MEASURED")
        assert all_equal and object_equal and common_equal, "Loaded-state values differ"
        assert negative_detected, "Wrong-moment semantic discriminator did not reject"
    finally:
        dist.destroy_process_group()
        events.close()


if __name__ == "__main__":
    main()
