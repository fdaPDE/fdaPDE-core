#!/usr/bin/env python3
"""Build and compare native loop selections using only the Python standard library."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

MODES = {"off": (0, 0), "assignment": (1, 0), "product": (0, 1), "all": (1, 1)}
ASSIGNMENT_CASES = [f"{layout}_{operation}" for layout in ("vector", "row3", "col3", "view")
                    for operation in ("scale", "copy", "broadcast", "add", "affine")]
ASSIGNMENT_CASES += ["row3_float_scale", "vector_float_affine"]
ASSIGNMENT_CONTROLS = ["vector_orientation_copy", "cross_layout_copy"]
PRODUCT_CASES = ["square_row", "square_col", "square_float", "square_float_col", "mixed_row",
                 "mixed_col", "rectangular", "odd", "views", "construct_col_mixed"]
STATIC_CASES = ["fem3", "fem4", "fem10"]
ASSIGNMENT_SIZES = [3 * (2 ** k + 1) for k in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 24)]
PRODUCT_SIZES = [3, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]


def run_json(command, timeout, allow_zero=False):
    """Run one isolated process and require its release correctness checks to pass."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    row = json.loads(result.stdout)
    if not row.get("verified") or row.get("error") is not None or row["median_ns"] < 0 or (row["median_ns"] == 0 and not allow_zero):
        raise RuntimeError("benchmark did not return a verified positive measurement")
    return row


def paired_summary(rows):
    """Keep process pairs as the statistical units rather than pooling timed rounds."""
    ratios = [item["off"]["median_ns"] / item["on"]["median_ns"] for item in rows]
    return {"ratio": statistics.median(ratios), "ratio_min": min(ratios), "ratio_max": max(ratios),
            "off_ns": statistics.median(item["off"]["median_ns"] for item in rows),
            "on_ns": statistics.median(item["on"]["median_ns"] for item in rows), "pairs": rows}


def plateau(history, minimum_buffer_bytes):
    """Require three consistent large-regime points and a fourth confirmation point."""
    if len(history) < 4:
        return False
    window = history[-4:]
    if any(item["buffer_bytes"] < minimum_buffer_bytes for item in window):
        return False
    ratios = [item["ratio"] for item in window]
    if max(ratios) / min(ratios) > 1.05:
        return False
    return all(item["ratio_max"] / item["ratio_min"] <= 1.10 for item in window)


def compile_command(compiler, repo, binary, mode):
    """Keep the expected compiler flags identical for fresh builds and manifest reuse checks."""
    assignment, product = MODES[mode]
    flags = ["-std=c++20", "-O3", "-DNDEBUG", "-DFDAPDE_NO_DEBUG", "-ffp-contract=off",
             "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-isystem", str(repo)]
    return [compiler, *flags, f"-DFDAPDE_ENABLE_SIMD_ASSIGNMENT={assignment}",
            f"-DFDAPDE_ENABLE_SIMD_PRODUCT={product}",
            str(repo / "tests/benchmarks/simd_sweep.cpp"), "-o", str(binary)]


def build(compiler, repo, output):
    """Build four binaries from identical source with only the two loop flags changed."""
    binaries = {}
    for name in MODES:
        binary = output / ("sweep-" + name)
        command = compile_command(compiler, repo, binary, name)
        with (output / (name + "-build.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[name] = binary
        (output / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
    return binaries


def command(binary, suite, case, size, repetitions, rounds):
    """Specify public workload shape and identical batch counts for the compared binaries."""
    return [str(binary), "--suite", suite, "--case", case, "--size", str(size),
            "--repetitions", str(repetitions), "--rounds", str(rounds)]


def calibrate(binaries, modes, suite, case, size, args):
    """Choose one shared repetition count using untimed independent calibration processes."""
    probes = []
    for mode in modes:
        probe_repetitions = 1
        while True:
            probe = run_json(command(binaries[mode], suite, case, size, probe_repetitions, 1), args.timeout, allow_zero=True)
            if probe["median_ns"] > 0:
                probes.append(probe)
                break
            if probe_repetitions >= args.max_repetitions:
                raise RuntimeError("calibration remains below clock resolution at the repetition limit")
            probe_repetitions = min(args.max_repetitions, probe_repetitions * 16)
    slowest = max(item["median_ns"] for item in probes)
    fastest = min(item["median_ns"] for item in probes)
    repetitions = max(1, min(args.max_repetitions, int(args.round_ms * 1e6 / max(fastest, 1)),
                             int(args.max_round_ms * 1e6 / max(slowest, 1))))
    return repetitions, probes


def measure(binaries, before, after, suite, case, size, index, args, raw, repetitions=None, rounds=None):
    """Alternate process order and retain each verified process output verbatim."""
    if repetitions is None:
        repetitions, probes = calibrate(binaries, (before, after), suite, case, size, args)
    else:
        probes = []
    if rounds is None:
        rounds = args.large_rounds if probes and max(probe["median_ns"] for probe in probes) > args.max_round_ms * 1e6 else args.rounds
    pairs = []
    for pair in range(args.pairs):
        modes = (before, after) if (index + pair) % 2 == 0 else (after, before)
        pair_rows = {}
        for mode in modes:
            result = run_json(command(binaries[mode], suite, case, size, repetitions, rounds), args.timeout)
            expected = MODES[mode]
            if (result["assignment"], result["product"]) != expected:
                raise RuntimeError("binary reported unexpected loop-selection flags")
            record = {"suite": suite, "case": case, "size": size, "comparison": f"{before}:{after}",
                      "pair": pair, "mode": mode, "result": result}
            raw.write(json.dumps(record) + "\n")
            raw.flush()
            pair_rows[mode] = result
        if pair_rows[before]["checksum"] != pair_rows[after]["checksum"]:
            raise RuntimeError("public outputs disagree between loop selections")
        pairs.append({"off": pair_rows[before], "on": pair_rows[after], "order": modes})
    summary = paired_summary(pairs)
    sample = pairs[0]["off"]
    summary.update({"suite": suite, "case": case, "size": size, "comparison": f"{before}:{after}",
                    "rows": sample["rows"], "inner": sample["inner"], "cols": sample["cols"],
                    "coefficients": sample["coefficients"], "buffer_bytes": sample["buffer_bytes"],
                    "working_set_bytes": sample["working_set_bytes"], "scalar": sample["scalar"],
                    "repetitions": repetitions, "rounds": rounds, "calibration": probes,
                    "lhs_order": sample["lhs_order"], "rhs_order": sample["rhs_order"],
                    "output_order": sample["output_order"], "pair_count": args.pairs,
                    "short_round": any(min(result["timings_ns"]) * repetitions < 1e6
                                       for pair in pairs for result in (pair["off"], pair["on"]))})
    return summary


def save(output, summaries, stops):
    """Save process-pair summaries and tables suitable for standard plotting tools."""
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "stops.json").write_text(json.dumps(stops, indent=2) + "\n")
    fields = ["compiler", "suite", "case", "size", "comparison", "rows", "inner", "cols", "scalar",
              "coefficients", "buffer_bytes", "working_set_bytes", "repetitions", "rounds",
              "off_ns", "on_ns", "ratio", "ratio_min", "ratio_max", "lhs_order", "rhs_order",
              "output_order", "pair_count", "short_round"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "stops.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("compiler", "suite", "case", "reason", "last_size", "plateau_observed", "factorial_incomplete"))
        writer.writeheader()
        writer.writerows(stops)


def self_test():
    """Check pairing and plateau decisions with exact synthetic samples."""
    values = [{"off": {"median_ns": v}, "on": {"median_ns": 1}} for v in (2, 8, 3)]
    assert paired_summary(values)["ratio"] == 3, "the median paired ratio is the oracle"
    stable = [{"buffer_bytes": 32, "ratio": v, "ratio_min": v * .99, "ratio_max": v * 1.01}
              for v in (2, 2.02, 1.99, 2.01)]
    assert plateau(stable, 16), "three large points and their fourth confirmation should qualify"
    assert not plateau(stable[:3], 16), "three points without a larger confirmation must not qualify"
    assert not plateau(stable, 64), "points below the cache-size threshold are insufficient"
    stable[-1]["ratio"] = 3
    assert not plateau(stable, 16), "a changed regime must invalidate a plateau"
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="/usr/bin/clang++")
    parser.add_argument("--label", default="appleclang21")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", choices=("assignment", "product", "all"), default="all")
    parser.add_argument("--cases", help="comma-separated subset of the published case names")
    parser.add_argument("--sizes", help="comma-separated sizes; otherwise use the geometric suite schedule")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--large-rounds", type=int, default=1)
    parser.add_argument("--round-ms", type=float, default=25)
    parser.add_argument("--max-round-ms", type=float, default=250)
    parser.add_argument("--max-repetitions", type=int, default=5000000)
    parser.add_argument("--max-call-seconds", type=float, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--cache-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--anchors-only", action="store_true")
    parser.add_argument("--factorial", action="store_true")
    parser.add_argument("--reuse-binaries", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or any(value <= 0 for value in
            (args.pairs, args.rounds, args.large_rounds, args.round_ms, args.max_round_ms, args.max_repetitions,
             args.max_call_seconds, args.timeout, args.cache_bytes)):
        parser.error("output and positive timing/count/cache parameters are required")
    if args.max_repetitions > 10000000 or max(args.rounds, args.large_rounds) > 101:
        parser.error("the C++ driver supports at most 10000000 repetitions and 101 rounds")
    selected_cases = args.cases.split(",") if args.cases else None
    known_cases = ASSIGNMENT_CASES + ASSIGNMENT_CONTROLS + PRODUCT_CASES + STATIC_CASES
    if selected_cases and (len(set(selected_cases)) != len(selected_cases) or any(case not in known_cases for case in selected_cases)):
        parser.error("case names must be known and unique")
    allowed_cases = known_cases if args.suite == "all" else (ASSIGNMENT_CASES + ASSIGNMENT_CONTROLS if args.suite == "assignment" else PRODUCT_CASES + STATIC_CASES)
    if selected_cases and any(case not in allowed_cases for case in selected_cases):
        parser.error("a selected case does not belong to the requested suite")
    selected_sizes = [int(value) for value in args.sizes.split(",")] if args.sizes else None
    if selected_sizes and (selected_sizes != sorted(set(selected_sizes)) or selected_sizes[0] <= 0):
        parser.error("sizes must be positive, unique and increasing")
    if args.resume and not args.reuse_binaries:
        parser.error("resume requires reuse-binaries to preserve the measured build")
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and not args.resume and not (args.build_only and args.reuse_binaries):
        parser.error("measurement output already exists; use resume or a new output directory")
    if args.resume and (output / "metadata.json").exists():
        previous_arguments = json.loads((output / "metadata.json").read_text())["arguments"]
        ignored = {"output", "resume", "reuse_binaries", "build_only", "self_test"}
        if any(previous_arguments.get(key) != value for key, value in vars(args).items() if key not in ignored):
            parser.error("resume requires the original case selection, size schedule and timing protocol")
    version = subprocess.run([args.compiler, "--version"], text=True, capture_output=True, check=True).stdout
    source_files = sorted((repo / "fdaPDE").rglob("*.h")) + [repo / "tests/benchmarks/simd_sweep.cpp"]
    hashes = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    manifest_path = output / "build.json"
    if args.reuse_binaries:
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != hashes or manifest["compiler_version"] != version or manifest["compiler"] != args.compiler:
            parser.error("the saved binaries do not match the current source or compiler; rebuild in a new output directory")
        binaries = {name: output / ("sweep-" + name) for name in MODES}
        if any(json.loads((output / (name + "-command.json")).read_text()) !=
               compile_command(args.compiler, repo, binary, name) for name, binary in binaries.items()):
            parser.error("the saved compile commands do not match the current benchmark flags; rebuild in a new output directory")
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != manifest["binary_sha256"][name] for name, path in binaries.items()):
            parser.error("a saved benchmark binary changed since it was built")
    else:
        binaries = build(args.compiler, repo, output)
        manifest = {"compiler": args.compiler, "compiler_version": version, "source_sha256": hashes,
                    "binary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.build_only:
        print(f"built four verified configurations in {output}")
        return
    metadata = {"arguments": vars(args) | {"output": str(output)}, "compiler_version": version,
                "platform": platform.platform(), "source_sha256": hashes,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / ("resume-metadata.json" if args.resume else "metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    suites = ("assignment", "product") if args.suite == "all" else (args.suite,)
    summaries = json.loads((output / "summary.json").read_text()) if args.resume and (output / "summary.json").exists() else []
    stops = json.loads((output / "stops.json").read_text()) if args.resume and (output / "stops.json").exists() else []
    with (output / "raw.jsonl").open("a") as raw:
        for suite in suites:
            cases = ASSIGNMENT_CASES + ASSIGNMENT_CONTROLS if suite == "assignment" else PRODUCT_CASES + STATIC_CASES
            if args.cases:
                cases = [case for case in selected_cases if case in cases]
            for case in cases:
                if any(stop["suite"] == suite and stop["case"] == case for stop in stops):
                    continue
                sizes = ASSIGNMENT_SIZES if suite == "assignment" else PRODUCT_SIZES
                if selected_sizes:
                    sizes = selected_sizes
                if args.anchors_only or case in ASSIGNMENT_CONTROLS:
                    sizes = sorted(set((sizes[0], sizes[len(sizes) // 2], sizes[-1])))
                if case in STATIC_CASES:
                    sizes = [int(case[3:])]
                before, after = ("off", "assignment") if suite == "assignment" else ("assignment", "all")
                history = [row for row in summaries if row["suite"] == suite and row["case"] == case
                           and row["comparison"] == f"{before}:{after}"]
                reason = "maximum_dimension"
                for index, size in enumerate(sizes):
                    if any(row["size"] == size for row in history):
                        continue
                    # avoid launching a predictably unbounded cubic product while retaining the measured limit
                    if suite == "product" and history:
                        predicted = history[-1]["off_ns"] * (size / history[-1]["size"]) ** 3 / 1e9
                        if predicted > args.max_call_seconds:
                            reason = "predicted_call_limit"
                            break
                    print(f"{args.label} {suite} {case} size={size}: measuring", flush=True)
                    try:
                        row = measure(binaries, before, after, suite, case, size, index, args, raw)
                    except subprocess.TimeoutExpired:
                        reason = "process_timeout"
                        break
                    row["compiler"] = args.label
                    summaries.append(row)
                    history.append(row)
                    print(f"{case} size={size}: ratio={row['ratio']:.3f} [{row['ratio_min']:.3f},{row['ratio_max']:.3f}]", flush=True)
                    save(output, summaries, stops)
                    if not args.anchors_only and plateau(history, args.cache_bytes):
                        reason = "confirmed_local_plateau"
                        break
                    if row["off_ns"] / 1e9 > args.max_call_seconds:
                        reason = "measured_call_limit"
                        break
                factorial_incomplete = False
                if args.factorial and suite == "product" and history:
                    anchors = {history[0]["size"], history[-1]["size"]}
                    anchors.add(min(history, key=lambda row: abs(row["size"] - 128))["size"])
                    for row in history:
                        if row["size"] not in anchors:
                            continue
                        for b, a in (("off", "product"), ("off", "all")):
                            if any(item["suite"] == suite and item["case"] == case and item["size"] == row["size"]
                                   and item["comparison"] == f"{b}:{a}" for item in summaries):
                                continue
                            try:
                                extra = measure(binaries, b, a, suite, case, row["size"], sizes.index(row["size"]), args, raw, row["repetitions"], row["rounds"])
                            except subprocess.TimeoutExpired:
                                factorial_incomplete = True
                                print(f"{case}: factorial process timeout at size={row['size']}", flush=True)
                                break
                            extra["compiler"] = args.label
                            summaries.append(extra)
                            save(output, summaries, stops)
                        if factorial_incomplete:
                            break
                stops.append({"compiler": args.label, "suite": suite, "case": case, "reason": reason,
                              "last_size": history[-1]["size"] if history else None,
                              "plateau_observed": reason == "confirmed_local_plateau", "factorial_incomplete": factorial_incomplete})
                save(output, summaries, stops)
                print(f"{case}: {reason}", flush=True)
    print(f"saved {len(summaries)} points to {output}", flush=True)


if __name__ == "__main__":
    main()
