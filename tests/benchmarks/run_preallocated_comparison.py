#!/usr/bin/env python3
# This file is part of fdaPDE, a C++ library for physics-informed
# spatial and functional data analysis.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Compare native OFF/ON with Eigen 3.4 using alternating single-thread processes."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

from run_simd_sweep import paired_summary

OPS = ('spmv', 'gemv', 'gemvt', 'project', 'momentum', 'scale', 'assemble', 'dense_construct', 'replay')
CAPSULE_FILES = ('-omega.mtx', '-c.bin', '-warm.bin', '-weight.bin', '.ready')


def digest(path):
    """hash exact source, capsule or binary bytes without changing their origin"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capsule_shape(prefix):
    """read the sparse shape before applying a local size bound, leaving coefficient validation to the driver"""
    with Path(prefix + '-omega.mtx').open() as stream:
        if stream.readline().split() != ['%%MatrixMarket', 'matrix', 'coordinate', 'real', 'general']:
            raise ValueError('capsule requires general real coordinate MatrixMarket storage')
        line = next(line for line in stream if line.strip() and not line.startswith('%'))
    rows, cols, nnz = map(int, line.split())
    if not 0 < rows == cols <= 2147483646 or not 0 < nnz <= 2147483647:
        raise ValueError('capsule has an unsupported sparse shape')
    return rows, cols


def cases(args, shapes):
    """retain full large controls while filtering both dimensions for explicitly bounded local runs"""
    result = []
    sparse_sizes = [33, 81, 426, 666, 1418, 8193] if args.quick else [33, 81, 426, 581, 666, 1418, 8193, 65537, 524289]
    for size in sparse_sizes:
        for width in ([19] if args.quick else [3, 19, 65]):
            for api in ('public', 'preallocated'):
                result.append(dict(op='spmv', rows=size, cols=size, nnz_per_row=width, api=api, offset=1))
    dense_shapes = [(33, 17), (129, 81), (1025, 1023)] if args.quick else [
        (33, 17), (129, 81), (1025, 81), (1025, 1023), (3200, 426), (3200, 1418), (4097, 1023), (4097, 4095)]
    for rows, cols in dense_shapes:
        for op in ('gemv', 'gemvt'):
            for api in ('public', 'preallocated'):
                result.append(dict(op=op, rows=rows, cols=cols, api=api, offset=1))
    for size in ([81, 1418, 65537] if args.quick else [81, 426, 1418, 65537, 1048577]):
        for op in ('project', 'momentum', 'scale'):
            for api in ('public', 'fused'):
                result.append(dict(op=op, rows=size, cols=size, api=api, offset=1))
    for size in ([426] if args.quick else [426, 1418, 65537]):
        result.append(dict(op='assemble', rows=size, cols=size, nnz_per_row=19, api='public', offset=0))
    for rows, cols in dense_shapes:
        result.append(dict(op='dense_construct', rows=rows, cols=cols, api='public', offset=0))
    for prefix, (rows, cols) in shapes.items():
        for op, apis in (('spmv', ('public', 'preallocated')), ('assemble', ('public',)), ('replay', ('preallocated',))):
            for api in apis:
                result.append(dict(op=op, rows=rows, cols=cols, input=prefix, api=api, offset=0))
    return [dict(case, seed=args.seed) for case in result if case['op'] in args.ops and
            (args.max_size is None or max(case['rows'], case['cols']) <= args.max_size)]


def command(binary, case, backend, repetitions, rounds, prefix):
    """use identical input and batch specifications for every compared implementation"""
    result = [*prefix, str(binary)]
    for key, value in dict(case, backend=backend, repetitions=repetitions, rounds=rounds).items():
        result += ['--' + key.replace('_', '-'), str(value)]
    return result


def validate(row, case, mode, backend, repetitions, rounds, allow_zero=False):
    """check the independent oracle certificate, echoed contract, raw timing and release allocation probe"""
    errors = []
    if not isinstance(row, dict):
        return ['process stdout is not a JSON object']
    if row.get('verified') is not True or 'error' not in row or row.get('error') is not None:
        errors.append('independent driver verification failed: ' + str(row.get('error')))
    oracle = 'independent-kkt' if case['op'] == 'replay' else 'canonical-construction' if case['op'] in ('assemble', 'dense_construct') else 'scalar-coefficients'
    expected = dict(op=case['op'], backend=backend, api=case['api'], rows=case['rows'], cols=case['cols'],
                    offset=case['offset'], seed=case['seed'], input=case.get('input', ''),
                    repetitions=repetitions, rounds=rounds, assignment=int(mode == 'on'), product=int(mode == 'on'),
                    eigen_version='3.4.0', internal_threads=1, oracle=oracle)
    for key, value in expected.items():
        if row.get(key) != value or isinstance(value, int) and type(row.get(key)) is not int:
            errors.append('unexpected ' + key)
    if not isinstance(row.get('input_hash'), str) or not re.fullmatch('[0-9a-fA-F]{1,16}', row['input_hash']):
        errors.append('invalid canonical input hash')
    if type(row.get('nnz')) is not int or row['nnz'] < 0:
        errors.append('invalid sparse entry count')
    for key in ('median_ns', 'max_abs_error', 'checksum', 'working_set_bytes'):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (key != 'checksum' and value < 0):
            errors.append('invalid ' + key)
    timings = row.get('timings_ns')
    if not isinstance(timings, list) or len(timings) != rounds or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in timings):
        errors.append('invalid raw timed rounds')
    elif row.get('median_ns') != sorted(timings)[len(timings) // 2] or not allow_zero and row['median_ns'] <= 0:
        errors.append('median disagrees with raw rounds or is below clock resolution')
    if backend == 'native':
        allocations = row.get('new_allocations_per_call')
        if not isinstance(allocations, int) or isinstance(allocations, bool) or allocations < 0:
            errors.append('invalid native allocation probe')
        if case['api'] == 'preallocated' and case['op'] in ('spmv', 'gemv', 'gemvt') and allocations != 0:
            errors.append('disjoint preallocated product allocated')
    elif row.get('new_allocations_per_call') is not None:
        errors.append('Eigen malloc cannot be counted as native new')
    if case['op'] == 'replay':
        replay = row.get('replay', {})
        if not isinstance(replay, dict):
            return errors + ['replay certificate is missing']
        for key in ('full_ns', 'profile_full_ns', 'profile_spmv_ns', 'raw_spmv_ns', 'timer_overhead_ns',
                    'kkt_relative', 'norm_error', 'weight_relative_error'):
            value = replay.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                errors.append('invalid replay ' + key)
        for key in ('spmv_calls', 'iterations', 'restarts', 'support_count'):
            if not isinstance(replay.get(key), int) or isinstance(replay[key], bool) or replay[key] < 0:
                errors.append('invalid replay ' + key)
        if replay.get('nonnegative') is not True or replay.get('converged') is not True:
            errors.append('replay failed nonnegativity or convergence')
        if isinstance(replay.get('kkt_relative'), (int, float)) and replay['kkt_relative'] > 1.0001e-8 or isinstance(replay.get('norm_error'), (int, float)) and replay['norm_error'] > 1e-8:
            errors.append('replay failed its independent KKT or normalization certificate')
        if not isinstance(replay.get('support_hash'), str) or not re.fullmatch('[0-9a-fA-F]{1,16}', replay['support_hash']):
            errors.append('invalid replay support hash')
        for key, count in (('final_weight', case['rows']), ('fista_timings_ns', repetitions * rounds)):
            values = replay.get(key)
            if not isinstance(values, list) or len(values) != count or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                errors.append('invalid replay ' + key)
    return errors


def invoke(cmd, env, timeout):
    """retain failed process output so a certificate failure remains visible in the raw record"""
    try:
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        observation = dict(command=cmd, exit_code=completed.returncode, stderr=completed.stderr)
        try:
            observation['result'] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            observation.update(result=None, stdout=completed.stdout)
    except (subprocess.TimeoutExpired, OSError) as error:
        observation = dict(command=cmd, exit_code=None, result=None, process_error=str(error))
        if isinstance(error, subprocess.TimeoutExpired):
            observation.update(stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
    return observation


def calibrate(observe, targets, args, replay=False):
    """double only zero-resolution probes and choose one shared batch from positive per-call timings"""
    probes = []
    for mode, backend in targets:
        repetitions = 1
        while True:
            row = observe(mode, backend, repetitions, 1, 'calibration', True)
            if row['median_ns'] > 0:
                probes.append(row)
                break
            if repetitions == args.max_repetitions:
                raise RuntimeError('calibration remains below clock resolution at the repetition limit')
            repetitions = min(args.max_repetitions, repetitions * 2)
    fastest, slowest = min(row['median_ns'] for row in probes), max(row['median_ns'] for row in probes)
    return 1 if replay else max(1, min(args.max_repetitions, int(args.round_ms * 1e6 / fastest),
                                      int(args.max_round_ms * 1e6 / slowest)))


def agreement(before, after, tolerance):
    """report input agreement and replay decisions without equating a FISTA replay to the production solver"""
    result = dict(input_hash_equal=before['input_hash'] == after['input_hash'],
                  shape_equal=(before['rows'], before['cols']) == (after['rows'], after['cols']),
                  checksum_difference=abs(before['checksum'] - after['checksum']))
    errors = []
    if not result['input_hash_equal'] or not result['shape_equal']:
        errors.append('canonical input hash or shape mismatch')
    if 'replay' in before and 'replay' in after:
        lhs, rhs = before['replay'], after['replay']
        a, b = lhs['final_weight'], rhs['final_weight']
        delta_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        norm_l2 = max(math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b)))
        norm_linf = max([abs(x) for x in a + b] or [0])
        result.update(support_agreement=(lhs['support_count'], lhs['support_hash']) == (rhs['support_count'], rhs['support_hash']),
                      iterations_agreement=lhs['iterations'] == rhs['iterations'], restarts_agreement=lhs['restarts'] == rhs['restarts'],
                      weight_relative_l2=delta_l2 / norm_l2 if norm_l2 else 0,
                      weight_relative_linf=max([abs(x - y) for x, y in zip(a, b)] or [0]) / norm_linf if norm_linf else 0)
        result['weight_agreement'] = max(result['weight_relative_l2'], result['weight_relative_linf']) <= tolerance
        if not result['support_agreement'] or not result['weight_agreement']:
            errors.append('replay support or cross-backend final-weight mismatch')
    result['errors'] = errors
    return result


def comparisons(case):
    """compare both native flags with each applicable Eigen storage format"""
    backends = ('eigen-col', 'eigen-row') if case['op'] in ('spmv', 'assemble', 'replay') else ('eigen-col',)
    return [('off', 'native', 'on', 'native'), *[(mode, 'native', 'off', backend) for mode in ('off', 'on') for backend in backends]]


def save(output, rows, manifest):
    """save readable partial results as well as machine-readable timing and replay diagnostics"""
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = ['operation', 'rows', 'cols', 'nnz_per_row', 'nnz', 'api', 'input', 'offset', 'seed', 'comparison', 'status',
              'off_ns', 'on_ns', 'ratio', 'ratio_min', 'ratio_max', 'before_new_allocations', 'after_new_allocations',
              'working_set_bytes', 'working_set_scope', 'repetitions', 'rounds', 'fista_off_ns', 'fista_on_ns', 'fista_ratio',
              'fista_ratio_min', 'fista_ratio_max', 'support_agreement', 'weight_relative_l2', 'weight_relative_linf']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, **dict(row['case'], operation=row['case']['op'])))
    lines = ['# Native OFF/ON vs Eigen 3.4', '',
             f"Status: **{manifest['status']}**; {len(rows)}/{manifest['expected_comparisons']} comparisons recorded.",
             f"Git base: `{manifest['git_commit']}`; exact headers, runner, statistics provider, Eigen, inputs and binaries are hashed in manifest.json.",
             f"Policy: {manifest['policy']}. No quiet-machine or universal ISA performance claim follows from this report.", '']
    if manifest.get('error'):
        lines += ['Error: ' + manifest['error'], '']
    lines += ['Ratios are before/after; >1 means the implementation after the colon was faster. Min/max describes process-pair dispersion, not a confidence interval.',
              'MISMATCH rows retain timings without establishing comparable inputs or replay output agreement; they support no speed conclusion.',
              'Public calls include allocation/snapshots. The untimed native new probe does not count Eigen malloc. Operands are independently prepared outside kernel timing.',
              'Native fused coefficientwise entries use assign_disjoint; Eigen fuses its coefficientwise expressions. No FISTA API is introduced.',
              'Replay main timings cover the complete harness including the final certificate; FISTA-only paired timings are separate. Profiling is a separate instrumented run with raw/adjusted SpMV time and timer overhead, not a measured production share.',
              'Captured-weight error is diagnostic, not an acceptance threshold. Cross-backend weight tolerance is ' + str(manifest['arguments']['replay_weight_tolerance']) + '; support hashes/counts are checked independently.',
              'Synthetic cases and captured capsules are distinguished by the input column. Replay covers FISTA only, not production pruning, support factorizations or migration validation.', '',
              'Dense 3200×426 and 3200×1418 controls use synthetic coefficients at application shapes; no captured X is available. Full schedules include large controls, whose logical storage estimate must be compared with the recorded machine cache before calling them beyond-cache.',
              'Working-set bytes describe the logical kernel estimate, excluding temporary/oracle storage and inactive backend representations. Replay includes ten logical FISTA vector buffers and CSR storage; it excludes raw/scaled canonical triplets and reference data walked by the complete final certificate. Complete-harness timing therefore touches more data than this kernel estimate.', '',
              '| Operation/shape | Input | API | Comparison | Status | Before ms | After ms | Ratio [min,max] | New before/after |',
              '|---|---|---|---|---|---:|---:|---|---:|']
    for row in rows:
        case = row['case']; label = f"{case['op']} {case['rows']}×{case['cols']}"
        lines.append(f"| {label} | {Path(case['input']).name if case.get('input') else 'synthetic'} | {case['api']} | {row['comparison']} | {row['status']} | {row['off_ns']/1e6:.6g} | {row['on_ns']/1e6:.6g} | {row['ratio']:.4g} [{row['ratio_min']:.4g},{row['ratio_max']:.4g}] | {row['before_new_allocations']}/{row['after_new_allocations'] if row['after_new_allocations'] is not None else 'unavailable'} |")
    for row in rows:
        case = row['case']
        if case['op'] == 'replay':
            lines += ['', f"Replay {case['input']} / {row['comparison']}: support agreement={row['support_agreement']}; max relative weight L2={row['weight_relative_l2']:.6g}, Linf={row['weight_relative_linf']:.6g}.",
                      'Per-pair KKT, normalization, captured-weight error, iterations/restarts and profile clocks are in summary.json; final weights and every raw timing are in raw.jsonl.']
            if row.get('fista_ratio') is not None:
                lines += [f"FISTA-only before/after ms: {row['fista_off_ns']/1e6:.6g}/{row['fista_on_ns']/1e6:.6g}; paired ratio {row['fista_ratio']:.4g} [{row['fista_ratio_min']:.4g},{row['fista_ratio_max']:.4g}]."]
            else:
                lines += ['FISTA-only ratio unavailable below clock resolution.']
            lines += ['']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def self_test():
    """check bounded schedules, zero-resolution calibration and visible replay mismatches without running a driver"""
    from types import SimpleNamespace
    import tempfile
    with tempfile.TemporaryDirectory(prefix='fdapde-preallocated-selfcheck-') as work:
        prefix = str(Path(work) / 'capsule')
        Path(prefix + '-omega.mtx').write_text('%%MatrixMarket matrix coordinate  real general\n% captured input\n3 3 1\n1 1 2\n')
        # token-based parsing accepts the repeated whitespace present in actual captured MatrixMarket headers
        assert capsule_shape(prefix) == (3, 3)
    args = SimpleNamespace(quick=False, ops=['spmv'], max_size=81, seed=1,
                           max_repetitions=64, round_ms=1, max_round_ms=50)
    selected = cases(args, {})
    # both dimensions obey the explicit bound, while sparse density and API controls remain present
    assert selected and all(max(row['rows'], row['cols']) <= 81 for row in selected)
    args.max_size = None
    # the unbounded schedule retains the largest sparse control for a full Kami campaign
    assert any(row['rows'] == 524289 for row in cases(args, {}))
    args.ops = ['gemv', 'dense_construct']
    # application-shaped dense controls retain synthetic coefficients and both constructor sizes
    assert any((row['rows'], row['cols']) == (3200, 1418) for row in cases(args, {}))
    args.max_size = 81
    # a constructor-only bounded local run still has a small eligible shape
    assert any(row['op'] == 'dense_construct' for row in cases(args, {}))
    case = dict(op='gemv', rows=3, cols=2, api='preallocated', offset=1, seed=1)
    row = dict(case, backend='native', input='', verified=True, error=None, oracle='scalar-coefficients',
               input_hash='abc', assignment=0, product=0, eigen_version='3.4.0', internal_threads=1,
               repetitions=10, rounds=3, median_ns=11, timings_ns=[10, 11, 12], max_abs_error=0,
               checksum=1, working_set_bytes=128, nnz=0, new_allocations_per_call=0)
    # the complete echoed driver contract and independent oracle certificate are accepted
    assert not validate(row, case, 'off', 'native', 10, 3)
    # a reported median cannot hide inconsistent raw round observations
    assert validate(dict(row, median_ns=12), case, 'off', 'native', 10, 3)
    # NaN cannot satisfy the process timing contract
    assert validate(dict(row, median_ns=float('nan')), case, 'off', 'native', 10, 3)
    seen = []
    def observe(mode, backend, repetitions, rounds, kind, allow_zero):
        """supply a synthetic clock that first resolves at four repetitions"""
        seen.append(repetitions)
        return {'median_ns': 5 if repetitions >= 4 else 0}
    calibrate(observe, [('off', 'native')], args)
    # only unresolved probes double, leaving large already-positive workloads untouched
    assert seen == [1, 2, 4]
    replay = dict(support_count=1, support_hash='a', iterations=4, restarts=1, final_weight=[1, 0])
    before = dict(rows=2, cols=2, input_hash='a', checksum=0, replay=replay)
    after = dict(before, replay=dict(replay, support_hash='b', final_weight=[0, 1]))
    checked = agreement(before, after, 1e-8)
    # differing supports and weights remain explicit mismatches rather than verified equivalence
    assert checked['errors'] and not checked['support_agreement'] and checked['weight_relative_l2'] > 1
    print('preallocated runner self-test passed')


def main():
    """build two recorded binaries and preserve raw, partial and complete paired comparisons"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--eigen-include')
    parser.add_argument('--output')
    parser.add_argument('--input', action='append', default=[], help='copied capsule prefix without -omega.mtx')
    parser.add_argument('--ops', help='comma-separated operation subset; default is every supported operation')
    parser.add_argument('--only-op', action='append', choices=OPS, help=argparse.SUPPRESS)
    parser.add_argument('--max-size', type=int, help='bound both dimensions, including captured capsules')
    parser.add_argument('--quick', action='store_true', help='reduced schedule; full schedule retains large controls')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--round-ms', type=float, default=10)
    parser.add_argument('--max-round-ms', type=float, default=50)
    parser.add_argument('--max-repetitions', type=int, default=1000000)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--replay-weight-tolerance', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--cpu', type=int, help='allowed Linux CPU for taskset; final runs require an exclusive job')
    parser.add_argument('--background', action='store_true', help='Darwin background policy and nice19 for provisional local runs')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not args.output or not args.eigen_include:
        parser.error('--output and --eigen-include are required')
    if args.ops and args.only_op:
        parser.error('use --ops or --only-op, not both')
    args.ops = args.ops.split(',') if args.ops else args.only_op or list(OPS)
    if not args.ops or len(set(args.ops)) != len(args.ops) or any(op not in OPS for op in args.ops):
        parser.error('operations must be known and unique')
    if args.pairs < 1 or not 1 <= args.rounds <= 101 or not 1 <= args.max_repetitions <= 10000000 or not 1 <= args.seed <= 2147483647 or args.max_size is not None and args.max_size < 1:
        parser.error('invalid pair, round, repetition, seed or size count')
    if any(not math.isfinite(v) or v <= 0 for v in (args.round_ms, args.max_round_ms, args.timeout, args.replay_weight_tolerance)):
        parser.error('timing parameters and replay weight tolerance must be finite and positive')
    if args.background and args.cpu is not None:
        parser.error('--background and --cpu describe mutually exclusive policies')
    repo = Path(__file__).resolve().parents[2]
    compiler = shutil.which(args.compiler)
    eigen = Path(args.eigen_include).expanduser().resolve()
    if compiler is None or not (eigen / 'Eigen/Core').is_file():
        parser.error('compiler executable or Eigen/Core was not found')
    compiler = os.path.abspath(compiler)
    prefix = []
    if args.background:
        tools = [shutil.which(name) for name in ('taskpolicy', 'nice')]
        if platform.system() != 'Darwin' or not all(tools):
            parser.error('--background requires Darwin taskpolicy and nice')
        prefix = [tools[0], '-b', tools[1], '-n', '19']
    elif args.cpu is not None:
        taskset = shutil.which('taskset')
        if not taskset or not hasattr(os, 'sched_getaffinity') or args.cpu not in os.sched_getaffinity(0):
            parser.error('taskset or the selected allowed CPU is unavailable')
        prefix = [taskset, '-c', str(args.cpu)]
    args.input = [str(Path(value).expanduser().resolve()) for value in args.input]
    if len(set(args.input)) != len(args.input):
        parser.error('capsule prefixes must be unique')
    inputs = {str(Path(value + ending)): digest(value + ending) for value in args.input for ending in CAPSULE_FILES}
    shapes = {value: capsule_shape(value) for value in args.input}
    selected = cases(args, shapes)
    if not selected:
        parser.error('the requested operation and size bounds select no cases')
    sources = [*sorted((repo / 'fdaPDE').rglob('*.h')), Path(__file__), repo / 'tests/benchmarks/run_simd_sweep.py',
               repo / 'tests/benchmarks/preallocated.cpp', repo / 'tests/benchmarks/preallocated_replay.h']
    source_hashes = {str(path.relative_to(repo)): digest(path) for path in sources}
    eigen_hashes = {str(path.relative_to(eigen)): digest(path) for path in sorted((eigen / 'Eigen').rglob('*')) if path.is_file()}
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    env = os.environ | {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
    env['LC_ALL'] = 'C'
    manifest = dict(git_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                    git_branch=subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip(),
                    tracked_changes=subprocess.check_output(['git', '-C', str(repo), 'diff', '--stat'], text=True),
                    compiler=compiler, compiler_version=subprocess.check_output([compiler, '--version'], text=True),
                    platform=platform.platform(), allowed_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    load_average=os.getloadavg() if hasattr(os, 'getloadavg') else None,
                    arguments=vars(args), sources=source_hashes, eigen_headers=eigen_hashes, inputs=inputs, cases=selected,
                    thread_environment={key: env[key] for key in env if key.endswith('NUM_THREADS') or key == 'LC_ALL'},
                    execution_prefix=prefix, policy='provisional Darwin background/nice19' if args.background else 'exclusive single-core required for final conclusions',
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='running', stage='build',
                    completed_cases=0, expected_comparisons=sum(len(comparisons(case)) for case in selected), builds={})
    rows = []; binaries = {}
    save(output, rows, manifest)
    try:
        for mode, flag in (('off', 0), ('on', 1)):
            binary = output / ('compare-' + mode)
            compile_cmd = [compiler, '-std=c++20', '-O3', '-DNDEBUG', '-DFDAPDE_NO_DEBUG', '-DEIGEN_DONT_PARALLELIZE', '-ffp-contract=off',
                           '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-isystem', str(repo), '-isystem', str(eigen),
                           f'-DFDAPDE_ENABLE_SIMD={flag}', str(repo / 'tests/benchmarks/preallocated.cpp'), '-o', str(binary)]
            manifest['builds'][mode] = dict(command=[*prefix, *compile_cmd])
            save(output, rows, manifest)
            with (output / (mode + '-build.log')).open('w') as log:
                subprocess.run([*prefix, *compile_cmd], stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
            binaries[mode] = binary
            manifest['builds'][mode]['binary_sha256'] = digest(binary)
            if any(digest(repo / path) != value for path, value in source_hashes.items()):
                raise RuntimeError('source bytes changed during compilation')
            save(output, rows, manifest)
        manifest['stage'] = 'timing'
        with (output / 'raw.jsonl').open('w') as raw:
            for index, case in enumerate(selected):
                manifest['current_case'] = case
                case_comparisons = comparisons(case)
                targets = [('off', 'native'), ('on', 'native'), *[('off', name) for name in (('eigen-col', 'eigen-row') if case['op'] in ('spmv', 'assemble', 'replay') else ('eigen-col',))]]
                def observe(mode, backend, repetitions, rounds, kind, allow_zero=False, **details):
                    """write every process observation before interpreting its independent certificate"""
                    observation = invoke(command(binaries[mode], case, backend, repetitions, rounds, prefix), env, args.timeout)
                    errors = validate(observation['result'], case, mode, backend, repetitions, rounds, allow_zero)
                    if observation['exit_code'] != 0:
                        errors.append('process did not exit successfully')
                    raw.write(json.dumps(dict(case=case, kind=kind, mode=mode, backend=backend, validation_errors=errors, **details, **observation)) + '\n'); raw.flush()
                    if errors:
                        raise RuntimeError(f"{case['op']} {mode}/{backend}: " + '; '.join(errors))
                    return observation['result']
                repetitions = calibrate(observe, targets, args, replay=case['op'] == 'replay')
                for before_mode, before_backend, after_mode, after_backend in case_comparisons:
                    comparison = f'{before_mode}/{before_backend}:{after_mode}/{after_backend}'
                    pairs = []; checks = []
                    for pair in range(args.pairs):
                        order = [('off', before_mode, before_backend), ('on', after_mode, after_backend)]
                        if (index + pair) % 2:
                            order.reverse()
                        values = {label: observe(mode, backend, repetitions, args.rounds, 'timing',
                                                 comparison=comparison, pair=pair, label=label)
                                  for label, mode, backend in order}
                        pairs.append(dict(values, order=[item[0] for item in order]))
                        checks.append(agreement(values['off'], values['on'], args.replay_weight_tolerance))
                    summary = paired_summary(pairs); summary.pop('pairs')
                    summary.update(case=case, comparison=comparison, repetitions=repetitions, rounds=args.rounds,
                                   status='MISMATCH' if any(check['errors'] for check in checks) else 'verified',
                                   before_new_allocations=pairs[0]['off']['new_allocations_per_call'],
                                   after_new_allocations=pairs[0]['on']['new_allocations_per_call'],
                                   working_set_bytes=pairs[0]['off']['working_set_bytes'],
                                   working_set_scope='logical FISTA kernel' if case['op'] == 'replay' else 'logical kernel',
                                   nnz=pairs[0]['off']['nnz'],
                                   pairs=[dict(order=pair['order'], agreement=check,
                                               off={key: value for key, value in pair['off'].items() if key not in ('timings_ns', 'replay')},
                                               on={key: value for key, value in pair['on'].items() if key not in ('timings_ns', 'replay')},
                                               replay={label: {key: value for key, value in pair[label].get('replay', {}).items() if key not in ('final_weight', 'fista_timings_ns')} for label in ('off', 'on')})
                                          for pair, check in zip(pairs, checks)])
                    if case['op'] == 'replay':
                        summary.update(support_agreement=all(check['support_agreement'] for check in checks),
                                       weight_relative_l2=max(check['weight_relative_l2'] for check in checks),
                                       weight_relative_linf=max(check['weight_relative_linf'] for check in checks))
                        fista_pairs = [{label: {'median_ns': statistics.median(pair[label]['replay']['fista_timings_ns'])} for label in ('off', 'on')} for pair in pairs]
                        if all(pair[label]['median_ns'] > 0 for pair in fista_pairs for label in ('off', 'on')):
                            summary.update({'fista_' + key: value for key, value in paired_summary(fista_pairs).items() if key != 'pairs'})
                    rows.append(summary)
                    save(output, rows, manifest)
                    print(case['op'], case['rows'], case['api'], comparison, summary['status'], f"ratio={summary['ratio']:.4g}", flush=True)
                manifest['completed_cases'] += 1
        final_hash_match = (all(digest(repo / path) == value for path, value in source_hashes.items()) and
                            all(digest(path) == value for path, value in inputs.items()) and
                            all(digest(binaries[mode]) == build['binary_sha256'] for mode, build in manifest['builds'].items()) and
                            all(digest(eigen / path) == value for path, value in eigen_hashes.items()))
        manifest['final_hash_match'] = final_hash_match
        if not final_hash_match:
            raise RuntimeError('source, Eigen, input or binary bytes changed during the campaign')
        manifest.update(status='complete', stage='complete', comparison_agreement='mismatch' if any(row['status'] == 'MISMATCH' for row in rows) else 'verified')
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='partial' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error) or type(error).__name__)
    finally:
        manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(output, rows, manifest)
    print('saved summary to', output / 'summary.md', flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
