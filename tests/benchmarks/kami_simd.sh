#!/usr/bin/env bash
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

set -eo pipefail
set +u

# source the site profiles with no launcher arguments and before enabling nounset
load_kami_environment() {
    local fdapde_env_file
    export SIMD_KAMI_ENV_DIR="${SIMD_KAMI_ENV_DIR:-$HOME}"
    for fdapde_env_file in "$SIMD_KAMI_ENV_DIR/kami-vars.sh" "$SIMD_KAMI_ENV_DIR/kami-load.sh"; do
        if [[ -r "$fdapde_env_file" ]]; then
            echo "loading Kami environment: $fdapde_env_file"
            source "$fdapde_env_file"
        fi
    done
}
load_kami_environment
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
action=${1:-submit}
if (( $# > 0 )); then shift; fi
if (( $# > 1 )) || [[ ! "$action" =~ ^(prepare|submit|run)$ ]]; then
    echo "usage: bash tests/benchmarks/kami_simd.sh prepare | submit [OUTPUT_DIR] | run OUTPUT_DIR" >&2
    exit 2
fi
# identify unavailable programs before reserving a node and in the compute-node log
require_tool() {
    if ! command -v "$1" >/dev/null; then
        export FDAPDE_SIMD_PREFLIGHT_ERROR="required program missing from PATH: $1"
        if [[ "$1" == */* ]]; then
            export FDAPDE_SIMD_PREFLIGHT_ERROR="required executable missing or not executable: $1"
        fi
        echo "$FDAPDE_SIMD_PREFLIGHT_ERROR" >&2
        exit 127
    fi
}
require_tool python3
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ is required"'
revision=$(python3 - "$root/tests/CMakeLists.txt" <<'PY'
import re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
print(re.search(r'FetchContent_Declare\(googletest\s+URL\s+[^\s)]+/archive/([0-9a-f]{40})\.zip',text).group(1))
PY
)
gtest_source=${FDAPDE_GTEST_SOURCE:-"$root/output/simd/kami/deps/googletest-$revision"}
gtest_source=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$gtest_source")

if [[ "$action" == prepare ]]; then
    if (( $# != 0 )); then echo "prepare takes no output directory" >&2; exit 2; fi
    python3 - "$gtest_source" "$revision" <<'PY'
import shutil,sys,tempfile,urllib.request,zipfile
from pathlib import Path
source=Path(sys.argv[1]).expanduser().resolve()
if not (source/'CMakeLists.txt').is_file():
    if source.exists(): raise SystemExit('GoogleTest destination exists without CMakeLists.txt; choose FDAPDE_GTEST_SOURCE')
    source.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gtest-',dir=source.parent) as work:
        work=Path(work)
        url='https://github.com/google/googletest/archive/'+sys.argv[2]+'.zip'
        with urllib.request.urlopen(url,timeout=180) as response, (work/'source.zip').open('wb') as archive:
            shutil.copyfileobj(response,archive)
        with zipfile.ZipFile(work/'source.zip') as archive:
            for name in archive.namelist():
                if not (work/name).resolve().is_relative_to(work): raise SystemExit('invalid archive member')
            archive.extractall(work)
        unpacked=work/('googletest-'+sys.argv[2])
        if not (unpacked/'CMakeLists.txt').is_file(): raise SystemExit('unexpected GoogleTest archive')
        unpacked.rename(source)
print('GoogleTest source ready: '+str(source))
PY
    exit
fi

export CXX=${CXX:-g++}
require_tool "$CXX"
CXX=$(command -v "$CXX")
CXX=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CXX")
export CXX FDAPDE_GTEST_SOURCE="$gtest_source" FDAPDE_GTEST_REVISION="$revision"
export SIMD_CMAKE=${SIMD_CMAKE:-cmake} SIMD_CTEST=${SIMD_CTEST:-ctest}
export SIMD_CPUS=${SIMD_CPUS:-4} SIMD_QUEUE=${SIMD_QUEUE:-test}
export SIMD_MEM=${SIMD_MEM:-32gb} SIMD_WALLTIME=${SIMD_WALLTIME:-72:00:00}
export SIMD_PAIRS=${SIMD_PAIRS:-3} SIMD_ROUNDS=${SIMD_ROUNDS:-5} SIMD_LARGE_ROUNDS=${SIMD_LARGE_ROUNDS:-1}
export SIMD_PREALLOCATED_PAIRS=${SIMD_PREALLOCATED_PAIRS:-5} SIMD_PREALLOCATED_ROUNDS=${SIMD_PREALLOCATED_ROUNDS:-5}
export SIMD_PREALLOCATED_ROUND_MS=${SIMD_PREALLOCATED_ROUND_MS:-10}
export SIMD_MAX_CALL_SECONDS=${SIMD_MAX_CALL_SECONDS:-60} SIMD_TIMEOUT=${SIMD_TIMEOUT:-900}
export SIMD_ASSIGNMENT_SIZES=${SIMD_ASSIGNMENT_SIZES:-9,27,99,387,1539,6147,24579,98307,393219,1572867,6291459,12582915,25165827,50331651,100663299,201326595}
export SIMD_PRODUCT_SIZES=${SIMD_PRODUCT_SIZES:-3,8,16,32,64,128,256,384,512,768,1024,1280,1537,1793,2049,2305,2561,3073,3585,4097,5121,6145,7169,8193}
[[ "$SIMD_CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "SIMD_CPUS must be a positive integer" >&2; exit 2; }

# verify the benchmark dependency without exposing Eigen to the native-only CMake lane
preflight_eigen() {
    local requested=${SIMD_EIGEN_INCLUDE:-${PATH_EIGEN_INCLUDE:-}}
    if ! SIMD_EIGEN_INCLUDE=$(python3 - "$requested" <<'PY_EIGEN'
import math,os,re,sys
from pathlib import Path
if not sys.argv[1]: raise SystemExit('set SIMD_EIGEN_INCLUDE or PATH_EIGEN_INCLUDE to the Eigen 3.4.0 include directory')
path=Path(sys.argv[1]).expanduser().resolve()
macros=path/'Eigen/src/Core/util/Macros.h'
if not (path/'Eigen/Core').is_file() or not macros.is_file(): raise SystemExit('Eigen/Core or Eigen/src/Core/util/Macros.h is missing from '+str(path))
text=macros.read_text()
version=[re.search(r'^\s*#\s*define\s+EIGEN_'+name+r'_VERSION\s+(\d+)\b',text,re.M) for name in ('WORLD','MAJOR','MINOR')]
if any(v is None for v in version) or tuple(int(v[1]) for v in version)!=(3,4,0): raise SystemExit('the comparison requires Eigen 3.4.0: '+str(path))
for key in ('SIMD_PREALLOCATED_PAIRS','SIMD_PREALLOCATED_ROUNDS'):
    value=int(os.environ[key])
    if value<=0 or (key.endswith('ROUNDS') and value>101): raise SystemExit(key+' has an invalid count')
value=float(os.environ['SIMD_PREALLOCATED_ROUND_MS'])
if not math.isfinite(value) or value<=0: raise SystemExit('SIMD_PREALLOCATED_ROUND_MS must be finite and positive')
print(path)
PY_EIGEN
    ); then
        export FDAPDE_SIMD_PREFLIGHT_ERROR="Eigen 3.4.0 or comparison-parameter preflight failed: $requested"
        return 2
    fi
    export SIMD_EIGEN_INCLUDE
}

# resolve portable offline copies and check all capsule bytes before reserving or using a node
preflight_replay_inputs() {
    if ! preallocated_inputs_json=$(python3 - "$root" <<'PY_INPUTS'
import hashlib,json,os,re,sys
from pathlib import Path
requested=os.environ.get('SIMD_REPLAY_MANIFEST')
path=Path(requested or Path(sys.argv[1])/'tests/benchmarks/fixtures/rgcca_replay/manifest.json').expanduser().resolve()
if not path.exists():
    if requested: raise SystemExit('requested replay manifest is missing: '+str(path))
    result={'status':'absent','manifest':str(path),'manifest_sha256':None,'cases':[]}
else:
    document=json.loads(path.read_text()); base=path.parent
    cases=document.get('cases')
    if not isinstance(cases,list) or not cases: raise SystemExit('replay manifest must contain nonempty cases: '+str(path))
    def relative(value):
        if not isinstance(value,str) or not value or '\n' in value or '\r' in value: raise SystemExit('invalid capsule relative path')
        candidate=Path(value); resolved=(base/candidate).resolve()
        if candidate.is_absolute() or not resolved.is_relative_to(base): raise SystemExit('capsule path must remain relative to the manifest directory: '+value)
        return resolved
    selected=[]; seen=set()
    for case in cases:
        prefix=relative(case['prefix_relative'])
        if prefix in seen: raise SystemExit('duplicate capsule prefix: '+str(prefix))
        seen.add(prefix)
        files=case.get('files')
        if not isinstance(files,list): raise SystemExit('capsule file inventory is missing: '+str(prefix))
        inventory={}
        for entry in files:
            copied=relative(entry['copy_relative'])
            if copied in inventory: raise SystemExit('duplicate capsule file: '+str(copied))
            inventory[copied]=entry
        expected={Path(str(prefix)+ending) for ending in ('-omega.mtx','-c.bin','-warm.bin','-weight.bin','.ready')}
        if set(inventory)!=expected: raise SystemExit('capsule requires exactly the five canonical files: '+str(prefix))
        for copied,entry in inventory.items():
            expected_hash=entry.get('sha256_copy')
            if not isinstance(expected_hash,str) or not re.fullmatch('[0-9a-fA-F]{64}',expected_hash): raise SystemExit('capsule SHA256 is missing: '+str(copied))
            if not copied.is_file() or hashlib.sha256(copied.read_bytes()).hexdigest()!=expected_hash.lower(): raise SystemExit('capsule is missing or its SHA256 changed: '+str(copied))
            if 'bytes' in entry and (type(entry['bytes']) is not int or copied.stat().st_size!=entry['bytes']): raise SystemExit('capsule byte count changed: '+str(copied))
        shape=case.get('matrix',{})
        selected.append({'name':case.get('name',prefix.name),'prefix':str(prefix),'component':case.get('component'),'observed_lambda':case.get('observed_lambda'),'rows':shape.get('rows'),'cols':shape.get('cols'),'nnz':shape.get('nnz')})
    result={'status':'present','manifest':str(path),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':selected}
expected=os.environ.get('FDAPDE_SIMD_EXPECTED_REPLAY_MANIFEST_SHA')
if expected and expected!=(result['manifest_sha256'] or 'absent'): raise SystemExit('replay manifest changed since submission')
print('offline replay inputs: '+result['status']+', capsules='+str(len(result['cases'])),file=sys.stderr)
print(json.dumps(result))
PY_INPUTS
    ); then
        export FDAPDE_SIMD_PREFLIGHT_ERROR="offline replay capsule preflight failed"
        return 2
    fi
    if [[ -n "${SIMD_REPLAY_MANIFEST:-}" ]]; then
        SIMD_REPLAY_MANIFEST=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "$preallocated_inputs_json")
        export SIMD_REPLAY_MANIFEST
    fi
}

if [[ "$action" == submit ]]; then
    require_tool qsub
    for tool in "$SIMD_CMAKE" "$SIMD_CTEST" taskset git; do require_tool "$tool"; done
    # keep the selected build tools when the compute-node profiles change PATH
    SIMD_CMAKE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$(command -v "$SIMD_CMAKE")")
    SIMD_CTEST=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$(command -v "$SIMD_CTEST")")
    [[ -f "$gtest_source/CMakeLists.txt" ]] || { echo "run prepare first, or set FDAPDE_GTEST_SOURCE" >&2; exit 2; }
    preflight_eigen
    preflight_replay_inputs
    run_dir=${1:-"$root/output/simd/kami/$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$root" rev-parse --short HEAD)"}
    run_dir=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$run_dir")
    mkdir -p -- "$(dirname -- "$run_dir")"
    mkdir -- "$run_dir"
    python3 - "$root" "$run_dir" <<'PY'
import shlex,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:])
script='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(root))+'\nexec bash '+shlex.quote(str(root/'tests/benchmarks/kami_simd.sh'))+' run '+shlex.quote(str(run))+'\n'
(run/'job.pbs').write_text(script)
PY
    export FDAPDE_SIMD_EXPECTED_COMMIT=$(git -C "$root" rev-parse HEAD)
    export FDAPDE_SIMD_EXPECTED_REPLAY_MANIFEST_SHA=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"] or "absent")' "$preallocated_inputs_json")
    qsub -V -N fdapde-simd -q "$SIMD_QUEUE" -l "select=1:ncpus=$SIMD_CPUS:mem=$SIMD_MEM" \
        -l place=excl -l "walltime=$SIMD_WALLTIME" -j oe -o "$run_dir/pbs.log" "$run_dir/job.pbs" | tee "$run_dir/job-id.txt"
    echo "results: $run_dir"
    exit
fi

[[ -n "${PBS_JOBID:-}" ]] || { echo "run must execute inside the submitted PBS job, not on the login node" >&2; exit 2; }
(( $# == 1 )) || { echo "run requires OUTPUT_DIR" >&2; exit 2; }
run_dir=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
[[ -d "$run_dir" && ! -e "$run_dir/job-metadata.json" ]] || { echo "run directory missing or already used" >&2; exit 2; }
exec > >(tee -a "$run_dir/job.log") 2>&1
stage=preflight

# update the live job state so logs and partial reports identify the current operation
set_stage() {
    stage=$1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $stage"
    if [[ -f "$run_dir/job-metadata.json" ]]; then
        python3 - "$run_dir" "$stage" <<'PY_STAGE'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])/'job-metadata.json'; metadata=json.loads(path.read_text()); metadata['stage']=sys.argv[2]
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY_STAGE
    fi
}

# retain CTest's exit status and reported counts even when a test fails
record_tests() {
    python3 - "$run_dir" "$1" "$stage" "$2" <<'PY_TESTS'
import json,re,sys
from pathlib import Path
run=Path(sys.argv[1]); match=re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)',(run/(sys.argv[3]+'.log')).read_text())
path=run/'job-metadata.json'; metadata=json.loads(path.read_text())
metadata['tests'].append({'mode':sys.argv[2],'status':'passed' if sys.argv[4]=='0' else 'failed','passed':int(match[3])-int(match[2]) if match else None,'failed':int(match[2]) if match else None,'exit_code':int(sys.argv[4]),'log':sys.argv[3]+'.log'})
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY_TESTS
}

# preserve failed or partial runs and generate their report before leaving the PBS job
finish() {
    result=$?
    trap - EXIT
    set +e
    python3 - "$run_dir" "$stage" "$result" <<'PY'
import datetime,json,os,platform,sys
from pathlib import Path
run=Path(sys.argv[1]); path=run/'job-metadata.json'
metadata=json.loads(path.read_text()) if path.exists() else {}
metadata.setdefault('job_id',os.environ.get('PBS_JOBID'))
metadata.setdefault('hostname',platform.node())
metadata.setdefault('compiler',os.environ.get('CXX'))
metadata.setdefault('cmake',os.environ.get('SIMD_CMAKE'))
metadata.setdefault('ctest',os.environ.get('SIMD_CTEST'))
if os.environ.get('FDAPDE_SIMD_PREFLIGHT_ERROR'):
    metadata['error']=os.environ['FDAPDE_SIMD_PREFLIGHT_ERROR']
metadata.update(stage=sys.argv[2],exit_code=int(sys.argv[3]),status='complete' if sys.argv[3]=='0' else 'failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY
    metadata_result=$?
    if (( result == 0 && metadata_result != 0 )); then result=$metadata_result; fi
    python3 "$root/tests/benchmarks/summarize_simd_sweep.py" "$run_dir/assignment" "$run_dir/product" \
        --preallocated "$run_dir/preallocated" --output "$run_dir/summary.md" --job-metadata "$run_dir/job-metadata.json"
    report_result=$?
    if (( report_result != 0 )); then
        if (( result == 0 )); then result=$report_result; fi
        python3 - "$run_dir" "$result" "$report_result" <<'PY_REPORT'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])/'job-metadata.json'; metadata=json.loads(path.read_text())
metadata.update(status='failed',failed_stage=metadata.get('stage'),stage='summary',exit_code=int(sys.argv[2]),summary_exit_code=int(sys.argv[3]))
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY_REPORT
    fi
    echo "finished: exit=$result stage=$stage results=$run_dir"
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for tool in "$SIMD_CMAKE" "$SIMD_CTEST" taskset git; do require_tool "$tool"; done
[[ -f "$gtest_source/CMakeLists.txt" ]] || { echo "cached GoogleTest source is missing" >&2; exit 2; }
preflight_eigen
preflight_replay_inputs
cd -- "$root"
[[ -z "${FDAPDE_SIMD_EXPECTED_COMMIT:-}" || "$(git rev-parse HEAD)" == "$FDAPDE_SIMD_EXPECTED_COMMIT" ]] || {
    echo "checkout changed since submission; fetch the intended commit and submit a new run" >&2; exit 2;
}
python3 - "$run_dir" "$preallocated_inputs_json" <<'PY'
import datetime,json,math,os,platform,subprocess,sys
from pathlib import Path
for key in ('SIMD_PAIRS','SIMD_ROUNDS','SIMD_LARGE_ROUNDS'):
    value=int(os.environ[key])
    if value<=0 or (key!='SIMD_PAIRS' and value>101): raise SystemExit(key+' has an invalid count')
for key in ('SIMD_MAX_CALL_SECONDS','SIMD_TIMEOUT'):
    value=float(os.environ[key])
    if not math.isfinite(value) or value<=0: raise SystemExit(key+' must be finite and positive')
for key in ('SIMD_ASSIGNMENT_SIZES','SIMD_PRODUCT_SIZES'):
    sizes=[int(v) for v in os.environ[key].split(',')]
    if not sizes or sizes!=sorted(set(sizes)) or sizes[0]<=0: raise SystemExit(key+' must be positive, unique and increasing')
    if key=='SIMD_ASSIGNMENT_SIZES' and any(v%3 for v in sizes): raise SystemExit(key+' requires multiples of three')
run=Path(sys.argv[1]); allowed=sorted(os.sched_getaffinity(0)); cpu=allowed[0]
cache=[]
for entry in sorted(Path('/sys/devices/system/cpu/cpu'+str(cpu)+'/cache').glob('index*')):
    cache.append({name:(entry/name).read_text().strip() for name in ('level','type','size','shared_cpu_list')})
def cache_bytes(value):
    units={'K':1024,'M':1024**2,'G':1024**3}
    return int(value[:-1])*units[value[-1]] if value[-1] in units else int(value)
threshold=int(os.environ.get('SIMD_CACHE_BYTES',max([cache_bytes(c['size']) for c in cache] or [16*1024**2])))
if threshold<=0: raise SystemExit('SIMD_CACHE_BYTES must be positive')
def output(args): return subprocess.check_output(args,text=True).strip()
hardware={'platform':platform.platform(),'hostname':platform.node(),'allowed_cpus':allowed,'affinity_cpu':cpu,'cache':cache,'load_average':os.getloadavg()}
import shutil
if shutil.which('lscpu'): hardware['lscpu']=output(['lscpu'])
(run/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
inputs=json.loads(sys.argv[2])
metadata={'git_commit':output(['git','rev-parse','HEAD']),'git_branch':output(['git','branch','--show-current']),'git_dirty':bool(output(['git','status','--porcelain','--untracked-files=no'])),'job_id':os.environ['PBS_JOBID'],'hostname':platform.node(),'compiler':os.environ['CXX'],'compiler_version':output([os.environ['CXX'],'--version']),'cmake':os.environ['SIMD_CMAKE'],'cmake_version':output([os.environ['SIMD_CMAKE'],'--version']),'ctest':os.environ['SIMD_CTEST'],'ctest_version':output([os.environ['SIMD_CTEST'],'--version']),'googletest_source':os.environ['FDAPDE_GTEST_SOURCE'],'googletest_declared_revision':os.environ['FDAPDE_GTEST_REVISION'],'eigen_include':os.environ['SIMD_EIGEN_INCLUDE'],'eigen_version':'3.4.0','preallocated_inputs':inputs,'preallocated_comparison':{'directory':'preallocated','status':'pending','exit_code':None},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','stage':'preflight','affinity_cpu':cpu,'cache_bytes':threshold,'parameters':{k:v for k,v in os.environ.items() if k.startswith('SIMD_')},'tests':[],'plots':'unavailable'}
(run/'job-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(run/'affinity-cpu.txt').write_text(str(cpu)+'\n'); (run/'cache-bytes.txt').write_text(str(threshold)+'\n')
(run/'preallocated-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
(run/'preallocated-input-prefixes.txt').write_text(''.join(case['prefix']+'\n' for case in inputs['cases']))
PY
cpu=$(cat "$run_dir/affinity-cpu.txt")
cache_bytes=$(cat "$run_dir/cache-bytes.txt")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export LC_ALL=C
python3 "$root/tests/benchmarks/run_simd_sweep.py" --self-test
python3 "$root/tests/benchmarks/summarize_simd_sweep.py" --self-test
python3 "$root/tests/benchmarks/run_preallocated_comparison.py" --self-test

# assertions stay enabled in the native suite for every independent flag combination
for mode in off assignment product all; do
    assignment=0; product=0
    [[ "$mode" == assignment || "$mode" == all ]] && assignment=1
    [[ "$mode" == product || "$mode" == all ]] && product=1
    set_stage "tests-$mode"
    build_dir="$run_dir/tests-$mode"
    "$SIMD_CMAKE" -S "$root/tests" -B "$build_dir" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release \
        -DFDAPDE_NATIVE_ONLY=ON -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$gtest_source" -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
        "-DCMAKE_CXX_FLAGS=-DFDAPDE_ENABLE_SIMD_ASSIGNMENT=$assignment -DFDAPDE_ENABLE_SIMD_PRODUCT=$product" > "$run_dir/$stage-configure.log" 2>&1
    "$SIMD_CMAKE" --build "$build_dir" --parallel "$SIMD_CPUS" > "$run_dir/$stage-build.log" 2>&1
    test_status=0
    "$SIMD_CTEST" --test-dir "$build_dir" --output-on-failure --no-tests=error -j "$SIMD_CPUS" > "$run_dir/$stage.log" 2>&1 || test_status=$?
    record_tests "$mode" "$test_status"
    (( test_status == 0 )) || exit "$test_status"
done

set_stage sanitizers
"$SIMD_CMAKE" -S "$root/tests" -B "$run_dir/tests-sanitizers" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Debug \
    -DFDAPDE_NATIVE_ONLY=ON -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$gtest_source" -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    '-DCMAKE_CXX_FLAGS=-DFDAPDE_ENABLE_SIMD=1 -fsanitize=address,undefined -fno-omit-frame-pointer' > "$run_dir/sanitizers-configure.log" 2>&1
"$SIMD_CMAKE" --build "$run_dir/tests-sanitizers" --target fdapde_dense_test fdapde_sparse_test --parallel "$SIMD_CPUS" > "$run_dir/sanitizers-build.log" 2>&1
test_status=0
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1 \
    "$SIMD_CTEST" --test-dir "$run_dir/tests-sanitizers" --output-on-failure --no-tests=error -R '^(Contiguous(Assignment|Product)|Preallocated(Product|Assignment|Replay)|SparseMultiplyInto)\.' \
    > "$run_dir/sanitizers.log" 2>&1 || test_status=$?
record_tests 'asan+ubsan' "$test_status"
(( test_status == 0 )) || exit "$test_status"

set_stage benchmark-build
for suite in assignment product; do
    python3 "$root/tests/benchmarks/run_simd_sweep.py" --compiler "$CXX" --label kami --suite "$suite" \
        --output "$run_dir/$suite" --build-only > "$run_dir/$suite-build.log" 2>&1
done

set_stage public-smoke
python3 - "$root" "$run_dir" <<'PY'
import json,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:]); sys.path.insert(0,str(root/'tests/benchmarks'))
import run_simd_sweep as sweep
rows=[]
for suite,cases in [('assignment',sweep.ASSIGNMENT_CASES+sweep.ASSIGNMENT_CONTROLS),('product',sweep.PRODUCT_CASES+sweep.STATIC_CASES)]:
    for case in cases:
        checksums=[]
        for mode,flags in sweep.MODES.items():
            size=9 if suite=='assignment' else int(case[3:]) if case in sweep.STATIC_CASES else 3
            result=sweep.run_json(sweep.command(run/suite/('sweep-'+mode),suite,case,size,100,3),60)
            assert (result['assignment'],result['product'])==flags, 'binary flags must match the selected mode'
            checksums.append(result['checksum']); rows.append({'mode':mode,'result':result})
        assert len(set(checksums))==1, 'all four flag combinations must produce the same public output'
(run/'public-smoke.json').write_text(json.dumps({'runs':len(rows),'verified':True,'results':rows},indent=2)+'\n')
path=run/'job-metadata.json'; metadata=json.loads(path.read_text())
metadata['public_smoke']={'cases':len(sweep.ASSIGNMENT_CASES+sweep.ASSIGNMENT_CONTROLS+sweep.PRODUCT_CASES+sweep.STATIC_CASES),'runs':len(rows),'verified':True,'file':'public-smoke.json'}
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY

# finish comparison builds before measuring either sweep and retain the runner's partial status
set_stage preallocated-comparison
python3 - "$run_dir" <<'PY_COMPARISON'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])/'job-metadata.json'; metadata=json.loads(path.read_text())
metadata['preallocated_comparison']['status']='running'
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY_COMPARISON
input_args=()
while IFS= read -r prefix; do input_args+=(--input "$prefix"); done < "$run_dir/preallocated-input-prefixes.txt"
comparison_status=0
python3 "$root/tests/benchmarks/run_preallocated_comparison.py" --compiler "$CXX" --eigen-include "$SIMD_EIGEN_INCLUDE" \
    --cpu "$cpu" --output "$run_dir/preallocated" --pairs "$SIMD_PREALLOCATED_PAIRS" --rounds "$SIMD_PREALLOCATED_ROUNDS" \
    --round-ms "$SIMD_PREALLOCATED_ROUND_MS" --timeout "$SIMD_TIMEOUT" ${input_args[@]+"${input_args[@]}"} \
    > "$run_dir/preallocated-comparison.log" 2>&1 || comparison_status=$?
comparison_status=$(python3 - "$run_dir" "$comparison_status" <<'PY_COMPARISON_RESULT'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); path=run/'job-metadata.json'; metadata=json.loads(path.read_text())
child=run/'preallocated/manifest.json'
try:
    comparison=json.loads(child.read_text()) if child.is_file() else {}
    if not isinstance(comparison,dict): raise ValueError('manifest is not an object')
except (OSError,ValueError) as error:
    comparison={'status':'failed','error':'comparison manifest could not be read: '+str(error)}
runner_exit=int(sys.argv[2]); effective_exit=runner_exit; error=comparison.get('error')
if runner_exit==0 and (comparison.get('status')!='complete' or comparison.get('comparison_agreement')!='verified' or comparison.get('final_hash_match') is not True):
    effective_exit=1
    error=error or 'comparison did not complete with verified agreement and unchanged hashes'
metadata['preallocated_comparison'].update(status='complete' if effective_exit==0 else 'failed',runner_exit_code=runner_exit,exit_code=effective_exit,comparison_agreement=comparison.get('comparison_agreement'),error=error)
path.write_text(json.dumps(metadata,indent=2)+'\n')
print(effective_exit)
PY_COMPARISON_RESULT
)
(( comparison_status == 0 )) || exit "$comparison_status"

# keep all timed processes serial and on the same allowed CPU after every build has finished
for suite in assignment product; do
    set_stage "timing-$suite"
    sizes=$SIMD_ASSIGNMENT_SIZES
    [[ "$suite" == product ]] && sizes=$SIMD_PRODUCT_SIZES
    extra=()
    [[ "$suite" == product ]] && extra=(--factorial)
    taskset -c "$cpu" python3 "$root/tests/benchmarks/run_simd_sweep.py" --compiler "$CXX" --label kami \
        --suite "$suite" --sizes "$sizes" --pairs "$SIMD_PAIRS" --rounds "$SIMD_ROUNDS" --large-rounds "$SIMD_LARGE_ROUNDS" \
        --max-call-seconds "$SIMD_MAX_CALL_SECONDS" --timeout "$SIMD_TIMEOUT" --cache-bytes "$cache_bytes" \
        --output "$run_dir/$suite" --reuse-binaries ${extra[@]+"${extra[@]}"} > "$run_dir/$suite-run.log" 2>&1
done

set_stage plots
if command -v Rscript >/dev/null; then
    plot_status=generated
    for suite in assignment product; do
        if ! Rscript "$root/tests/benchmarks/plot_simd_sweep.R" "$run_dir/$suite/summary.csv" "$run_dir/$suite/plots" \
                > "$run_dir/$suite-plots.log" 2>&1; then plot_status=failed; fi
    done
    python3 - "$run_dir" "$plot_status" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])/'job-metadata.json'; metadata=json.loads(path.read_text()); metadata['plots']=sys.argv[2]
path.write_text(json.dumps(metadata,indent=2)+'\n')
PY
fi
set_stage complete
