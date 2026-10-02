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

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    """check the actual launcher with cached sources and local scheduler stubs"""
    launcher = Path(__file__).resolve().with_name("kami_simd.sh")
    checks = []
    subprocess.run(["/bin/bash", "-n", str(launcher)], check=True)
    with tempfile.TemporaryDirectory(prefix="fdapde-kami-environment-") as directory:
        work = Path(directory)
        profiles, tools, utilities, cache = (work / name for name in ("profiles", "tools", "utilities", "cache"))
        for path in (profiles, tools, utilities, cache):
            path.mkdir()
        (cache / "CMakeLists.txt").write_text("# existing source prevents downloads\n")
        for name in ("dirname", "mkdir", "tee", "date", "cat"):
            (utilities / name).symlink_to(shutil.which(name))
        (tools / "python3").symlink_to(sys.executable)
        for name in ("cmake", "ctest", "taskset", "mock-cxx"):
            (tools / name).write_text("#!/bin/bash\necho 'unexpected tool execution' >&2\nexit 99\n")
        commit = "a" * 40
        (tools / "git").write_text(f"#!/bin/bash\nprintf '%s\\n' '{commit}'\n")
        (tools / "qsub").write_text("#!" + sys.executable + "\n" + """import json,os,sys
from pathlib import Path
Path(os.environ['MOCK_QSUB_RECORD']).write_text(json.dumps({
    'arguments':sys.argv[1:], 'compiler':os.environ['CXX'], 'path':os.environ['PATH'],
    'profiles':os.environ['SIMD_KAMI_ENV_DIR'], 'commit':os.environ['FDAPDE_SIMD_EXPECTED_COMMIT']}))
print('1234.mock')
""")
        for path in tools.iterdir():
            if not path.is_symlink():
                path.chmod(0o755)
        (profiles / "kami-vars.sh").write_text("""printf 'vars:%s:%s\\n' "$#" "$MOCK_UNDEFINED" >> "$MOCK_TRACE"
export MOCK_PROFILE_ORDER=vars CXX=mock-cxx FDAPDE_GTEST_SOURCE="$MOCK_CACHE"
""")
        (profiles / "kami-load.sh").write_text("""printf 'load:%s:%s\\n' "$#" "$MOCK_PROFILE_ORDER" >> "$MOCK_TRACE"
export PATH="$MOCK_TOOLS:$PATH"
""")
        trace, record = work / "trace", work / "qsub.json"
        env = os.environ | {"PATH": str(utilities), "SIMD_KAMI_ENV_DIR": str(profiles),
                            "MOCK_CACHE": str(cache), "MOCK_TOOLS": str(tools),
                            "MOCK_TRACE": str(trace), "MOCK_QSUB_RECORD": str(record)}
        for name in ("MOCK_UNDEFINED", "PBS_JOBID", "FDAPDE_SIMD_EXPECTED_COMMIT"):
            env.pop(name, None)

        def invoke(*arguments, overrides=None, nounset=False):
            """run one CLI operation without changing home or invoking real PBS"""
            trace.unlink(missing_ok=True)
            result = subprocess.run(["/bin/bash", *(["-u"] if nounset else []), str(launcher), *map(str, arguments)],
                                    env=env | (overrides or {}), text=True, capture_output=True, timeout=30)
            # the trace verifies source order, empty function arguments and an initially undefined variable
            assert trace.read_text().splitlines() == ["vars:0:", "load:0:vars"], result
            return result

        prepared = invoke("prepare")
        # cached preparation must succeed with Python becoming available only after load
        assert prepared.returncode == 0 and str(cache) in prepared.stdout, prepared
        checks.append("profiles_order_zero_arguments_undefined_variable_cached_prepare")
        inherited = invoke("prepare", nounset=True)
        # the loader must disable inherited nounset before profiles expand an undefined variable
        assert inherited.returncode == 0 and str(cache) in inherited.stdout, inherited
        checks.append("profiles_with_inherited_nounset")
        submitted = invoke("submit", work / "submitted")
        # the fake scheduler proves all tools and the compiler became available after loading
        assert submitted.returncode == 0 and record.exists(), submitted
        received = json.loads(record.read_text())
        # PBS must retain the configured profile path and resolved compiler driver
        assert received["profiles"] == str(profiles) and received["compiler"] == str(tools / "mock-cxx"), received
        # scheduler arguments must export the environment and retain the pinned checkout identity
        assert "-V" in received["arguments"] and received["commit"] == commit, received
        checks.append("submit_mock_scheduler_exported_environment")

        for name in ("cmake", "ctest", "taskset", "git"):
            tool = tools / name
            tool.rename(tools / (name + ".hidden"))
            record.unlink(missing_ok=True)
            rejected = invoke("submit", work / ("missing-" + name))
            # each unavailable tool must be named before scheduler invocation or output creation
            assert rejected.returncode == 127 and name in rejected.stderr and not record.exists(), rejected
            checks.append("submit_missing_" + name + "_before_qsub")
            (tools / (name + ".hidden")).rename(tool)

        (tools / "cmake").unlink()
        run_dir = work / "worker"
        run_dir.mkdir()
        failed = invoke("run", run_dir, overrides={"PBS_JOBID": "1234.mock"})
        # worker finalization must leave metadata before its contents are inspected
        assert (run_dir / "job-metadata.json").exists(), failed
        metadata = json.loads((run_dir / "job-metadata.json").read_text())
        message = "required program missing from PATH: cmake"
        # the worker must preserve the missing-tool failure and identify its preflight stage
        assert failed.returncode == 127 and metadata["status"] == "failed" and metadata["stage"] == "preflight", metadata
        # the diagnostic must survive JSON finalization and the generated partial summary
        assert metadata["exit_code"] == 127 and metadata["error"] == message and message in (run_dir / "summary.md").read_text(), metadata
        # a preflight failure must stop before native configuration or benchmark execution
        assert not list(run_dir.glob("tests-*-configure.log")), list(run_dir.iterdir())
        checks.append("worker_missing_tool_metadata_and_partial_summary")

        optional = subprocess.run(["/bin/bash", str(launcher), "prepare"],
                                  env=env | {"SIMD_KAMI_ENV_DIR": str(work / "absent"),
                                             "PATH": str(tools) + os.pathsep + str(utilities),
                                             "FDAPDE_GTEST_SOURCE": str(cache)},
                                  text=True, capture_output=True, timeout=30)
        # machines without site profiles must still prepare from an existing source cache
        assert optional.returncode == 0 and "loading Kami environment" not in optional.stdout, optional
        checks.append("optional_profiles_absent")
    print(json.dumps({"passed": True, "checks": checks, "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
                      "real_qsub_calls": 0, "ssh_calls": 0, "compiler_invocations": 0, "benchmark_invocations": 0}, indent=2))


if __name__ == "__main__":
    main()
