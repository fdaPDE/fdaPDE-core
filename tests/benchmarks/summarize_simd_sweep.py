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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""summarize saved SIMD sweeps without running benchmarks or assuming missing results"""

import argparse
import csv
import html
import json
import math
from pathlib import Path

from run_simd_sweep import (ASSIGNMENT_CASES, ASSIGNMENT_CONTROLS, ASSIGNMENT_SIZES,
                            PRODUCT_CASES, PRODUCT_SIZES, STATIC_CASES, plateau)

CASES = {case: "assignment" for case in ASSIGNMENT_CASES + ASSIGNMENT_CONTROLS}
CASES.update({case: "product" for case in PRODUCT_CASES + STATIC_CASES})
PRIMARY = {"assignment": "off:assignment", "product": "assignment:all"}


def escape(value):
    """preserve metadata as literal text inside Markdown tables"""
    if value is None:
        return "non disponibile"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("`", "\\`").replace("\n", "<br>")


def read_json(path, errors):
    """read optional provenance or retain a diagnostic for an unreadable saved file"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"{path.name}: {error}")
        return None


def records(path, name, errors):
    """prefer the full JSON records and fall back to CSV after recording JSON failures"""
    value = read_json(path / (name + ".json"), errors)
    if value is not None:
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
        errors.append(f"{name}.json: expected a list of records")
    csv_path = path / (name + ".csv")
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        except (OSError, csv.Error) as error:
            errors.append(f"{name}.csv: {error}")
    return []


def is_true(value):
    """accept the runner's JSON booleans and equivalent CSV fields"""
    return value is True or str(value).lower() == "true"


def numeric(row, key, scale=1):
    """format recorded numbers without turning absent values into zero"""
    try:
        value = float(row[key]) / scale
        return f"{value:.6g}" if math.isfinite(value) else "non disponibile"
    except (KeyError, TypeError, ValueError):
        return "non disponibile"


def row_error(row):
    """reject inconsistent summary fields or explicitly failed release checks"""
    try:
        values = [float(row[key]) for key in ("off_ns", "on_ns", "ratio", "ratio_min", "ratio_max")]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            return "nonpositive or nonfinite timing/ratio"
        if not values[3] <= values[2] <= values[4] or int(row["size"]) <= 0 or int(row["buffer_bytes"]) <= 0:
            return "inconsistent size or paired ratio range"
        for pair in row.get("pairs", []):
            if any(not is_true(result.get("verified")) or result.get("error") is not None
                   for result in (pair["off"], pair["on"])):
                return "failed coefficient verification in a recorded process"
    except (KeyError, OverflowError, TypeError, ValueError):
        return "missing or malformed timing/verification fields"
    return None


def point_order(row):
    """keep malformed point identifiers printable while valid sizes remain ordered"""
    try:
        return int(row["size"])
    except (KeyError, OverflowError, TypeError, ValueError):
        return -1


def load_dataset(path):
    """load one explicit dataset, including directories absent after a failed job"""
    errors = []
    metadata = read_json(path / "metadata.json", errors) or {}
    build = read_json(path / "build.json", errors) or {}
    if not isinstance(metadata, dict) or not isinstance(build, dict):
        errors.append("metadata/build must be JSON objects")
        metadata, build = {}, {}
    if not isinstance(metadata.get("arguments", {}), dict):
        errors.append("metadata arguments must be a JSON object")
        metadata["arguments"] = {}
    rows = records(path, "summary", errors)
    stops = records(path, "stops", errors)
    invalid = set()
    seen = set()
    for row in rows:
        case = row.get("case")
        problem = row_error(row)
        identity = (case, row.get("comparison"), str(row.get("size")))
        if case not in CASES or row.get("suite") != CASES.get(case):
            problem = "unrecognized case or suite"
        elif identity in seen:
            problem = "duplicate case/comparison/size"
        seen.add(identity)
        if problem:
            errors.append(f"{case}, size {row.get('size')}: {problem}")
            invalid.add(case)
    return {"path": path, "metadata": metadata, "build": build, "rows": rows,
            "stops": stops, "invalid": invalid, "errors": errors}


def selected_cases(dataset):
    """recover the requested suite or infer it from the explicit assignment/product directory"""
    arguments = dataset["metadata"].get("arguments", {})
    suite = arguments.get("suite", dataset["path"].name)
    selected = {case for case, kind in CASES.items() if suite not in PRIMARY or kind == suite}
    subset = arguments.get("cases")
    if subset:
        selected &= set(subset.split(",") if isinstance(subset, str) else subset)
    return selected


def primary_rows(dataset, case):
    """keep isolated comparisons separate from factorial measurements"""
    return [row for row in dataset["rows"]
            if row.get("case") == case and row.get("comparison") == PRIMARY[CASES[case]]]


def expected_sizes(dataset, case):
    """reconstruct the runner schedule, including static and three-anchor cases"""
    arguments = dataset["metadata"].get("arguments", {})
    sizes = arguments.get("sizes")
    if sizes:
        sizes = [int(value) for value in (sizes.split(",") if isinstance(sizes, str) else sizes)]
    else:
        sizes = ASSIGNMENT_SIZES if CASES[case] == "assignment" else PRODUCT_SIZES
    if arguments.get("anchors_only") or case in ASSIGNMENT_CONTROLS:
        sizes = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
    return [int(case[3:])] if case in STATIC_CASES else sizes


def case_status(dataset, case, job):
    """distinguish completed schedules, bounded runs, missing stops and process failures"""
    history = primary_rows(dataset, case)
    stop = next((row for row in dataset["stops"] if row.get("case") == case), {})
    reason = stop.get("reason")
    if case in dataset["invalid"]:
        return "fallito: dati incoerenti"
    if reason == "process_timeout":
        return "fallito: timeout; curva parziale" if history else "fallito: timeout; nessun punto"
    if not history:
        return "non misurato"
    if not stop:
        return "incompleto: stop assente"
    if reason == "confirmed_local_plateau":
        arguments = dataset["metadata"].get("arguments", {})
        threshold = arguments.get("cache_bytes", job.get("cache_bytes"))
        window = [{key: float(row[key]) for key in ("buffer_bytes", "ratio", "ratio_min", "ratio_max")}
                  for row in sorted(history, key=point_order)]
        if not is_true(stop.get("plateau_observed")) or len(window) < 4:
            return "incompleto: plateau incoerente"
        if not plateau(window, int(threshold) if threshold is not None else 0):
            return "incompleto: criterio plateau non soddisfatto"
        return "completo: plateau locale" if threshold is not None else "completo: plateau registrato; soglia ignota"
    if reason == "maximum_dimension":
        if "arguments" not in dataset["metadata"]:
            return "incompleto: schedule non disponibile"
        actual = {int(row["size"]) for row in history}
        if not set(expected_sizes(dataset, case)) <= actual:
            return "incompleto: punti dello schedule mancanti"
        return "completo: fine schedule"
    if reason in ("predicted_call_limit", "measured_call_limit"):
        return "limitato: cap previsto" if reason == "predicted_call_limit" else "limitato: cap misurato"
    return "incompleto: " + str(reason)


def table(items):
    """render explicit provenance fields without deriving unavailable status values"""
    return ["| Campo | Valore registrato |", "|---|---|"] + [
        f"| {escape(key)} | {escape(value)} |" for key, value in items]


def render(datasets, job, hardware, provenance_errors):
    """assemble all recorded points and the coverage of the 37 published cases"""
    measured = {case for dataset in datasets for case in CASES
                if primary_rows(dataset, case) and case not in dataset["invalid"]}
    requested = set().union(*(selected_cases(dataset) for dataset in datasets))
    lines = ["# Riepilogo sweep SIMD", "",
             f"Stato globale registrato del job: **{escape(job.get('status'))}**; stage: {escape(job.get('stage'))}.",
             f"Copertura primaria: **{len(measured)}/37 casi con punti validi**; {len(requested)}/37 selezionati.",
             "La presenza di punti non significa curva completa. Cap, timeout, dati mancanti e stop assenti sono distinti.",
             "I valori mancanti sono indicati come non disponibili, senza sostituirli con zero.", "",
             "## Confronti e lettura dei numeri", "",
             "Assignment usa `off:assignment` (flag 00 → 10). Product usa `assignment:all` (10 → 11),",
             "mantenendo lo stesso percorso di copia finale. `off:product` (00 → 01) e `off:all` (00 → 11)",
             "sono confronti factorial separati. OFF/ON nelle tabelle significano prima/dopo quel confronto.",
             "Ogni tempo è in millisecondi per chiamata. Il rapporto è la mediana dei rapporti delle coppie,",
             "non il rapporto delle due mediane aggregate. Sopra 1 la variante dopo impiega meno tempo.",
             "**Min–max è dispersione osservata delle coppie, non un intervallo di confidenza.**", "",
             "Un plateau locale richiede quattro dimensioni crescenti con buffer output almeno pari alla soglia configurata,",
             "rapporti centrali max/min ≤ 1,05 e dispersione max/min ≤ 1,10 per ciascun punto.",
             "Il report controlla il criterio quando la soglia è registrata. Un cap o una fine schedule non conferma un plateau.",
             "I risultati descrivono questi dati e le condizioni registrate; non stabiliscono prestazioni universali.", "",
             "## Job, Git e verifiche", ""]
    job_keys = ("git_commit", "git_branch", "git_dirty", "job_id", "hostname", "compiler", "compiler_version",
                "started_utc", "finished_utc", "status", "stage", "exit_code", "affinity_cpu", "cache_bytes", "plots",
                "googletest_source", "googletest_declared_revision")
    lines += table((key, job.get(key)) for key in job_keys)
    if job.get("parameters"):
        lines += ["", "Parametri del job:", "```json", json.dumps(job["parameters"], indent=2, ensure_ascii=False), "```"]
    lines += ["", "### Test registrati", ""]
    tests = job.get("tests")
    if tests:
        lines += ["| Modalità | Stato | Passed | Failed | Log |", "|---|---|---:|---:|---|"]
        for test in tests:
            lines.append("| " + " | ".join(escape(test.get(key)) for key in ("mode", "status", "passed", "failed", "log")) + " |")
    else:
        lines.append("Stato dei test non disponibile: nessun risultato registrato nel metadata del job.")
    lines += ["", "Configurazioni attese: `off`, `assignment`, `product`, `all`. Le modalità senza record hanno stato",
              "non disponibile, anche quando altre modalità passano. I sanitizer sono riportati soltanto se registrati."]
    if job.get("status") == "failed":
        lines.append("**Il job è fallito: nessun risultato dei test mancanti viene dedotto da quelli completati.**")
    lines += ["", "### Public smoke registrato", ""]
    smoke = job.get("public_smoke", {})
    lines += table((key, smoke.get(key)) for key in ("cases", "runs", "verified", "file"))
    lines += ["", "### Hardware registrato", ""]
    if hardware:
        lines += table((key, value) for key, value in hardware.items() if not isinstance(value, str) or "\n" not in value)
        for key, value in hardware.items():
            if not isinstance(value, str) or "\n" not in value:
                continue
            lines += [f"#### {escape(key)}", "", "```text",
                      value.replace("```", "` ` `"), "```", ""]
    else:
        lines.append("Hardware non disponibile; il report non usa quello della macchina su cui viene generato.")
    if provenance_errors:
        lines += ["", "Problemi di provenienza:"] + ["- " + escape(error) for error in provenance_errors]
    lines += ["", "## Copertura per caso", "", "| Caso | Suite | Punti primari | Stato per dataset |",
              "|---|---|---:|---|"]
    for case, suite in CASES.items():
        active = [dataset for dataset in datasets if case in selected_cases(dataset) or primary_rows(dataset, case)]
        statuses = "; ".join(f"{escape(dataset['path'].name)}: {escape(case_status(dataset, case, job))}" for dataset in active)
        count = sum(len(primary_rows(dataset, case)) for dataset in active)
        lines.append(f"| `{case}` | {suite} | {count} | {statuses or 'non selezionato'} |")
    for dataset in datasets:
        path = dataset["path"]
        metadata, build = dataset["metadata"], dataset["build"]
        lines += ["", f"## Dataset {escape(path.name)}", "", f"Percorso: `{escape(path)}`", ""]
        hashes = metadata.get("source_sha256", build.get("source_sha256", {}))
        if not isinstance(hashes, dict):
            hashes = {}
        lines += table([("compiler_version", metadata.get("compiler_version", build.get("compiler_version"))),
                        ("driver_sha256", hashes.get("tests/benchmarks/simd_sweep.cpp")),
                        ("runner_sha256", metadata.get("runner_sha256")), ("started", metadata.get("started")),
                        ("arguments", metadata.get("arguments"))])
        if dataset["errors"]:
            lines += ["", "**Problemi nei dati: il dataset non può essere presentato come pienamente verificato.**"]
            lines += ["- " + escape(error) for error in dataset["errors"]]
        for case in CASES:
            history = primary_rows(dataset, case)
            if case not in selected_cases(dataset) and not history:
                continue
            stop = next((row for row in dataset["stops"] if row.get("case") == case), {})
            lines += ["", f"### {case}", "", f"Stato: **{escape(case_status(dataset, case, job))}**.",
                      f"Stop registrato: {escape(stop.get('reason'))}; ultima size: {escape(stop.get('last_size'))}.",
                      "Plateau: " + ("registrato dal runner" if is_true(stop.get("plateau_observed")) else "non stabilito") + "."]
            if is_true(stop.get("factorial_incomplete")):
                lines.append("Factorial incompleto secondo il record del runner.")
            case_rows = [row for row in dataset["rows"] if row.get("case") == case]
            for comparison in dict.fromkeys(row.get("comparison") for row in case_rows):
                lines += ["", f"Confronto `{escape(comparison)}`:", "",
                          "| N / size | Forma | Scalar | OFF ms | ON ms | Rapporto | Min | Max | Coppie | Round | Buffer MiB | Round <1 ms |",
                          "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
                group = [row for row in case_rows if row.get("comparison") == comparison]
                for row in sorted(group, key=point_order):
                    shape = " × ".join(escape(row.get(key)) for key in
                                       (("rows", "cols") if CASES[case] == "assignment" else ("rows", "inner", "cols")))
                    values = [escape(row.get("size")), shape, escape(row.get("scalar")), numeric(row, "off_ns", 1e6),
                              numeric(row, "on_ns", 1e6), numeric(row, "ratio"), numeric(row, "ratio_min"), numeric(row, "ratio_max"),
                              escape(row.get("pair_count")), escape(row.get("rounds")), numeric(row, "buffer_bytes", 1024**2),
                              ("sì" if is_true(row["short_round"]) else "no") if "short_round" in row else "non disponibile"]
                    lines.append("| " + " | ".join(values) + " |")
            if not case_rows:
                lines.append("Nessun punto salvato per questo caso.")
    return "\n".join(lines) + "\n"


def self_test():
    """check complete, partial, bounded and failed fixtures using only temporary files"""
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "assignment"
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({"arguments": {"suite": "assignment", "cases": "vector_scale", "sizes": "3,8"}}))
        row = {"suite": "assignment", "case": "vector_scale", "comparison": "off:assignment", "size": 3,
               "rows": 3, "inner": 0, "cols": 1, "scalar": "double", "off_ns": 2000000, "on_ns": 1000000,
               "ratio": 2, "ratio_min": 1.8, "ratio_max": 2.2, "pair_count": 3, "rounds": 5, "buffer_bytes": 24}
        stop = {"case": "vector_scale", "reason": "maximum_dimension", "last_size": 3, "plateau_observed": False}
        (path / "summary.json").write_text(json.dumps([row]))
        (path / "stops.json").write_text(json.dumps([stop]))
        dataset = load_dataset(path)
        # verify that a saved stop cannot conceal a missing requested size
        assert case_status(dataset, "vector_scale", {}) == "incompleto: punti dello schedule mancanti"
        dataset["rows"].append(dict(row, size=8))
        # verify completed schedule coverage against both explicitly requested sizes
        assert case_status(dataset, "vector_scale", {}) == "completo: fine schedule"
        output = render([dataset], {}, {}, [])
        # verify ns-to-ms conversion and recorded ratio bounds from exact synthetic values
        assert "| 2 | 1 | 2 | 1.8 | 2.2 |" in output and "non un intervallo di confidenza" in output
        stop["reason"] = "measured_call_limit"
        dataset["stops"] = [stop]
        # verify that a cost cap remains bounded rather than a complete schedule or plateau
        assert case_status(dataset, "vector_scale", {}) == "limitato: cap misurato"
        dataset["stops"] = []
        # verify interrupted output using the absence of a stop record
        assert case_status(dataset, "vector_scale", {}) == "incompleto: stop assente"
        dataset["stops"] = [dict(stop, reason="process_timeout")]
        # verify process failure is retained even when earlier points are usable
        assert case_status(dataset, "vector_scale", {}).startswith("fallito: timeout")
        dataset["rows"] = [dict(row, size=size, buffer_bytes=32, ratio_min=1.99, ratio_max=2.01)
                           for size in (3, 8, 16, 32)]
        dataset["stops"] = [dict(stop, reason="confirmed_local_plateau", plateau_observed=True)]
        # verify four stable points and the recorded threshold against the runner's plateau oracle
        assert case_status(dataset, "vector_scale", {"cache_bytes": 16}) == "completo: plateau locale"
        dataset["rows"][-1].update(ratio=3, ratio_min=2.99, ratio_max=3.01)
        # verify a changed last regime prevents a falsely confirmed plateau
        assert case_status(dataset, "vector_scale", {"cache_bytes": 16}) == "incompleto: criterio plateau non soddisfatto"
        missing = load_dataset(Path(temporary) / "product")
        # verify the absent product directory preserves all 13 requested cases without fabricated timings
        assert len(selected_cases(missing)) == 13 and case_status(missing, "square_row", {}) == "non misurato"
        failed = dict(row, pairs=[{"off": {"verified": False}, "on": {"verified": True}}])
        # verify explicit failed coefficient checks prevent a successful-data classification
        assert row_error(failed) is not None
        (path / "summary.json").write_text(json.dumps([dict(row, size="invalid")]))
        malformed = render([load_dataset(path)], {}, {}, [])
        # verify malformed measurements remain reportable as failures instead of aborting the summary
        assert "fallito: dati incoerenti" in malformed and "| invalid |" in malformed
        (path / "summary.json").write_text("{")
        with (path / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        fallback = load_dataset(path)
        # verify a truncated JSON snapshot falls back to CSV while preserving its diagnostic
        assert len(fallback["rows"]) == 1 and fallback["errors"]
        job = {"status": "failed", "googletest_source": "cached", "googletest_declared_revision": "pinned",
               "tests": [{"mode": "off", "status": "passed", "passed": 1, "failed": 0}],
               "public_smoke": {"cases": 37, "runs": 148, "verified": True, "file": "public-smoke.json"}}
        failed_job = render([missing], job, {}, [])
        # verify global failure, unavailable configurations and explicit smoke counts survive rendering
        assert "**failed**" in failed_job and "nessun risultato dei test mancanti" in failed_job and "| runs | 148 |" in failed_job
    print("summary self-test passed")


def main():
    """write a report even when a job stopped before producing all expected datasets"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-metadata", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.results or args.output is None:
        parser.error("result directories and --output are required")
    errors = []
    job_path = args.job_metadata or args.results[0].parent / "job-metadata.json"
    job = read_json(job_path, errors) or {}
    hardware = read_json(job_path.parent / "hardware.json", errors) or {}
    if not isinstance(job, dict) or not isinstance(hardware, dict):
        errors.append("job metadata and hardware must be JSON objects")
        job, hardware = {}, {}
    report = render([load_dataset(path.resolve()) for path in args.results], job, hardware, errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"saved summary to {args.output}")


if __name__ == "__main__":
    main()
