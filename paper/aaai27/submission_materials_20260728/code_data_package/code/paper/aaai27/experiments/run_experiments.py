from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name("experiment_manifest.json")
STATE_DIR = REPO_ROOT / "outputs/aaai27_experiments/_state"


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tasks = index_tasks(manifest["tasks"])
    selected = select_tasks(tasks, args.task, args.group)
    environment = build_environment(manifest.get("defaults", {}))

    if args.action == "audit":
        rows = [audit_task(task, environment) for task in selected]
        print_audit(rows)
        write_audit(rows)
        return

    if args.action == "collect":
        command = [sys.executable, str(Path(__file__).with_name("collect_results.py"))]
        subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)
        return

    order = dependency_order(tasks, [task["id"] for task in selected])
    failures = 0
    for task_id in order:
        task = tasks[task_id]
        status = audit_task(task, environment)
        if status["status"] == "complete" and not args.force:
            print(f"[reuse] {task_id}: {status['evidence']}", flush=True)
            continue
        if args.dry_run:
            print(f"[dry-run:{status['status']}] {task_id}", flush=True)
            for command in task.get("commands", []):
                print(f"  {expand(command, environment)}", flush=True)
            continue
        if task.get("kind") in {"artifact", "manual", "external"} and not task.get("commands"):
            print(f"[blocked:{task.get('kind')}] {task_id}: {task.get('note', 'required input is missing')}", flush=True)
            failures += 1
            if not args.keep_going:
                break
            continue
        if not run_task(task, environment):
            failures += 1
            if not args.keep_going:
                break
    if failures:
        raise SystemExit(f"Experiment run stopped with {failures} incomplete/failed task(s).")


def run_task(task: dict[str, Any], environment: dict[str, str]) -> bool:
    task_id = task["id"]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / f"{task_id}.log"
    state_path = STATE_DIR / f"{task_id}.json"
    started = now()
    write_state(state_path, task_id, "running", started=started)
    print(f"[run] {task_id}: log={log_path}", flush=True)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started}] task={task_id}\n")
            for raw_command in task.get("commands", []):
                command = expand(raw_command, environment)
                log.write(f"$ {command}\n")
                log.flush()
                process = subprocess.Popen(
                    ["bash", "-lc", command],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)
    except (OSError, subprocess.CalledProcessError) as exc:
        write_state(state_path, task_id, "failed", started=started, finished=now(), error=str(exc))
        print(f"[failed] {task_id}: {exc}", flush=True)
        return False

    status = audit_task(task, environment)
    if status["status"] != "complete":
        write_state(state_path, task_id, "incomplete", started=started, finished=now(), evidence=status["evidence"])
        print(f"[incomplete] {task_id}: command succeeded but expected evidence is missing", flush=True)
        return False
    write_state(state_path, task_id, "complete", started=started, finished=now(), evidence=status["evidence"])
    print(f"[complete] {task_id}: {status['evidence']}", flush=True)
    return True


def audit_task(task: dict[str, Any], environment: dict[str, str]) -> dict[str, str]:
    evidence_specs = task.get("evidence", [])
    if not evidence_specs:
        return {"id": task["id"], "group": task.get("group", ""), "status": "pending", "evidence": "no evidence rule"}
    descriptions = []
    complete = True
    for spec in evidence_specs:
        raw_patterns = spec.get("globs")
        if raw_patterns is None:
            raw_patterns = [spec["glob"]]
        patterns = [expand(pattern, environment) for pattern in raw_patterns]
        matches_by_path = {
            str(path.resolve()): path
            for pattern in patterns
            for path in glob_files(pattern)
        }
        matches = list(matches_by_path.values())
        matches = [item for item in matches if item.is_file() and item.stat().st_size >= int(spec.get("min_bytes", 1))]
        required = int(spec.get("min_count", 1))
        descriptions.append(f"{len(matches)}/{required} {' OR '.join(patterns)}")
        complete = complete and len(matches) >= required
    return {
        "id": task["id"],
        "group": task.get("group", ""),
        "status": "complete" if complete else "missing",
        "evidence": "; ".join(descriptions),
    }


def glob_files(pattern: str) -> list[Path]:
    wildcard_positions = [pattern.find(char) for char in "*?[" if char in pattern]
    if not wildcard_positions:
        path = Path(pattern)
        return [path] if path.is_file() else []
    first_wildcard = min(wildcard_positions)
    separator = max(pattern.rfind("/", 0, first_wildcard), pattern.rfind("\\", 0, first_wildcard))
    base = Path(pattern[:separator] or ".")
    relative_pattern = pattern[separator + 1 :]
    return [path for path in base.glob(relative_pattern) if path.is_file()]


def dependency_order(tasks: dict[str, dict[str, Any]], selected: list[str]) -> list[str]:
    result: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise ValueError(f"Dependency cycle at {task_id}")
        if task_id not in tasks:
            raise KeyError(f"Unknown dependency: {task_id}")
        visiting.add(task_id)
        for dependency in tasks[task_id].get("depends_on", []):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)
        result.append(task_id)

    for task_id in selected:
        visit(task_id)
    return result


def build_environment(defaults: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    for key, value in defaults.items():
        environment.setdefault(key, expand(value, environment))
    return environment


def expand(value: str, environment: dict[str, str]) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    previous = None
    while value != previous:
        previous = value
        value = pattern.sub(lambda match: environment.get(match.group(1), match.group(0)), value)
    return value


def select_tasks(tasks: dict[str, dict[str, Any]], ids: list[str], groups: list[str]) -> list[dict[str, Any]]:
    if not ids and not groups:
        return list(tasks.values())
    unknown = sorted(set(ids) - set(tasks))
    if unknown:
        raise SystemExit(f"Unknown tasks: {', '.join(unknown)}")
    return [task for task in tasks.values() if task["id"] in ids or task.get("group") in groups]


def index_tasks(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {item["id"]: item for item in items}
    if len(result) != len(items):
        raise ValueError("Duplicate task id in manifest")
    return result


def print_audit(rows: list[dict[str, str]]) -> None:
    width = max([len(row["id"]) for row in rows] + [4])
    for row in rows:
        print(f"{row['status']:8} {row['id']:<{width}}  {row['evidence']}")


def write_audit(rows: list[dict[str, str]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "audit.json").write_text(json.dumps({"generated_at": now(), "tasks": rows}, indent=2) + "\n", encoding="utf-8")


def write_state(path: Path, task_id: str, status: str, **extra: Any) -> None:
    payload = {"task": task_id, "status": status, **extra}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit, resume, and collect AAAI-27 experiments.")
    parser.add_argument("action", choices=["audit", "run", "collect"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
