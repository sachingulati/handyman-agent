import json
import sys

from handyman import server


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: cli.py <delegate|check|cancel> ...")
        return 1

    command, *rest = argv

    if command == "delegate":
        if len(rest) != 2:
            print("usage: cli.py delegate <task> <working_dir>")
            return 1
        task, working_dir = rest
        result = server.gemma_delegate(task, working_dir)
    elif command == "check":
        if len(rest) != 1:
            print("usage: cli.py check <job_id>")
            return 1
        result = server.gemma_check(rest[0])
    elif command == "cancel":
        if len(rest) != 1:
            print("usage: cli.py cancel <job_id>")
            return 1
        result = server.gemma_cancel(rest[0])
    else:
        print(f"unknown command: {command}")
        return 1

    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


def entrypoint() -> None:
    """Console-script entry point declared in pyproject.toml."""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
