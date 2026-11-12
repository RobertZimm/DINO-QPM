import sys
from dino_qpm.helpers.entrypoint import USAGE, parse_global_args, split_command
from dino_qpm.helpers.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    global_args, command_argv = parse_global_args(raw_argv)

    setup_logging(level=global_args.log_level)
    cmd, forwarded_argv = split_command(command_argv)
    if cmd is None:
        print(USAGE)
        return

    command_handlers = {
        "inference": "dino_qpm.inference.main:inference_cli",
        "evaluate": "dino_qpm.evaluation.main:evaluation_cli",
        "train": "dino_qpm.training.main:main_cli",
    }

    module_path, handler_name = command_handlers[cmd].split(":", maxsplit=1)
    module = __import__(module_path, fromlist=[handler_name])
    handler = getattr(module, handler_name)
    handler(forwarded_argv)


if __name__ == "__main__":
    main()
