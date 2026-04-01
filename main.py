import sys
from dino_qpm.helpers.entrypoint import configure_datasets_root_env, parse_global_args, split_command
from dino_qpm.helpers.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    global_args, command_argv = parse_global_args(raw_argv)

    setup_logging(level=global_args.log_level)
    configure_datasets_root_env()

    cmd, forwarded_argv = split_command(command_argv)

    if cmd == "inference":
        from dino_qpm.inference.main import inference_cli
        inference_cli(forwarded_argv)
        return

    elif cmd == "evaluate":
        from dino_qpm.evaluation.main import evaluation_cli
        evaluation_cli(forwarded_argv)
        return

    elif cmd == "train":
        from dino_qpm.training.main import main_cli
        main_cli(forwarded_argv)
        return

    raise ValueError(
        f"Unknown command '{cmd}'. Expected one of: train, inference, evaluate"
    )


if __name__ == "__main__":
    main()
