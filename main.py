import sys
from dino_qpm.helpers.entrypoint import configure_datasets_root_env, split_command


def main(argv: list[str] | None = None) -> None:
    configure_datasets_root_env()

    cmd, forwarded_argv = split_command(
        list(sys.argv[1:] if argv is None else argv))

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
