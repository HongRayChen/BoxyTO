"""Command-line entry point. Run from the repository with python -m boxyto."""
import argparse
import json

from .optimize import optimize, read_config, render_run


def main():
    parser = argparse.ArgumentParser(description="BoxyTO trimmed neural field topology optimization")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("optimize", help="Optimize a JSON problem configuration")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--device")
    run.add_argument("--epochs", type=int)
    run.add_argument("--fe-backend", choices=("cpu", "cupy"))
    run.add_argument("--no-render", action="store_true")
    render = commands.add_parser("render", help="Re-render a saved run")
    render.add_argument("output")
    render.add_argument("--device", default="cpu")
    render.add_argument("--samples", type=int)
    render.add_argument("--size", type=int)
    args = parser.parse_args()
    if args.command == "optimize":
        overrides = {
            key: getattr(args, key) for key in ("device", "epochs", "fe_backend") if getattr(args, key) is not None
        }
        optimize(read_config(args.config), args.output, options_override=overrides, render=not args.no_render)
    else:
        print(json.dumps(render_run(args.output, args.device, args.samples, args.size)))


if __name__ == "__main__":
    main()
