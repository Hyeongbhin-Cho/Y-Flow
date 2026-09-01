# -*- coding: utf-8 -*-
# main.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import DictConfig, OmegaConf

from utils.config import add_config_override_args, apply_overrides, load_config
from utils.seed import seed_everything

COMMANDS = (
    "all",
    "flowmatch",
    "hardflow",
    "yflow",
    "safeflow",
    "uniconflow",
    "guideflow",
)

_METHOD_ORDER = (
    "flowmatch",
    "hardflow",
    "yflow",
    "safeflow",
    "uniconflow",
    "guideflow",
)

_TRAINABLE = frozenset({"flowmatch", "hardflow", "yflow", "guideflow"})


def _not_ready(name: str, action: str):
    def run(_cfg: DictConfig) -> None:
        raise NotImplementedError(f"{name} {action} is not implemented yet")

    return run


def _train_flowmatch(cfg: DictConfig) -> None:
    from train.trainer import run_train

    ckpt = run_train(cfg, method="flowmatch")
    print(f"saved {ckpt}")


def _train_hardflow(cfg: DictConfig) -> None:
    from train.hard_flow import ensure_flowmatch_ckpt

    ckpt = ensure_flowmatch_ckpt(cfg)
    print(f"hardflow backbone {ckpt}")


def _train_yflow(cfg: DictConfig) -> None:
    from train.y_flow import ensure_flowmatch_ckpt

    ckpt = ensure_flowmatch_ckpt(cfg)
    print(f"yflow backbone {ckpt}")


def _train_guideflow(cfg: DictConfig) -> None:
    from eval.guide_flow import owns_backbone

    if owns_backbone(cfg):
        from train.guide_flow import run_train_guideflow

        ckpt = run_train_guideflow(cfg)
        print(f"saved guideflow backbone {ckpt}")
        return
    from train.guide_flow import ensure_flowmatch_ckpt

    ckpt = ensure_flowmatch_ckpt(cfg)
    print(f"guideflow backbone {ckpt}")


def _eval_method(cfg: DictConfig, method: str) -> None:
    from eval.evaluate import run_eval

    run_eval(cfg, method)


_TRAIN = {
    "flowmatch": _train_flowmatch,
    "hardflow": _train_hardflow,
    "yflow": _train_yflow,
    "safeflow": _not_ready("safeflow", "train"),
    "uniconflow": _not_ready("uniconflow", "train"),
    "guideflow": _train_guideflow,
}

_EVAL = {
    "flowmatch": lambda cfg: _eval_method(cfg, "flowmatch"),
    "hardflow": lambda cfg: _eval_method(cfg, "hardflow"),
    "yflow": lambda cfg: _eval_method(cfg, "yflow"),
    "safeflow": _not_ready("safeflow", "eval"),
    "uniconflow": _not_ready("uniconflow", "eval"),
    "guideflow": lambda cfg: _eval_method(cfg, "guideflow"),
}


def _run_all(cfg: DictConfig, mode: str) -> None:
    table = _TRAIN if mode == "train" else _EVAL
    names = _METHOD_ORDER if mode == "eval" else [m for m in _METHOD_ORDER if m in _TRAINABLE]
    for name in names:
        print(f"==> {mode} {name}")
        try:
            table[name](cfg)
        except NotImplementedError as exc:
            print(exc)
    if mode == "eval":
        from eval.evaluate import write_run_metrics

        path = write_run_metrics(cfg)
        print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, DictConfig]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?", choices=COMMANDS)
    pre.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp_01_swiss_roll.yaml"),
    )
    pre.add_argument("--mode", choices=("train", "eval"), default="train")
    pre.add_argument("--run_name", default=None, help="output root: runs/{run_name}/{method}")
    pre_args, _ = pre.parse_known_args(argv)

    cfg = load_config(pre_args.config)
    parser = argparse.ArgumentParser(
        description="Y-Flow entry. --mode train|eval. CLI flags matching yaml keys override the config.",
        parents=[pre],
    )
    add_config_override_args(parser, cfg)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("command is required (all, flowmatch, hardflow, yflow, ...)")
    if not args.run_name:
        parser.error("--run_name is required")
    cfg = apply_overrides(cfg, args)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"run_name": args.run_name, "mode": args.mode}))
    return args, cfg


def main(argv: list[str] | None = None) -> None:
    args, cfg = parse_args(argv)
    seed_everything(int(cfg.seed))
    if args.command == "all":
        _run_all(cfg, args.mode)
        return
    table = _TRAIN if args.mode == "train" else _EVAL
    table[args.command](cfg)


if __name__ == "__main__":
    main()
