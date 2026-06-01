"""Download or cache model/tokenizer assets for offline Stage 1 runs.

The script uses Hugging Face Hub APIs only when explicitly executed. It never
prints raw tokens and respects HF_ENDPOINT/HF_HOME/HF_HUB_CACHE from the
environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gpu_utils import disk_summary, hf_environment_summary
from src.logging_utils import write_json
from src.model_utils import huggingface_token_kwargs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Download Hugging Face assets for p-MoE offline usage.")
    parser.add_argument("--model_name", default="deepseek-ai/deepseek-moe-16b-base", help="HF repo id or model name.")
    parser.add_argument("--local_dir", default="models/deepseek-moe-16b-base", help="Target local directory.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument(
        "--mode",
        choices=["cache", "local"],
        default="local",
        help="cache uses HF cache only; local also materializes files in --local_dir.",
    )
    parser.add_argument("--resume_download", action="store_true", help="Resume an interrupted download when supported.")
    parser.add_argument("--allow_patterns", nargs="*", default=None, help="Optional allow patterns for snapshot_download.")
    parser.add_argument("--ignore_patterns", nargs="*", default=None, help="Optional ignore patterns for snapshot_download.")
    parser.add_argument("--summary_path", default="outputs/download_summary.json", help="Summary JSON path.")
    return parser


def main() -> None:
    """Download model assets using snapshot_download."""

    args = build_parser().parse_args()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required. Install transformers or huggingface_hub.") from exc

    local_dir = Path(args.local_dir)
    if args.mode == "local":
        local_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": args.model_name,
        "revision": args.revision,
        "resume_download": args.resume_download,
        "allow_patterns": args.allow_patterns,
        "ignore_patterns": args.ignore_patterns,
    }
    kwargs.update(huggingface_token_kwargs())
    if args.mode == "local":
        kwargs["local_dir"] = str(local_dir)
        kwargs["local_dir_use_symlinks"] = False

    try:
        downloaded_path = snapshot_download(**kwargs)
    except Exception as exc:
        text = str(exc).lower()
        if "no space left on device" in text or "disk quota" in text:
            raise RuntimeError("Download failed because disk space is insufficient. Check HF_HOME and --local_dir.") from exc
        if "401" in text or "403" in text or "gated" in text:
            raise RuntimeError("Download failed due to authorization. Set HF_TOKEN in the environment if required.") from exc
        raise RuntimeError(f"Download failed: {type(exc).__name__}: {exc}") from exc

    summary = {
        "model_name": args.model_name,
        "revision": args.revision,
        "mode": args.mode,
        "local_dir": str(local_dir),
        "downloaded_path": downloaded_path,
        "hf_environment": hf_environment_summary(),
        "disk": disk_summary([local_dir, "models", "outputs"]),
    }
    path = write_json(args.summary_path, summary)
    print(f"Wrote download summary to {path}")


if __name__ == "__main__":
    main()
