import argparse
import os
from pathlib import Path

# 路径常量定义
ENV_DIR = Path(__file__).absolute().parents[1]
THIS_DIR = Path(__file__).absolute().parent

def run_cmds(*cmds: str):
    # 过滤掉空字符串，防止生成的命令中出现多余空格
    cmd_str = ' '.join(map(str, filter(None, cmds)))
    print(f"Executing: {cmd_str}")
    os.system(cmd_str)

# --- 命令逻辑处理函数 ---

def handle_create(args):
    run_cmds(
        "conda", "create",
        "--prefix", str(ENV_DIR / args.name),
        f"python=={args.python}",
        '-y' if args.yes else ''
    )

def handle_clone(args):
    run_cmds(
        "conda", "create",
        "--prefix", str(ENV_DIR / args.name),
        "--clone", args.source,
        '-y' if args.yes else ''
    )

def handle_remove(args):
    run_cmds(
        "conda", "env", "remove",
        "--prefix", str(ENV_DIR / args.name),
        '-y' if args.yes else ''
    )

# --- ArgumentParser 配置 ---

def main():
    parser = argparse.ArgumentParser(description="Conda Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True  # 确保必须输入子命令

    # Create 命令
    parser_create = subparsers.add_parser("create", help="command for conda create")
    parser_create.add_argument("-n", "--name", required=True, help="env name")
    parser_create.add_argument("-py", "--python", default="3.10", help="python version, like 3.10")
    parser_create.add_argument("-y", "--yes", action="store_true", help="confirm yes")
    parser_create.set_defaults(func=handle_create)

    # Clone 命令
    parser_clone = subparsers.add_parser("clone", help="command for conda env clone")
    parser_clone.add_argument("-s", "--source", required=True, help="original env path")
    parser_clone.add_argument("-n", "--name", required=True, help="env name")
    parser_clone.add_argument("-y", "--yes", action="store_true", help="confirm yes")
    parser_clone.set_defaults(func=handle_clone)

    # Remove 命令
    parser_remove = subparsers.add_parser("remove", help="command for conda env remove")
    parser_remove.add_argument("-n", "--name", required=True, help="env name")
    parser_remove.add_argument("-y", "--yes", action="store_true", help="confirm yes")
    parser_remove.set_defaults(func=handle_remove)

    # 解析参数并分发
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()