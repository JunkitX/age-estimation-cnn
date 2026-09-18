import argparse
import subprocess
import sys
from pathlib import Path

from age_estimation_v2.common import PROJECT_ROOT, resolve_path


def data_arg(source, destination):
    return f"{source}{';' if sys.platform.startswith('win') else ':'}{destination}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="AgeEstimatorV2")
    parser.add_argument("--checkpoint", default="age_estimation_v2/checkpoints/best_model.pth")
    parser.add_argument("--onefile", action="store_true", help="Build a single exe. Slower startup, easier to copy.")
    parser.add_argument("--console", action="store_true", help="Keep a console window for debugging.")
    args = parser.parse_args()

    repo_root = PROJECT_ROOT
    package_root = Path(__file__).resolve().parents[1]
    app_path = package_root / "gui" / "app.py"
    checkpoint_path = resolve_path(args.checkpoint)
    configs_dir = package_root / "configs"
    dist_dir = package_root / "dist"
    build_dir = package_root / "build"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        args.name,
        "--noconfirm",
        "--clean",
        "--paths",
        str(repo_root),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(build_dir),
        "--add-data",
        data_arg(configs_dir, "age_estimation_v2/configs"),
        "--add-data",
        data_arg(checkpoint_path, "age_estimation_v2/checkpoints"),
        "--collect-data",
        "cv2",
        "--hidden-import",
        "PIL._tkinter_finder",
    ]

    if args.onefile:
        command.append("--onefile")
    if not args.console:
        command.append("--windowed")

    command.append(str(app_path))

    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, cwd=repo_root, check=True)

    if args.onefile:
        print(f"Built executable: {dist_dir / (args.name + '.exe')}")
    else:
        print(f"Built app directory: {dist_dir / args.name}")
        print(f"Executable: {dist_dir / args.name / (args.name + '.exe')}")


if __name__ == "__main__":
    main()
