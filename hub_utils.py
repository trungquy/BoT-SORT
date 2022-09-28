"""Adapted from the brilliant https://github.com/ultralytics/yolov5"""
from pathlib import Path
import pkg_resources as pkg
from subprocess import check_output

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def check_requirements(requirements=ROOT / 'hub-requirements.txt', exclude=(), install=True, cmds=''):
    # Check installed dependencies meet YOLOv5 requirements (pass *.txt file or list of packages or single package str)
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            s += f'"{r}" '
            n += 1
    def p(s):
        print(f'BoT-SORT: {s}')

    if s and install:  # check environment variable
        p(f"requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            assert check_online(), "AutoUpdate skipped (offline)"
            p(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{n} package{'s' * (n > 1)} updated per {source}\n" \
                f"⚠️ {'Restart runtime or rerun command for updates to take effect'}\n"
            p(s)
        except Exception as e:
            p(f'❌ {e}')   

def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False
