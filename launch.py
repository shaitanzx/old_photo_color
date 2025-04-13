import os
import ssl
import sys

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import version

#from build_launcher import build_launcher
from launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
#from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "req.txt")

    print(f"Python {sys.version}")
    print(f"Old Photo Color {version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
    def install_requirements(requirements_file):
      with open(requirements_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Игнорируем пустые строки и комментарии
                run_pip(f"install {line}", desc=line)
    if REINSTALL_ALL or not requirements_met(requirements_file):
        install_requirements(requirements_file)

    return

prepare_environment()

from demo import *
