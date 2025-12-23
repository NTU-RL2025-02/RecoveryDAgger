from setuptools import setup
import sys

assert (
    sys.version_info.major == 3 and sys.version_info.minor >= 10
), "You should have Python 3.6 and greater."

setup(
    name="recoverydagger",
    py_modules=["recoverydagger"],
    version="1.0.2",
    install_requires=[
        "mujoco",
        "cloudpickle",
        "gymnasium",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "seaborn",
        "torch",
        "tqdm",
        "moviepy",
        "opencv-python",
        "torchvision",
        "h5py",
        "hidapi",
        "pygame",
        "robosuite_models",
        "black",
        "wandb",
        "DeepDiff",
        "swig",
        "gymnasium[Box2D]",
        "stable-baselines3",
        "gymnasium_robotics",
    ],
    description="Code for RecoveryDagger. Modified from ThriftyDAgger's codebase.",
    author="NTU-RL2025-02",
)
