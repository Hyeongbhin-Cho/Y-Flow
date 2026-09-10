import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "experiments" / "exp_04_robot_arm"
RUN_ROOT = ROOT / "runs" / "exp_04_robot_arm"


class TestExp04RobotArm(unittest.TestCase):
    def test_numbered_layout_and_preserved_outputs(self):
        self.assertTrue((ROOT / "configs" / "exp_04_robot_arm.yaml").is_file())
        self.assertTrue((ROOT / "datasets" / "robot_arm" / "default").is_dir())
        self.assertTrue((TASK_ROOT / "safe_flow_mpc").is_dir())
        self.assertTrue((TASK_ROOT / "data").resolve().samefile(
            ROOT / "datasets" / "robot_arm" / "default"
        ))
        self.assertTrue((RUN_ROOT / "ood_no_replace" / "REPORT_OOD.md").is_file())

    def test_cpu_integrity_suite(self):
        command = [
            sys.executable,
            "-m",
            "unittest",
            "test_pov_projection.py",
            "test_pov_projection_phase2.py",
        ]
        subprocess.run(command, cwd=TASK_ROOT, check=True, capture_output=True, text=True)

        script = """
import importlib
for name in (
    'experiments.pov_factorial_ablation.test_factorial',
    'experiments.pov_projection_phase15_yflow_main.test_phase15',
):
    module = importlib.import_module(name)
    for attr in dir(module):
        if attr.startswith('test_'):
            getattr(module, attr)()
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=TASK_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
