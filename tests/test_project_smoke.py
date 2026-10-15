"""Basic smoke checks for the Healthcare Experience Agent project."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProjectSmokeTests(unittest.TestCase):
    def test_main_app_exists(self):
        self.assertTrue((ROOT / "app" / "frontend" / "app.py").is_file())

    def test_route_optimization_modules_exist(self):
        self.assertTrue((ROOT / "ai" / "optimization" / "routing" / "route_optimizer.py").is_file())
        self.assertTrue((ROOT / "ai" / "optimization" / "multi_vehicle" / "ga_vrp.py").is_file())

    def test_python_sources_compile(self):
        sources = list(ROOT.rglob("*.py"))
        self.assertGreater(len(sources), 0)
        for source in sources:
            with self.subTest(source=str(source.relative_to(ROOT))):
                compile(source.read_text(encoding="utf-8"), str(source), "exec")


if __name__ == "__main__":
    unittest.main()
