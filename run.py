import os
import re
import sys
import importlib
from typing import List

yeare = re.compile(r".*-(\d\d\d\d)")


def create_runner(init_modules: List[str], survey_root: str = "survey"):
    def run():
        modules = [f"{survey_root}.{s[:-3]}"
                   for s in os.listdir(survey_root) if s.endswith(".py")]
        modules = [m for m in modules if m not in init_modules]
        if "RSE_SURVEY_YEAR" not in os.environ and yeare.match(__file__):
            os.environ["RSE_SURVEY_YEAR"] = yeare.match(__file__).groups()[0]
        if len(sys.argv) == 1:
            run_modules = init_modules + modules
        elif sys.argv[1] == "init":
            run_modules = init_modules
        elif sys.argv[1] == "country":
            import lib.country_report
            lib.country_report.run()
            print("Country reports ✓")
            sys.exit(0)
        else:
            run_modules = [f"survey.{m.replace('-', '_')}" for m in sys.argv[1:]]
            modules_not_found = set(run_modules) - set(init_modules + modules)
            if modules_not_found:
                print("Error: these modules were not found, aborting.")
                print("  " + " ".join(modules_not_found))
                sys.exit(1)
        for m in run_modules:
            try:
                importlib.import_module(m).run()
                print(m, "✓")
            except Exception:
                print(m, "✗")
    return run
