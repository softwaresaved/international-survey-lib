import os
import re
import sys
import importlib
import traceback
from typing import List

yeare = re.compile(r".*-(\d\d\d\d)")


def create_runner(init_modules: List[str], survey_root: str = "survey"):
    def run():
        modules = [f"{survey_root}.{s[:-3]}"
                   for s in os.listdir(survey_root) if s.endswith(".py")]
        modules = [m for m in modules if m not in init_modules]

        # Fix: just exit if current/previous survey years not defined in envionment
        if "RSE_SURVEY_YEAR" not in os.environ or "RSE_SURVEY_YEAR_PREV" not in os.environ:
            print("**** No survey year / previous survey year set in environment, exiting")
            sys.exit(1)
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
        N = len(run_modules)
        for i, m in enumerate(run_modules):
            progress = f"[{i + 1}/{N}]"
            try:
                importlib.import_module(m).run()
                print(progress, m, "DONE")
            except Exception as e:
                print(progress, m, "FAIL")
                traceback.print_exc()

    return run
