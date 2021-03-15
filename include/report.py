import os
import sys
import chevron
import datetime
import pandas as pd
from pathlib import Path

# List of all countries of interest
COUNTRIES = [
    "Australia",
    "Germany",
    "Netherlands",
    "New Zealand",
    "South Africa",
    "United Kingdom",
    "United States",
]
COUNTRIES_WITH_WORLD = COUNTRIES + ["World"]

REPORT_PATH = "_section"
BASEURL = "/international-survey-analysis/"
REQUIRED_PATHS = ["csv", "fig", REPORT_PATH]
FIGURE_TYPE = os.environ.get("RSE_SURVEY_FIGURE_TYPE", "svg")


def slugify(x):
    x = x.replace("&", "").replace("/", "-")
    return "-".join(x.lower().split())


def convert_time(x):
    try:
        return datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return x


def write_cache(name, data):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()
    data.to_csv(cache_folder / (name + ".csv"))


def read_cache(name):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        raise ValueError(
            "Cached item '%s' not found, run overview_and_sampling to generate"
        )
    return pd.read_csv(cache_folder / (name + ".csv"))


def make_report(file):
    def inner_decorator(generator):
        def func(**kwargs):
            base = Path(file).stem.replace("_", "-")
            year = os.environ.get("RSE_SURVEY_YEAR")
            if year is None:
                print("Set current year in RSE_SURVEY_YEAR environment variable")
                sys.exit(1)
            else:
                year = int(year)
            filename = base + ".md"
            template = first_existing(
                [Path("templates") / filename, Path("../templates") / filename]
            )
            if template is None:
                print("E: No template found for:", base)
                print("   Put a corresponding template in the appropriate folder")
                print("   For this year, in a 'template' subfolder of this folder")
                print(
                    "   For all years, in a 'template' subfolder of the parent folder"
                )
                sys.exit(1)

            # Ensure these paths are present
            for p in map(Path, REQUIRED_PATHS):
                if not p.exists():
                    p.mkdir()

            report = generator(survey_year=year, **kwargs)
            with template.open() as fp:
                (Path(REPORT_PATH) / (filename)).write_text(chevron.render(fp, report))
            return report

        return func

    return inner_decorator


def table(name, data, index=True):
    csv = "csv/%s.csv" % name
    data.to_csv(csv, index=index)
    return {
        "t_"
        + name: data.to_markdown(index=index)
        + "\n\n[Download CSV](%s){: .button}" % (BASEURL + csv)
    }


def table_country(country, name, data, index=True):
    csv = "csv/%s_%s.csv" % (name, slugify(country))
    data.to_csv(csv, index=index)
    return {
        "t_"
        + name: data.to_markdown(index=index)
        + "\n\n[Download CSV](%s){: .button}" % (BASEURL + csv)
    }


def svg_tag_text(file):
    if not os.path.exists(file):
        return ""
    with open(file) as f:
        svg_start_token = False
        lines = []
        for line in f:
            if line.strip().startswith("<svg"):
                svg_start_token = True
            if svg_start_token:
                lines.append(line)
    return "".join(lines)


def figure(name, plt):
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.hashsalt'] = 'softwaresaved/international-survey-analysis'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'sans-serif']
    figpath = f"fig/{name}.{FIGURE_TYPE}"
    plt.savefig(figpath, dpi=os.environ.get('RSE_SURVEY_FIGURE_DPI', 300))
    plt.close('all')
    if FIGURE_TYPE == "svg":
        # Embed svg to allow font-family use from CSS
        return {f"f_{name}": f"{{% raw %}}\n{svg_tag_text(figpath)}\n{{% endraw %}}"}
    else:
        return {f"f_{name}": f"![{name}]({BASEURL}{figpath})"}


def figure_country(country, name, plt):
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.hashsalt'] = 'softwaresaved/international-survey-analysis'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'sans-serif']
    slug = slugify(country)
    figpath = f"fig/{name}_{slug}.{FIGURE_TYPE}"
    plt.savefig(figpath, dpi=os.environ.get('RSE_SURVEY_FIGURE_DPI', 300))
    plt.close('all')
    if FIGURE_TYPE == "svg":
        # Embed svg to allow font-family use from CSS
        return {f"f_{name}": f"{{% raw %}}\n{svg_tag_text(figpath)}\n{{% endraw %}}"}
    else:
        return {f"f_{name}": f"![{name}_{slug}]({BASEURL}{figpath})"}


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None
