import os
import sys
import chevron
import datetime
import pandas as pd
from pathlib import Path

COUNTRIES = [c for c in
             (Path(__file__).parent.parent / "COUNTRIES").read_text().split("\n")
             if c]
COUNTRIES_WITH_WORLD = COUNTRIES + ["World"]

REPORT_PATH = "_section"
BASEURL = os.environ.get("RSE_SURVEY_BASEURL", "/" + Path(__file__).parent.parent.stem + "/")
REQUIRED_PATHS = ["csv", "fig", REPORT_PATH]
FIGURE_DPI = int(os.environ.get("RSE_SURVEY_FIGURE_DPI", 300))
FIGURE_TYPE = set(os.environ.get("RSE_SURVEY_FIGURE_TYPE", "png").lower().split(","))
FIGURE_TYPE.add("svg")


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

    # Fix: ensure we use low_memory=False, otherwise Pandas chunks the columns when reading the
    # dataset to determine data types which is a bit random and causes warnings. Note this uses
    # more memory however
    return pd.read_csv(cache_folder / (name + ".csv"), low_memory=False)


def make_report(file):
    def inner_decorator(generator):
        def func(**kwargs):
            base = Path(file).stem.replace("_", "-")
            year = os.environ.get("RSE_SURVEY_YEAR")
            prev_year = os.environ.get("RSE_SURVEY_YEAR_PREV")
            if year is None or prev_year is None:
                print("Set current and previous years in RSE_SURVEY_YEAR and RSE_SURVEY_YEAR_PREV environment variable")
                sys.exit(1)
            else:
                year = int(year)
                prev_year = int(prev_year)
            filename = base + ".md"
            template = first_existing(
                [Path("templates") / filename, Path("lib/templates") / filename]
            )
            if template is None:
                print("E: No template found for:", base)
                print("   Put a corresponding template in the appropriate folder")
                print("   For this year, in a 'template' subfolder of this folder")
                print(
                    "   For all years, templates folder in softwaresaved/international-survey-lib"
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

    # Fix: allow for zero data
    if len(data.index) == 0:
        return {
            "t_"
            + name: "No data found in survey."
        }

    # Fix: set lower precision for markdown
    data = data.round(decimals=2)

    return {
        "t_"
        + name: data.to_markdown(index=index)
        + "\n\n[Download CSV](%s){: .button}" % (BASEURL + csv)
    }


def table_country(country, name, data, index=True):
    csv = "csv/%s_%s.csv" % (name, slugify(country))
    data.to_csv(csv, index=index)

    # Fix: allow for zero data
    if len(data.index) == 0:
        return {
            "t_"
            + name: "No data found in survey."
        }

    # Fix: set lower precision for markdown
    data = data.round(decimals=2)

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


def figure(name, plt, country=None):
    # Fix: remove this setting, as it defaults to serif, which we don't want
    #plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.hashsalt'] = 'softwaresaved/international-survey-analysis'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    image_links = {}
    slug = f"{name}_{slugify(country)}" if country else name
    embedded_image = ""
    for figure_type in FIGURE_TYPE:
        figpath = f"fig/{slug}.{figure_type}"
        plt.tight_layout()
        plt.savefig(figpath, dpi=FIGURE_DPI)
        if figure_type == "svg":
            embedded_image = f"{{% raw %}}\n{svg_tag_text(figpath)}\n{{% endraw %}}"
        image_links[figure_type] = f"{BASEURL}{figpath}"
    plt.close('all')
    return {f"f_{name}": embedded_image + "\n\n" +
            " ".join(f"[{figure_type.upper()}]({figpath}){{: .button}}"  # image links
                     for figure_type, figpath in image_links.items()) + "\n"}


def figure_country(country, name, plt):
    return figure(name, plt, country)


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None
