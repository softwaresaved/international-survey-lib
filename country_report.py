
import os
import collections
import pandas as pd
from pathlib import Path
import chevron

import sys
from lib.report import COUNTRIES_WITH_WORLD, BASEURL, slugify, svg_tag_text, first_existing

FIGURE_TYPE = os.environ.get("RSE_SURVEY_FIGURE_TYPE", "svg")


def read_template(templatepath):
    data = collections.defaultdict(list)
    #with open(templatepath) as fp:
    with templatepath.open() as fp:
        for line in fp:
            if not line.startswith("{{"):
                continue
            slug = (
                line.strip()
                .replace("{{{", "")
                .replace("}}}", "")
                .replace("{{", "")
                .replace("}}", "")
            )
            data[slug[0]].append(slug[2:])
    return data


def table_markup(path):
    if not Path(path).exists():
        return ""

    data = pd.read_csv(path)

    # Fix: allow for zero data
    if len(data.index) == 0:
        return ("No data found in survey.")

    # Fix: set lower precision for markdown
    data = data.round(decimals=2)

    return (
        data.to_markdown(index=False)
        + f"\n\n[Download CSV]({BASEURL}{path}){{: .button}}"
    )


def template_data(country, templatepath):
    data = {"country": country}
    country_slug = slugify(country)
    template = read_template(templatepath)
    data.update(
        {
            f"t_{key}": table_markup(f"csv/{key}_{country_slug}.csv")
            for key in template["t"]
        }
    )

    if FIGURE_TYPE == "svg":
        #figure_data = [(key, figure_path) for key in template["f"] if Path(f"fig/{key}_{country_slug}.svg").exists()]
        figure_data = []
        for key in template["f"]:
            figure_path = f"fig/{key}_{country_slug}.svg"
            if Path(figure_path).exists():
                figure_data.append((key, figure_path))

        data.update(
            {
                f"f_{key}": f"{{% raw %}}\n{svg_tag_text(value)}\n{{% endraw %}}"
                for key, value in figure_data
            }
        )
    else:
        data.update(
            {
                f"f_{key}": f"![{key}]({BASEURL}fig/{key}_{country_slug}.png)"
                for key in template["f"] if Path(f"fig/{key}_{country_slug}.png").exists()
            }
        )
    return data


def run():
    # Fix: ensure country report template is also read from any optional local override
    # of templates if it exists
    templatepath = first_existing(
        [Path("templates") / "country-report.md", Path("lib/templates") / "country-report.md"]
    )
    folder = Path("_country")
    if not folder.exists():
        folder.mkdir()
    for country in COUNTRIES_WITH_WORLD:
        with templatepath.open() as fp:
            (folder / (slugify(country) + ".md")).write_text(
                chevron.render(fp, template_data(country, templatepath))
            )


if __name__ == "__main__":
    run()
