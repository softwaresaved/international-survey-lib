# Country report
import collections
import pandas as pd
from pathlib import Path
import chevron

import sys
from report import COUNTRIES_WITH_WORLD, BASEURL, slugify


def read_template(template):
    data = collections.defaultdict(list)
    with open(template) as fp:
        for line in fp:
            if not line.startswith("{{"):
                continue
            slug = line.strip().replace("{{", "").replace("}}", "")
            data[slug[0]].append(slug[2:])
    return data


def table_markup(path):
    if not Path(path).exists():
        return ""
    return pd.read_csv(path).to_markdown(index=False) + f"\n\n[Download CSV]({BASEURL}{path})"


def template_data(country):
    data = {"country": country}
    country_slug = slugify(country)
    template = read_template("../templates/country-report.md")
    data.update({
        f"t_{key}": table_markup(f"csv/{key}_{country_slug}.csv")
        for key in template["t"]
    })
    data.update(
        {
            f"f_{key}": f"![{key}]({BASEURL}fig/{key}_{country_slug}.png)"
            for key in template["f"]
        }
    )
    return data


def run():
    template = Path("../templates/country-report.md")
    folder = Path("_country")
    if not folder.exists():
        folder.mkdir()
    for country in COUNTRIES_WITH_WORLD:
        with template.open() as fp:
            (folder / (slugify(country) + ".md")).write_text(
                chevron.render(fp, template_data(country))
            )


if __name__ == "__main__":
    run()
