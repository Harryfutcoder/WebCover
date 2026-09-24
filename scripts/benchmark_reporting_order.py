"""Paper subject order, independent of runtime and metric computation.

These are reporting groups, not mutually exclusive software-license categories.
Keep deployments/editions explicit in the benchmark documentation.
"""

SITE_ORDER = (
    "4gaboards", "agilefant", "gadael", "petclinic", "realworld",
    "splittypie", "timeoff", "github", "nextcloud", "odoo",
)
SITE_RANK = {site: rank for rank, site in enumerate(SITE_ORDER)}


def site_sort_key(site):
    name = str(site or "").strip().casefold()
    return SITE_RANK.get(name, len(SITE_ORDER)), name


def external_row_sort_key(row):
    return (
        *site_sort_key(row.get("site")),
        str(row.get("algorithm", "")),
        str(row.get("seed", "")),
        str(row.get("run_dir", "")),
    )
