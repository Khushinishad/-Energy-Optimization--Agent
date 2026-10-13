"""Convert energy savings to climate and cost impact."""
from . import config


def kwh_to_co2(kwh, factor=config.EMISSION_FACTOR):
    return kwh * factor


def trees_equivalent(co2_kg):
    """Approximate number of mature trees needed to absorb this CO2 in one year."""
    return co2_kg / config.KG_CO2_PER_TREE_YEAR
