"""Calendar-SPCA: sparse representation learning on cyclic calendar domains."""

from .graph import (
    DEFAULT_CALENDAR_SHAPE,
    ToroidalCalendarGraph,
    flatten_calendar,
    reshape_calendar,
)
from .model import CalendarGraphFusedSparsePCA
from .stability import ComponentMatch, fit_restarts, match_components

# Public name used by the paper and documentation.  The longer class name is
# retained as an explicit alias for users who prefer the descriptive name.
CalendarSPCA = CalendarGraphFusedSparsePCA

__all__ = [
    "CalendarSPCA",
    "CalendarGraphFusedSparsePCA",
    "ComponentMatch",
    "DEFAULT_CALENDAR_SHAPE",
    "ToroidalCalendarGraph",
    "fit_restarts",
    "flatten_calendar",
    "match_components",
    "reshape_calendar",
]

__version__ = "1.0.0"
