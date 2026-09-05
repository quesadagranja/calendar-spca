"""Calendar-SPCA: sparse representation learning on cyclic calendar domains."""

from .api import CalendarSPCA
from .graph import (
    DEFAULT_CALENDAR_SHAPE,
    ToroidalCalendarGraph,
    flatten_calendar,
    reshape_calendar,
)
from .stability import ComponentMatch, fit_restarts, match_components

__all__ = [
    "CalendarSPCA",
    "ComponentMatch",
    "DEFAULT_CALENDAR_SHAPE",
    "ToroidalCalendarGraph",
    "fit_restarts",
    "flatten_calendar",
    "match_components",
    "reshape_calendar",
]

__version__ = "1.0.0"
