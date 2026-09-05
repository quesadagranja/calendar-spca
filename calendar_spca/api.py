"""Public estimator API for Calendar-SPCA."""

from .model import CalendarGraphFusedSparsePCA


class CalendarSPCA(CalendarGraphFusedSparsePCA):
    """Calendar-SPCA estimator.

    This class provides the public, paper-aligned estimator name while reusing
    the established implementation in :class:`CalendarGraphFusedSparsePCA`.
    """

