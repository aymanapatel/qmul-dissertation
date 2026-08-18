"""WCAG criterion and issue-family registry for evidence-aware routing."""

from .criteria import CriterionRegistry, CriterionRecord
from .families import FamilyRegistry, FamilyRecord

__all__ = ["CriterionRegistry", "CriterionRecord", "FamilyRegistry", "FamilyRecord"]
