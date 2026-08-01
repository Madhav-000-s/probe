"""Rubric compilation: job description + resume -> a set of competencies with
priors. The compiler maps onto the fixed taxonomy and never invents ids."""

from probe.rubric.taxonomy import Taxonomy, load_taxonomy

__all__ = ["Taxonomy", "load_taxonomy"]
