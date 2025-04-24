import pytest
from ai_tutor.skills import SKILL_REGISTRY

def test_all_skills_have_cost_and_registry():
    assert SKILL_REGISTRY, "Skill registry should not be empty."
    for name, fn in SKILL_REGISTRY.items():
        assert hasattr(fn, '_skill_cost'), f"Skill '{name}' is missing _skill_cost attribute."
        cost = getattr(fn, '_skill_cost')
        assert cost in {"low", "medium", "high"}, f"Skill '{name}' has invalid cost tag: {cost}" 