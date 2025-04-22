# Skill registry implementation
_REGISTRY = {}

def tool(name: str | None = None, cost: str | None = None):
    """Decorator to register a function as a skill tool, with optional cost flag."""
    def decorator(fn):
        key = name or fn.__name__
        # Attach cost metadata to the function
        setattr(fn, '_skill_cost', cost)
        _REGISTRY[key] = fn
        return fn
    return decorator


def get_tool(name: str):
    """Retrieve a registered skill by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Skill '{name}' not found in registry.")


def list_tools():
    """List all registered skill names."""
    return list(_REGISTRY)

# --- Import all skill modules *after* defining the registry helpers ---

"""
Automatically import all skill modules in this package so they register with the tool registry.
"""
import importlib, pkgutil
__path__ = __import__(__name__).__path__  # ensure proper package path
for finder, module_name, ispkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
# All modules under ai_tutor/skills are now imported and registered 