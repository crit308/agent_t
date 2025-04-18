import os
# Override package path to prefer local src/agents implementation
__path__.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'agents')))
from src.agents import function_tool 