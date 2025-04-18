import os
import sys
# Ensure local src directory is on PYTHONPATH so tests import local agents package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))) 