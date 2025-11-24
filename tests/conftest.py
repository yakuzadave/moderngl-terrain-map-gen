from src.utils import gl_context
import pytest
import moderngl
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(scope="session")
def ctx():
    """
    Creates a ModernGL context for the entire test session.
    Uses standalone context (headless) if possible.
    """
    try:
        # Try to use the project's context creation logic
        context = gl_context.create_context(standalone=True)
    except Exception:
        # Fallback
        context = moderngl.create_standalone_context()

    yield context
    context.release()
