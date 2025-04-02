# Tools package initialization

# Import all tool modules to register them
from euclid.tools import basic
from euclid.tools import advanced
from euclid.tools import batch
from euclid.tools import agent
from euclid.tools import file_operations

# Import web tools if dependencies are available
try:
    from euclid.tools import web
    HAVE_WEB_TOOLS = True
except ImportError:
    HAVE_WEB_TOOLS = False