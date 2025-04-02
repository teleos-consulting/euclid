"""Package for structured function implementations."""

# Import web functions if dependencies are available
try:
    from euclid.functions import web
    HAVE_WEB_FUNCTIONS = True
except ImportError:
    HAVE_WEB_FUNCTIONS = False