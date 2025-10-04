from shapely.errors import GEOSException

def safe_geometry_op(op, *args, **kwargs):
    """
    Run a shapely geometry operation, catching GEOSException and returning None if it fails.
    """
    try:
        return op(*args, **kwargs)
    except GEOSException:
        return None
