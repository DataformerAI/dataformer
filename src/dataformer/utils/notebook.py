def in_notebook() -> bool:
    """Checks if the current code is being executed from a Jupyter Notebook.
    This is useful for better handling the `asyncio` events under `nest_asyncio`,
    as Jupyter Notebook runs a separate event loop.

    Returns:
        Whether the current code is being executed from a Jupyter Notebook.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
