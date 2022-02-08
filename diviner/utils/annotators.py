def experimental(func):
    """
    Docstring modifier for experimental APIs that are subject to change.

    :param func: A function to add an experimental annotation to
    :return: the passed in function with the decoration applied
    """
    notice = (
        "    .. Note:: Experimental: This method may change, be moved, or removed in a "
        "future release with no prior warning.\n\n"
    )
    func.__doc__ = notice + func.__doc__

    return func
