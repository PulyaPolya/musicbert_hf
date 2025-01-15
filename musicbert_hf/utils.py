from itertools import zip_longest


def zip_longest_with_error(*args):
    """
    >>> list(zip_longest_with_error([1, 2, 3], "abc"))
    [(1, 'a'), (2, 'b'), (3, 'c')]
    >>> list(zip_longest_with_error([1, 2, 3], "ab"))
    Traceback (most recent call last):
    ValueError: At least one iterable is exhausted before the others
    >>> list(zip_longest_with_error([1, 2], "abc"))
    Traceback (most recent call last):
    ValueError: At least one iterable is exhausted before the others
    >>> list(zip_longest_with_error([], ""))
    []
    """
    sentinel = object()
    for results in zip_longest(*args, fillvalue=sentinel):
        if sentinel in results:
            raise ValueError("At least one iterable is exhausted before the others")
        yield results
