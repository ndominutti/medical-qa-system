import functools
import logging


def log():
    """ """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logging.info(f"Success {func.__name__}")
                return result
            except Exception as e:
                logging.error(
                    f"Error in {func.__name__} with args={args}, kwargs={kwargs} | Exception: {e}"
                )
                raise

        return wrapper

    return decorator
