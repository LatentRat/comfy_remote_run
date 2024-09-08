from typing import Union, Optional, Any, TypeVar
from collections.abc import Iterable, Collection, Sequence, Callable, Generator, AsyncGenerator

import time

class Timer():
    def __init__(self,
                 print_format: Optional[Union[str, bool]] = None,
                 start: bool = False,
                 run_fn: Optional[Callable[[float], Any]] = None,
                 run_fn_full: Optional[Callable[[float, float, float], Any]] = None,
                 print_fn: Callable[[str], Any] = print,
                 time_fn: Callable[[], float] = time.monotonic,
                 ):
        if isinstance(print_format, bool):
            if print_format is True:
                print_format = "{secs}s"
            else:
                print_format = None

        self.print_format = print_format
        self.run_fn = run_fn
        self.run_fn_full = run_fn_full
        self.print_fn = print_fn
        self.time_fn = time_fn

        # start/end timestamps, by default monotonic time unless time_fn is set
        self.start_ts = None
        self.end_ts = None
        self.seconds: Optional[float] = None

        if start:
            self.start()

    def start(self):
        self.start_ts = self.time_fn()

    def end(self, run: bool = True) -> float:
        self.end_ts = self.time_fn()
        self.seconds = self.end_ts - self.start_ts

        if run:
            self._end()

        return self.seconds

    def _end(self):
        if self.print_format is not None:
            self.print_fn(self.print_format.format(
                self.seconds, self.seconds, self.seconds, self.seconds, self.seconds, self.seconds,
                s = self.seconds, secs = self.seconds, seconds = self.seconds,
                start = self.start_ts, end = self.end_ts,
            ))

        if self.run_fn is not None:
            self.run_fn(self.seconds)

        if self.run_fn_full is not None:
            self.run_fn_full(self.start_ts, self.end_ts, self.seconds)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()
