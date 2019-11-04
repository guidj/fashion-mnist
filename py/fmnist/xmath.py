import functools

from typing import Iterator, Union


class SeqOp(object):
    @staticmethod
    def multiply(sequence: Iterator[Union[int, float]]):
        return functools.reduce(lambda x, y: x * y, sequence)

