class CallPut(int):
    def __new__(cls, value):
        if value not in [1, -1]:
            raise ValueError("Input value must be either 1 or -1")
        return super().__new__(cls, value)


class UnknownOptionTypeError(ValueError):
    pass
