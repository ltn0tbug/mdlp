# def decorator_verbose(func):
#     def wrapper_function(*args, **kwargs):
#         _self = args[0]
#         source = args[1]

#         print("Instance: {} - Source Type: {}".format(_self.__class__.__name__, type(source)), end="")
#         if hasattr(source, 'shape'):
#             print(" - Source shape: {}".format(source.shape))
#         else:
#             print()
#         x = func(*args, **kwargs)
#         return x

#     return wrapper_function


def list_type(source):
    if not isinstance(source, list):
        return [
            source.__class__.__name__,
        ]
    source_type = []
    for src in source:
        source_type += list_type(src)

    return source_type


def list_shape(source):
    if not isinstance(source, list):
        return [
            source.shape if hasattr(source, "shape") else None,
        ]
    source_shape = []
    for src in source:
        source_shape += list_shape(src)

    return source_shape


def list_pipe(pipe):
    if not isinstance(pipe, list):
        return [
            pipe.__class__.__name__,
        ]
    pipe_type = []
    for p in pipe:
        pipe_type += list_pipe(p)

    return pipe_type

