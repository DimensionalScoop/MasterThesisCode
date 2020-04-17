from multiprocessing import Pipe, Process
import numpy as np
from joblib import Parallel, delayed
from itertools import chain


def _spawn(f):
    def fun(pipe, x):
        pipe.send(f(x))
        pipe.close()

    return fun


def multiprocess(f, arguments, flatten=False):
    """returns `list([f(i) for i in arguments])`, but each f(i) is computed in its own process. f does not need to be pickle-able."""
    pipe = [Pipe() for x in arguments]
    proc = [
        Process(target=_spawn(f), args=(c, x)) for x, (p, c) in zip(arguments, pipe)
    ]
    [p.start() for p in proc]
    [p.join() for p in proc]
    if flatten:
        res = []
        [res.extend(p.recv()) for (p, c) in pipe]
        return res
    else:
        return [p.recv() for (p, c) in pipe]


def parallize(f, argument_list, cpus=16, verbose=1, flatten=False, reshape=False):
    result = Parallel(cpus, verbose=verbose)([delayed(f)(i) for i in argument_list])
    if flatten:
        result = list(chain(*result))
    if reshape is not False:
        result = list(chain(*result))
        result = np.array(result).reshape(reshape)
    return result


def iterToNumpy(f):
    def toArray(*args, **kwargs):
        return np.array(list(f(*args, **kwargs)))

    return toArray
