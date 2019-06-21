import numpy as np
import pandas as pd

layers_spec = [
    [[{"k": 7, "s": 2}]],
    [[{"k": 3, "s": 2}]],  # MP
    [[{"k": 3, "s": 1}]],
    [[{"k": 3, "s": 2}]],  # MP

    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],

    [[{"k": 3, "s": 2}]],  # MP

    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],

    [[{"k": 3, "s": 2}]],  # MP

    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],
    [[{"k": 1, "s": 1}], [{"k": 1, "s": 1}, {"k": 3, "s": 1}], [{"k": 1, "s": 1}, {"k": 5, "s": 1}],[{"k": 3, "s": 1}, {"k": 1, "s": 1}]],

    [[{"k": 7, "s": 1}]],
]


def rf_finder(layers_spec):
    rin = [[1]]
    jin = [1]

    for idx, ls in enumerate(layers_spec):
        max_strides = 0
        for pipe in ls:
            # define rout_local,jout_local
            rout_local = []
            for r in rin[idx]:
                ro = None
                for krnl in pipe:
                    ro = (ro if ro is not None else r) + (krnl['k'] - 1) * jin[idx]
                rout_local.append(ro)
            if len(rin) < idx + 2:
                rin.append(rout_local)
            else:
                rin[idx + 1].extend(rout_local)
            rin[idx + 1] = sorted(list(set(rin[idx + 1])))
            strides = 1
            for krnl in pipe:
                strides *= krnl['s']
            max_strides = max(max_strides,strides)
        jin.append(jin[idx] * max_strides)
    return pd.DataFrame({"Rin": rin[:-1], "Rout": rin[1:], "Jin": jin[:-1], "Jout": jin[1:]})


results = rf_finder(layers_spec)
