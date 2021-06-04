# How we represent nulls in holoclean.
NULL_REPR = '_nan_'

# A feature value to represent co-occurrence with NULLs, which is not applicable.
NA_COOCCUR_FV = 0

def dictify_df(frame):
    ret = {}
    for row in frame.values:
        cur_level = ret
        for elem in row[:-2]:
            if elem not in cur_level:
                cur_level[elem] = {}
            cur_level = cur_level[elem]
        cur_level[row[-2]] = row[-1]
    return ret
