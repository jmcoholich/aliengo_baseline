import pstats
from pstats import SortKey
p = pstats.Stats('prof_test')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)
