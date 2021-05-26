import pstats
from pstats import SortKey
p = pstats.Stats('profiler_output')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
