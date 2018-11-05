https://markhaa.se/debugging-python-memory-leaks.html
https://pythonhosted.org/Pympler/index.html
https://mg.pov.lt/objgraph/objgraph.html#objgraph.show_backrefs
http://cosven.me/blogs/54

g =objgraph.show_backrefs(objs, max_depth=5)
g.view()


import gc
all_objects = gc.get_objects()
import torch
from collections import Counter
type_counter = Counter()

for o in all_objects:
    type_counter[type(o)] += 1
