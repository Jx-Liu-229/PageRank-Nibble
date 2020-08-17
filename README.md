# PageRank-Nibble
A Python implementation of PageRank-Nibble algorithm.

An example is provided in the **example** folder.

## 中文说明
实现了 PageRank-Nibble 算法，用于局部社区发现。
1. 读图，目前仅支持无向图，输入文件格式为 edge list 格式，可以为带权图或无权图，默认分隔符为 '\t'，默认注释字符为 '#'，可以通过参数调整。
2. 建图后调用类中的 local_community_detection_pagerank_nibble 方法，输入为种子集，输出对应的结果社区和结果社区的 Conductance，注意**认为种子集中的所有种子属于同一个社区**，即只返回一个社区，种子集至少需要包含 1 个种子。
3. 具体使用可以参考 example/example.py

## Ref
1. Andersen R, Chung F, Lang K. Local graph partitioning using pagerank vectors[C]//2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS'06). IEEE, 2006: 475-486.
2. https://github.com/ahollocou/multicom