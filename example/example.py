# An example


def main():
    my_graph = MyGraph('example_graph.txt')

    # use 'abc' replaced 12,
    # check whether the algorithm is suitable for node_id in different formats
    seed_set_list = [{'abc'}, {'35'}]

    result_set_list = []
    cond_list = []

    for seed_set in seed_set_list:
        res, cond = my_graph.local_community_detection_pagerank_nibble(seed_set)
        result_set_list.append(res)
        cond_list.append(cond)

    for i, res in enumerate(result_set_list):
        print "result_" + str(i+1) + ": "
        print res
        print cond_list[i]


if __name__ == '__main__':
    import sys
    from os import path, curdir
    sys.path.append(path.abspath('..') + "/src/")
    from myGraph import MyGraph
    main()
