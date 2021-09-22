#
# 09/19 11:34 - {builders_utils.py:208} - n_genes, n_pathways 9229 1387
# 09/19 11:34 - {builders_utils.py:208} - n_genes, n_pathways 1387 1066
# 09/19 11:34 - {builders_utils.py:208} - n_genes, n_pathways 1066 447
# 09/19 11:34 - {builders_utils.py:208} - n_genes, n_pathways 447 147
# 09/19 11:34 - {builders_utils.py:208} - n_genes, n_pathways 147 26

def run():
    nodes = [9229, 1387, 1066, 447, 147, 26, 1]

    params_count = []
    for n1, n2 in zip(nodes, nodes[1:]):
        print n1, n2
        params_count.append(n1 * n2)

    print (sum(params_count))
    print (sum(nodes))
    print (27687 * 9229)


if __name__ == "__main__":
    run()
