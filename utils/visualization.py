import datetime
import sys
import os
import platform

from graphviz import Digraph

from configs.genotypes import Genotype


def main(format):
    #if 'Windows' in platform.platform():
    #    os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
    #try:
    # genotypes = [Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_sparse', 2, 1), ('f_identity', 3, 2), ('f_dense', 4, 2), ('a_mean', 5, 2), ('a_max', 6, 3), ('a_max', 7, 4), ('f_dense_last', 8, 7), ('f_sparse_last', 9, 7), ('f_sparse_last', 10, 5)], concat_node=[5, 6, 7, 8, 9, 10], score_func=None), Genotype(alpha_cell=[('pre_mult', 1, 0), ('f_dense', 2, 1), ('f_dense', 3, 2), ('f_dense', 4, 3), ('a_max', 5, 2), ('a_mean', 6, 3), ('a_sum', 7, 4), ('f_identity', 8, 6), ('f_sparse_last', 9, 6), ('f_dense_last', 10, 6)], concat_node=[5, 6, 7, 8, 9, 10], score_func='sf_DisMult')]
    # genotypes = [Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_sparse', 2, 1), ('a_sum', 3, 2), ('f_sparse_last', 4, 3)], concat_node=[3, 4], score_func=None), Genotype(alpha_cell=[('pre_mult', 1, 0), ('f_sparse', 2, 1), ('a_sum', 3, 2), ('f_dense_last', 4, 3)], concat_node=[3, 4], score_func=None)]
    genotypes = [Genotype(alpha_cell=[('pre_mult', 1, 0), ('f_sparse', 2, 1), ('a_sum', 3, 2), ('f_identity', 4, 3)], concat_node=[4], score_func=None), Genotype(alpha_cell=[('pre_mult', 1, 0), ('f_sparse', 2, 1), ('a_sum', 3, 2), ('f_identity', 4, 3)], concat_node=[4], score_func=None)]
    #except AttributeError:
    #    print('{} is not specified in genotype.py'.format(genotypes))
    #    sys.exit(1)

    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for idx, geno in enumerate(genotypes):
        cell_name = '{}-{}-{}'.format("alpha_cell", idx, t)
        plot(geno.alpha_cell, geno.concat_node, geno.score_func, cell_name, format=format, directory="./vis_img")


def plot(alpha_cell, concat_node, score_func, filename, format='pdf', directory="./cell_visualize"):
    g = Digraph(
        format=format,
        graph_attr=dict(dpi='800'),
        edge_attr=dict(fontsize='20'),
        node_attr=dict(style='filled', shape='rect', align='center',
                       fontsize='20', height='0.3', width='0.3',
                       penwidth='2'),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])

    steps = len(alpha_cell)
    for i in range(steps):
        op, curr, prev = alpha_cell[i]
        if curr == 0:
            g.node('in', fillcolor='darkseagreen2')
        else:
            g.node(str(curr), fillcolor='lightblue')

    for i in range(steps):
        op, curr, prev = alpha_cell[i]
        curr = str(curr)
        prev = 'in' if prev == 0 else str(prev)
        g.edge(prev, curr, label=op, fillcolor='gray')

    if score_func:
        g.node('cat', fillcolor='lightblue')
        for i in concat_node:
            g.edge(str(i), 'cat', fillcolor='gray')
        g.node('out', fillcolor='palegoldenrod')
        g.edge('cat', 'out', label=score_func, fillcolor='gray')
    else:
        g.node('out', fillcolor='palegoldenrod')
        for i in concat_node:
            g.edge(str(i), 'out', fillcolor='gray')

    g.render(filename=filename, directory=directory, view=True)

if __name__ == '__main__':
    # support {'jpeg', 'png', 'pdf', 'tiff', 'svg', 'bmp', 'tif', 'tiff'}
    main(format='tiff')

