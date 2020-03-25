import matplotlib.pyplot as plt
# Functions equivalent to figure_management

def newfig(width, **kwargs):
    ratio = kwargs.pop('ratio', None)
    fig, ax = plt.subplots(**kwargs)
    return fig, ax

def savefig(rel_save_path=None):
    plt.show()

# def save_fig(fig, rel_save_path=None):
#     print(rel_save_path)
#     if rel_save_path:
#         path_to_out_file = rel_save_path + '.pdf'
#         path_to_out_dir = os.path.dirname(os.path.join(savedir_figures, path_to_out_file))
#         if not os.path.exists(path_to_out_dir):
#             os.makedirs(path_to_out_dir)
#         # fig.set_size_inches(cols, rows)
#         # fig.savefig(save_path, dpi=700)
#         fig.savefig(os.path.join(savedir_figures, path_to_out_file), bbox_inches='tight')
#
#     else:
#         plt.show()