# Based on the Code from http://bkanuka.com/posts/native-latex-plots/ (Practical Machine Learning)


def loading_figure_mode(develop=False):
    if develop:
        from pystatplottools.pdf_env import figure_devolpment as fma
    else:
        from pystatplottools.pdf_env import figure_management as fma
    import matplotlib.pyplot as plt
    return fma, plt
