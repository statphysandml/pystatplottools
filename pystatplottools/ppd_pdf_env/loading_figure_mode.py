
def loading_figure_mode(develop=False):
    if develop:
        from pystatplottools.ppd_pdf_env import figure_devolpment as fma
    else:
        from pystatplottools.ppd_pdf_env import figure_management as fma
    import matplotlib.pyplot as plt
    return fma, plt