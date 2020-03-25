#from plotting_environment.working_mode import working_mode

def loading_figure_mode(working_mode):
    if working_mode is "development":
        from plotting_environment import figure_devolpment as fma
    else:
        from plotting_environment import figure_management as fma
    return fma
