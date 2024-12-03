def matplotlib_setup(tick_labelsize=16, axes_labelsize=18, legend_fontsize=14):
    import matplotlib.pyplot as plt
    import seaborn as sns 
    import matplotlib as mpl
    
    sns.set(style = 'white')
    mpl.rc('xtick.minor', visible=True) 
    mpl.rc('ytick.minor', visible=True) 
    mpl.rc('xtick', direction='in', top=True, bottom=True, labelsize=tick_labelsize) 
    mpl.rc('ytick', direction='in', right=True, left=True, labelsize=tick_labelsize)
    mpl.rc('axes', labelsize=axes_labelsize)
    mpl.rc('legend', fontsize=legend_fontsize)
