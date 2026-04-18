import matplotlib as mpl

import matplotlib.pyplot as plt

def setup_matplotlib_style(single_column=True, figure_height=7, dpi=300, font_size=5, style='default'):
    """
    统一设置matplotlib绘图格式
    
    Args:
        figsize: 根据单栏和双栏设置图形尺寸，默认 (8.3cm, 6cm) 或 (17cm, 6cm)
        dpi: 分辨率，默认 300
        font_size: 字体大小，默认 6
        style: 样式风格，默认 'default'
        single_column: 是否为单列布局，默认 True
    """
    if single_column:
        width_cm = 8.3
    else:
        width_cm = 17.0
    width_inches = width_cm / 2.54
    height_inches = figure_height / 2.54
    figsize = (width_inches, height_inches)

    # 设置整体样式
    plt.style.use(style)
    # 预设论文排版图片尺寸
    mpl.rcParams['figure.autolayout'] = True

    
    # 设置字体
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size - 1
    mpl.rcParams['ytick.labelsize'] = font_size - 1
    mpl.rcParams['legend.fontsize'] = font_size - 1
    
    # 设置图形参数
    mpl.rcParams['figure.figsize'] = figsize
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['savefig.dpi'] = dpi
    
    # 设置线条和标记
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['lines.markeredgewidth'] = 0.75
    mpl.rcParams['boxplot.boxprops.linewidth'] = 0.75
    mpl.rcParams['boxplot.capprops.linewidth'] = 0.75
    mpl.rcParams['boxplot.flierprops.linewidth'] = 0.75
    mpl.rcParams['boxplot.medianprops.linewidth'] = 0.75
    mpl.rcParams['boxplot.meanprops.linewidth'] = 0.75
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 0.75
    
    # 设置坐标轴刻度长度
    mpl.rcParams['xtick.major.size'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['ytick.major.size'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['xtick.minor.width'] = 0.25
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['ytick.minor.width'] = 0.25

    # 设置网格
    mpl.rcParams['axes.linewidth'] = 0.5

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.7
    
    # 设置图例
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.shadow'] = False
    mpl.rcParams['legend.framealpha'] = 0.8
    mpl.rcParams['legend.edgecolor'] = 'black'
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['patch.linewidth'] = 0.5 # 设置legend边框宽度

    #
    # 设置标题与图形上边缘的距离
    mpl.rcParams['axes.titlepad'] = 2
    # 设置x和y轴标签与轴的距离
    mpl.rcParams['axes.labelpad'] = 2
    mpl.rcParams['figure.subplot.wspace'] = 0.2


def get_figure(figsize=(10, 6), dpi=100):
    """创建标准化的图形对象"""
    return plt.figure(figsize=figsize, dpi=dpi)


def save_figure(fig, filepath, dpi=100, bbox_inches='tight'):
    """保存图形，使用统一格式"""
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)