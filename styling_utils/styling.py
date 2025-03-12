def colab(matplotlib, plt, pio, go):
    import google
    is_dark = google.colab.output.eval_js(
        'document.documentElement.matches("[theme=dark]")'
    )

    matplotlib.rcParams["figure.dpi"] = 100
    plt.rcParams["hatch.color"] = "white"

    if is_dark:
        # load style sheet for matplotlib, a plotting library we use for 2D visualizations
        plt.style.use(
            "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
        )
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": (0.22, 0.22, 0.22, 1.0),
                "axes.facecolor": (0.22, 0.22, 0.22, 1.0),
                "savefig.facecolor": (0.22, 0.22, 0.22, 1.0),
                "grid.color": (0.4, 0.4, 0.4, 1.0),
            }
        )

        plotly_template = pio.templates["plotly_dark"]
        pio.templates["draft"] = go.layout.Template(
            layout=dict(
                plot_bgcolor="rgba(56,56,56,0)",
                paper_bgcolor="rgba(56,56,56,0)",
            )
        )
        pio.templates.default = "plotly_dark+draft"
    else:
        plt.style.use(
            "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
    )