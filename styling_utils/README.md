# Styling Scripts for Notebooks

This folder has scripts for automatically setting up some nice styling for notebooks.  You trade the flexibility of performing styling in your notebook for slightly simpler copy/pasting and a slightly cleaner notebook.

In your notebook...
Imports:

```
...
import matplotlib
from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
...
```

Get the script
```
%%bash
curl -o styling.py https://raw.githubusercontent.com/danhaive/4.453x-resources/master/styling_utils/styling.py &> /dev/null

```

Run the styler:

```
import styling
styling.colab(matplotlib, plt, pio, go)
```