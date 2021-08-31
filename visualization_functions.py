import traceback
#EDA libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import plotly.tools as tls
from plotly.offline import init_notebook_mode
## Wordclouds
import altair as alt

from  altair.vega import v5
from IPython.display import HTML
import json

#Basic Preprocessing
from collections import defaultdict, Counter
from process_data import *


'''
These contain template functions that I have used for visualizations from my previous works. 
The base code can be found on my Kaggle account https://www.kaggle.com/hoshi7
'''

# Defining functions for visualizations: 

def pie_plot(labels, values, colors, title):
    fig = {
      "data": [
        {
          "values": values,
          "labels": labels,
          "domain": {"x": [0, .48]},
          "name": "Job Type",
          "sort": False,
          "marker": {'colors': colors},
          "textinfo":"percent+label",
          "textfont": {'color': '#FFFFFF', 'size': 10},
          "hole": .6,
          "type": "pie"
        } ],
        "layout": {
            "title":title,
            "annotations": [
                {
                    "font": {
                        "size": 25,

                    },
                    "showarrow": False,
                    "text": ""

                }
            ]
        }
    }
    return fig



##-----------------------------------------------------------
# This whole section is for vega wordcloud chart
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("works?");
    }});
    console.log("recheck to see if it works?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>")))
    


## Defining functions for the wordcloud to be created



def word_cloud(df, pixwidth=6000, pixheight=350, column="index", counts="count"):
    data= [dict(name="dataset", values=df.to_dict(orient="records"))]
    wordcloud = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": pixwidth,
        "height": pixheight,
        "padding": 0,
        "title": "Hover to see number of occureances from all the sequences",
        "data": data
    }
    scale = dict(
        name="color",
        type="ordinal",
        range=["cadetblue", "royalblue", "steelblue", "navy", "teal"]
    )
    mark = {
        "type":"text",
        "from":dict(data="dataset"),
        "encode":dict(
            enter=dict(
                text=dict(field=column),
                align=dict(value="center"),
                baseline=dict(value="alphabetic"),
                fill=dict(scale="color", field=column),
                tooltip=dict(signal="datum.count + ' occurrances'")
            )
        ),
        "transform": [{
            "type": "wordcloud",
            "text": dict(field=column),
            "size": [pixwidth, pixheight],
            "font": "Helvetica Neue, Arial",
            "fontSize": dict(field="datum.{}".format(counts)),
            "fontSizeRange": [10, 60],
            "padding": 2
        }]
    }
    wordcloud["scales"] = [scale]
    wordcloud["marks"] = [mark]
    
    return wordcloud



def wordcloud_create(df, field):
    #Finishes in one pass instead of iterating again. Faster, saves ram usage
    ult = {}
    corpus = df[field].values.tolist()
    final = defaultdict(int) #Declaring an empty dictionary for count (Saves ram usage)
    for words in corpus:
        for word in words.split():
             final[word]+=1
    temp = Counter(final)
    for k, v in  temp.most_common(300):
        ult[k] = v
    corpus = pd.Series(ult) #Creating a dataframe from the final default dict
    return render(word_cloud(corpus.to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))