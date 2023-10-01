import streamlit as st
import os.path
from collections import OrderedDict
# Define TITLE, TEAM_MEMBERS and PROMOTION values, in config.py.
import config

# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
else:
    st.session_state.sidebar_state = 'auto'


st.set_page_config (
    page_title=config.TITLE,
    page_icon= "assets/faviconV2.png",
    initial_sidebar_state=st.session_state.sidebar_state
)

# Define the root folders depending on local/cloud run
thisfile = os.path.abspath(__file__)
if ('/' in thisfile): 
    os.chdir(os.path.dirname(thisfile))


# Tabs in the ./tabs folder, imported here.
from tabs import intro, exploration_tab, data_viz_tab, modelisation_dict_tab, modelisation_seq2seq_tab


with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# Add tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (exploration_tab.sidebar_name, exploration_tab),
        (data_viz_tab.sidebar_name, data_viz_tab),
        (modelisation_dict_tab.sidebar_name, modelisation_dict_tab),
        (modelisation_seq2seq_tab.sidebar_name, modelisation_seq2seq_tab),
    ]
)


def run():
    
    st.sidebar.image(
        "assets/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]
    tab.run()


if __name__ == "__main__":
    run()
