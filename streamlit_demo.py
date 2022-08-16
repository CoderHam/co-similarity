# Jeeves is back, and this time he's playing for keeps
import streamlit as st
import pandas as pd
import numpy as np
import similarity_search

import cohere
import difflib

bg_df = pd.read_csv("bgg_2000.csv")
co = cohere.Client(st.secrets["cohere_api_token"])

def info_from_ids(game_ranks):
    game_names = bg_df.loc[game_ranks, 'name'].to_list()
    img_urls = bg_df.loc[game_ranks, 'thumbnail'].to_list()
    game_urls = bg_df.loc[game_ranks, 'url'].to_list()
    return game_names, img_urls, game_urls

# selected_game_rank = 396
# results = similarity_search.get_similar_from_game(selected_game_rank, k=9)

# query_name, query_url, query_url = info_from_ids([selected_game_rank])
# game_names, img_urls, game_urls = info_from_ids(results)
# print("Games similar to ->", query_name)
# print('-----')
# print('\n'.join(game_names))

# sentence = "railway build across United States"
# query_embeds = co.embed(model='small', texts=[sentence]).embeddings
# results = similarity_search.get_similar_from_embeds(query_embeds, 8)
# game_names, img_urls, game_urls = info_from_ids(results)
# print("Games with descriptions similar to -> \"", sentence,"\"")
# print('-----')
# print('\n'.join(game_names))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Board Game Similarity Search",
        page_icon="https://vectorified.com/images/board-game-icon-4.png",
        layout="wide",
    )
    name_query = st.text_input('Board Game Name', '', key="game name")
    description = st.text_input('Board Game Description', '', key="game description")

    difflib.get_close_matches(bg_df.loc[:,'name'].to_list(), name_query)

