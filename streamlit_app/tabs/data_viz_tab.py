import streamlit as st
from PIL import Image
import os
import ast
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import networkx as nx
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors


title = "Data Vizualization"
sidebar_name = "Data Vizualization"

with contextlib.redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')

# Première ligne à charger
first_line = 0
# Nombre maximum de lignes à charger
max_lines = 140000
if ((first_line+max_lines)>137860):
    max_lines = max(137860-first_line ,0)
# Nombre maximum de ligne à afficher pour les DataFrame
max_lines_to_display = 50

@st.cache_data(ttl='1h00s')
def load_data(path):
    
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        
    # On convertit les majuscules en minulcule
    data = data.lower()
    
    data = data.split('\n')
    return data[first_line:min(len(data),first_line+max_lines)]

@st.cache_data(ttl='1h00s')
def load_preprocessed_data(path,data_type):
    
    input_file = os.path.join(path)
    if data_type == 1:
        return pd.read_csv(input_file, encoding="utf-8", index_col=0)
    else:
        with open(input_file, "r",  encoding="utf-8") as f:
            data = f.read()
            data = data.split('\n')
        if data_type==0:
            data=data[:-1]
        elif data_type == 2:
            data=[eval(i) for i in data[:-1]]
        elif data_type ==3:
            data2 = []
            for d in data[:-1]:
                data2.append(ast.literal_eval(d))
            data=data2
        return data
    
@st.cache_data(ttl='1h00s')
def load_all_preprocessed_data(lang):
    txt           =load_preprocessed_data('../data/preprocess_txt_'+lang,0)
    corpus        =load_preprocessed_data('../data/preprocess_corpus_'+lang,0)
    txt_split     = load_preprocessed_data('../data/preprocess_txt_split_'+lang,3)
    df_count_word = pd.concat([load_preprocessed_data('../data/preprocess_df_count_word1_'+lang,1), load_preprocessed_data('../data/preprocess_df_count_word2_'+lang,1)]) 
    sent_len      =load_preprocessed_data('../data/preprocess_sent_len_'+lang,2)
    vec_model= KeyedVectors.load_word2vec_format('../data/mini.wiki.'+lang+'.align.vec')
    return txt, corpus, txt_split, df_count_word,sent_len, vec_model

#Chargement des textes complet dans les 2 langues
full_txt_en, full_corpus_en, full_txt_split_en, full_df_count_word_en,full_sent_len_en, vec_model_en = load_all_preprocessed_data('en')
full_txt_fr, full_corpus_fr, full_txt_split_fr, full_df_count_word_fr,full_sent_len_fr, vec_model_fr = load_all_preprocessed_data('fr')


def plot_word_cloud(text, title, masque, stop_words, background_color = "white"):
    
    mask_coloring = np.array(Image.open(str(masque)))
    # Définir le calque du nuage des mots
    wc = WordCloud(background_color=background_color, max_words=200, 
                   stopwords=stop_words, mask = mask_coloring, 
                   max_font_size=50, random_state=42)
    # Générer et afficher le nuage de mots
    fig=plt.figure(figsize= (20,10))
    plt.title(title, fontsize=25, color="green")
    wc.generate(text)
    
    # getting current axes
    a = plt.gca()
 
    # set visibility of x-axis as False
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)
 
    # set visibility of y-axis as False
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    
    plt.imshow(wc)
    # plt.show()
    st.pyplot(fig)
 
def drop_df_null_col(df):
    # Check if all values in each column are 0
    columns_to_drop = df.columns[df.eq(0).all()]
    # Drop the columns with all values as 0
    return df.drop(columns=columns_to_drop)

def calcul_occurence(df_count_word):
    nb_occurences = pd.DataFrame(df_count_word.sum().sort_values(axis=0,ascending=False))
    nb_occurences.columns = ['occurences']
    nb_occurences.index.name = 'mot'
    nb_occurences['mots'] = nb_occurences.index
    return nb_occurences

def dist_frequence_mots(df_count_word):
    
    df_count_word = drop_df_null_col(df_count_word)
    nb_occurences = calcul_occurence(df_count_word)
    
    sns.set()
    fig = plt.figure() #figsize=(4,4)
    plt.title("Nombre d'apparitions des mots", fontsize=16)

    chart = sns.barplot(x='mots',y='occurences',data=nb_occurences.iloc[:40]); 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    st.pyplot(fig)
    
def dist_longueur_phrase(sent_len):
    sns.set()
    fig = plt.figure() # figsize=(12, 6*row_nb)

    fig.tight_layout()
    chart = sns.histplot(data=sent_len, color='r', binwidth=1, binrange=[3,18])
    chart.set(title='Distribution du nb de mots/phrase'); 
    st.pyplot(fig)

def graphe_co_occurence(txt_split,corpus):
    dic = corpora.Dictionary(txt_split) # dictionnaire de tous les mots restant dans le token
    # Equivalent (ou presque) de la DTM : DFM, Document Feature Matrix
    dfm = [dic.doc2bow(tok) for tok in txt_split]

    mes_labels = [k for k, v in dic.token2id.items()]

    from gensim.matutils import corpus2csc
    term_matrice = corpus2csc(dfm)

    term_matrice = np.dot(term_matrice, term_matrice.T)

    for i in range(len(mes_labels)):
        term_matrice[i,i]= 0
    term_matrice.eliminate_zeros()

    G = nx.from_scipy_sparse_matrix(term_matrice)
    G.add_nodes = dic
    pos=nx.spring_layout(G, k=5)  # position des nodes


    fig = plt.figure();
    # plt.title("", fontsize=30, color='b',fontweight="bold")

    # nx.draw_networkx_labels(G,pos,dic,font_size=15, font_color='b', bbox={"boxstyle": "round,pad=0.2", "fc":"white", "ec":"black", "lw":"0.8", "alpha" : 0.8} )
    nx.draw_networkx_labels(G,pos,dic,font_size=8, font_color='b')
    nx.draw_networkx_nodes(G,pos, dic, \
                           node_color="tab:red", \
                           node_size=90, \
                           cmap=plt.cm.Reds_r, \
                           alpha=0.8);
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.1)

    plt.axis("off");
    st.pyplot(fig)

def proximite():
    global vec_model_en,vec_model_fr

    # Creates and TSNE model and plots it"
    labels = []
    tokens = []

    nb_words = st.slider('Nombre de mots à afficher :',8,50, value=20)
    df = pd.read_csv('../data/dict_we_en_fr',header=0,index_col=0, encoding ="utf-8", keep_default_na=False)
    words_en = df.index.to_list()[:nb_words]
    words_fr = df['Francais'].to_list()[:nb_words]

    for word in words_en: 
        tokens.append(vec_model_en[word])
        labels.append(word)
    for word in words_fr: 
        tokens.append(vec_model_fr[word])
        labels.append(word)
    tokens = pd.DataFrame(tokens)

    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    fig =plt.figure(figsize=(16, 16)) 
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    for i in range(len(x)):
        if i<nb_words  : color='green'
        else: color='blue'
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     color= color,
                     size=20)
    plt.title("Proximité des mots anglais avec leur traduction", fontsize=30, color="green")
    plt.legend(loc='best');
    st.pyplot(fig)
    

def run():
    
    global max_lines, first_line, Langue
    global full_txt_en, full_corpus_en, full_txt_split_en, full_df_count_word_en,full_sent_len_en, vec_model_en 
    global full_txt_fr, full_corpus_fr, full_txt_split_fr, full_df_count_word_fr,full_sent_len_fr, vec_model_fr 
    
    st.title(title)

    # 
    st.write("## **Paramètres :**\n")
    Langue = st.radio('Langue:',('Anglais','Français'), horizontal=True)
    first_line = st.slider('No de la premiere ligne à analyser :',0,137859)
    max_lines = st.select_slider('Nombre de lignes à analyser :',
                              options=[1,5,10,15,100, 500, 1000,'Max'])
    if max_lines=='Max':
        max_lines=137860
    if ((first_line+max_lines)>137860):
        max_lines = max(137860-first_line,0)
     
    # Chargement des textes sélectionnés (max lignes = max_lines)
    last_line = first_line+max_lines
    if (Langue == 'Anglais'):
        txt_en = full_txt_en[first_line:last_line]
        corpus_en = full_corpus_en[first_line:last_line]
        txt_split_en = full_txt_split_en[first_line:last_line]
        df_count_word_en =full_df_count_word_en.loc[first_line:last_line-1]
        sent_len_en = full_sent_len_en[first_line:last_line]
    else:
        txt_fr = full_txt_fr[first_line:last_line]
        corpus_fr = full_corpus_fr[first_line:last_line]
        txt_split_fr = full_txt_split_fr[first_line:last_line]
        df_count_word_fr =full_df_count_word_fr.loc[first_line:last_line-1]
        sent_len_fr = full_sent_len_fr[first_line:last_line]
        
    if (Langue=='Anglais'):
        st.dataframe(pd.DataFrame(data=full_txt_en,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    else:
        st.dataframe(pd.DataFrame(data=full_txt_fr,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["World Cloud", "Frequence","Distribution longueur", "Co-occurence", "Proximité"])

    with tab1:
        st.subheader("World Cloud")
        if (Langue == 'Anglais'):
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('english'))
            for e in txt_en : text += e
            plot_word_cloud(text, "English words corpus", "../images/coeur.png", stop_words)
        else:
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('french'))
            for e in txt_fr : text += e
            plot_word_cloud(text,"Mots français du corpus", "../images/coeur.png", stop_words)
            
    with tab2:
        st.subheader("Frequence d'apparition des mots")
        if (Langue == 'Anglais'):
            dist_frequence_mots(df_count_word_en)
        else:
            dist_frequence_mots(df_count_word_fr)
    with tab3:
        st.subheader("Distribution des longueurs de phases")
        if (Langue == 'Anglais'):
            dist_longueur_phrase(sent_len_en)
        else:
            dist_longueur_phrase(sent_len_fr)
    with tab4:
        st.subheader("Co-occurence des mots dans une phrase") 
        if (Langue == 'Anglais'):
            graphe_co_occurence(txt_split_en[:1000],corpus_en)
        else:
            graphe_co_occurence(txt_split_fr[:1000],corpus_fr)
    with tab5:
        st.subheader("Proximité sémantique des mots (Word Embedding)") 
        proximite()
        

