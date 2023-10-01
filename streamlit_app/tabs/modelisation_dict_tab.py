import streamlit as st
import pandas as pd
import numpy as np
import os
from sacrebleu import corpus_bleu
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

title = "Traduction mot à mot"
sidebar_name = "Traduction mot à mot"

@st.cache_data(ttl='1h00s')
def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

@st.cache_data(ttl='1h00s')
def load_BOW(path, l):
    input_file = os.path.join(path)
    df1 = pd.read_csv(input_file+'1_'+l, encoding="utf-8", index_col=0)
    df2 = pd.read_csv(input_file+'2_'+l, encoding="utf-8", index_col=0)
    df_count_word  = pd.concat([df1, df2]) 
    return df_count_word

df_data_en = load_corpus('../data/preprocess_txt_en')
df_data_fr = load_corpus('../data/preprocess_txt_fr')
df_count_word_en = load_BOW('../data/preprocess_df_count_word', 'en')
df_count_word_fr = load_BOW('../data/preprocess_df_count_word', 'fr')
n1 = 0

nb_mots_en = 199 # len(corpus_en)
nb_mots_fr = 330 # len(corpus_fr)

# On modifie df_count_word en indiquant la présence d'un mot par 1 (au lieu du nombre d'occurences)
df_count_word_en = df_count_word_en[df_count_word_en==0].fillna(1)
df_count_word_fr = df_count_word_fr[df_count_word_fr==0].fillna(1)

# On triche un peu parce que new et jersey sont toujours dans la même phrase et donc dans la même classe
if ('new' in df_count_word_en.columns):
    df_count_word_en['new']=df_count_word_en['new']*2
    df_count_word_fr['new']=df_count_word_fr['new']*2

def accuracy(dict_ref,dict):
    correct_words = 0
    
    for t in dict.columns:
        if t in dict_ref.columns:
            if str(dict[t]) == str(dict_ref[t]): 
                correct_words +=1
        else: print("dict ref: manque:",t)
    print(correct_words," mots corrects / ",min(dict.shape[1],dict_ref.shape[1]))
    return correct_words/min(dict.shape[1],dict_ref.shape[1])

# ============

def calc_kmeans(l_src,l_tgt):
    global df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt

    # Algorithme de K-means
    init_centroids = df_count_word_tgt.T
    kmeans = KMeans(n_clusters = nb_mots_tgt, n_init=1, max_iter=1, init=init_centroids, verbose=0)

    kmeans.fit(df_count_word_tgt.T)

    # Centroids and labels
    centroids= kmeans.cluster_centers_
    labels = kmeans.labels_

    # Création et affichage du dictionnaire
    df_dic = pd.DataFrame(data=df_count_word_tgt.columns[kmeans.predict(df_count_word_src.T)],index=df_count_word_src.T.index,columns=[l_tgt])
    df_dic.index.name= l_src
    df_dic = df_dic.T
    # print("Dictionnaire Anglais -> Français:")
    # translation_quality['Précision du dictionnaire'].loc['K-Means EN->FR'] =round(accuracy(dict_EN_FR_ref,dict_EN_FR)*100, 2)
    # print(f"Précision du dictionnaire = {translation_quality['Précision du dictionnaire'].loc['K-Means EN->FR']}%")
    # display(dict_EN_FR)
    return df_dic

def calc_knn(l_src,l_tgt, metric):
    global df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt

    #Définition de la metrique (pour les 2 dictionnaires
    knn_metric = metric   # minkowski, cosine, chebyshev, manhattan, euclidean

    # Algorithme de KNN
    X_train = df_count_word_tgt.T
    y_train = range(nb_mots_tgt)

    # Création du classifieur et construction du modèle sur les données d'entraînement
    knn = KNeighborsClassifier(n_neighbors=1, metric=knn_metric)
    knn.fit(X_train, y_train)

    # Création et affichage du dictionnaire
    df_dic = pd.DataFrame(data=df_count_word_tgt.columns[knn.predict(df_count_word_src.T)],index=df_count_word_src.T.index,columns=[l_tgt])
    df_dic.index.name = l_src
    df_dic = df_dic.T

    # print("Dictionnaire Anglais -> Français:")
    # translation_quality['Précision du dictionnaire'].loc['KNN EN->FR'] =round(accuracy(dict_EN_FR_ref,knn_dict_EN_FR)*100, 2)
    # print(f"Précision du dictionnaire = {translation_quality['Précision du dictionnaire'].loc['KNN EN->FR']}%")
    # display(knn_dict_EN_FR)
    return df_dic

def calc_rf(l_src,l_tgt):

    # Algorithme de Random Forest
    X_train = df_count_word_tgt.T
    y_train = range(nb_mots_tgt)

    # Création du classifieur et construction du modèle sur les données d'entraînement
    rf = RandomForestClassifier(n_jobs=-1, random_state=321)
    rf.fit(X_train, y_train)

    # Création et affichage du dictionnaire
    df_dic = pd.DataFrame(data=df_count_word_tgt.columns[rf.predict(df_count_word_src.T)],index=df_count_word_src.T.index,columns=[l_tgt])
    df_dic.index.name= l_src
    df_dic = df_dic.T

    # print("Dictionnaire Anglais -> Français:")
    # translation_quality['Précision du dictionnaire'].loc['RF EN->FR'] = round(accuracy(dict_EN_FR_ref,rf_dict_EN_FR)*100, 2)
    # print(f"Précision du dictionnaire = {translation_quality['Précision du dictionnaire'].loc['RF EN->FR']}%")
    # display(rf_dict_EN_FR)
    return df_dic

def calcul_dic(Lang,Algo,Metrique):

    if Lang[:2]=='en': 
        l_src = 'Anglais'
        l_tgt = 'Francais'
    else:
        l_src = 'Francais'
        l_tgt = 'Anglais'

    if Algo=='Manuel':
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index(axis=1)
    elif Algo=='KMeans':
         df_dic = calc_kmeans(l_src,l_tgt)
    elif Algo=='KNN':
        df_dic = calc_knn(l_src,l_tgt, Metrique)
    elif Algo=='Random Forest':
         df_dic = calc_rf(l_src,l_tgt)
    else:
        df_dic = pd.read_csv('../data/dict_we_'+Lang,header=0,index_col=0, encoding ="utf-8", keep_default_na=False).T.sort_index(axis=1)

    return df_dic
# ============

def display_translation(n1,dict, Lang):
    global df_data_src, df_data_tgt, placeholder

    s = df_data_src.iloc[n1:n1+5][0].tolist()
    s_trad = []
    s_trad_ref = df_data_tgt.iloc[n1:n1+5][0].tolist()
    source = Lang[:2]
    target = Lang[-2:]
    for i in range(5):
        # for col in s.split():
        #     st.write('col: '+col)
        #     st.write('dict[col]! '+dict[col])
        s_trad.append((' '.join(dict[col].iloc[0] for col in s[i].split())))
        st.write("**"+source+"   :**  "+ s[i])
        st.write("**"+target+"   :**  "+s_trad[-1])
        st.write("**ref. :** "+s_trad_ref[i])
        st.write("")
    with placeholder:
        st.write("<p style='text-align:center;background-color:red; color:white')>Score Bleu = "+str(int(round(corpus_bleu(s_trad,[s_trad_ref]).score,0)))+"%</p>", \
                 unsafe_allow_html=True)
                     
def display_dic(df_dic):
    st.dataframe(df_dic.T, height=600)

def save_dic(path, df_dic):
    output_file = os.path.join(path)
    df_dic.T.to_csv(output_file, encoding="utf-8")
    return

def load_dic(path):
    input_file = os.path.join(path)
    return pd.read_csv(input_file, encoding="utf-8", index_col=0).T

def run():
    global df_data_src, df_data_tgt, df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt, n1, placeholder
    global df_data_en, df_data_fr, nb_mots_en, df_count_word_en, df_count_word_fr, nb_mots_en, nb_mots_fr

    st.title(title)

    #
    st.write("## **Explications :**\n")
    st.markdown(
        """
        Dans une première approche naïve, nous avons implémenté un système de traduction mot à mot.  
        Cette traduction est réalisée grâce à un dictionnaire qui associe un mot de la langue source à un mot de la langue cible, dans small_vocab  
        Ce dictionnaire est calculé de 3 manières:  
        * :red[**Manuellement**] en choisissant pour chaque mot source le mot cible. Ceci nous a permis de définir un dictionnaire de référence
        * Avec le :red[**Bag Of World**] (chaque mot dans la langue cible = une classe, BOW = features)  
        """)
    st.image("assets/BOW.jpg",use_column_width=True)
    st.markdown(
        """
        * Avec le :red[**Word Embedding**], c'est à dire en associant chaque mot à un vecteur "sémantique" de dimensions=300, et en selectionnant le vecteur de langue cible 
        le plus proche du vecteur de langue source.  

        Enfin nous calculons:  
        * la :red[**précision**] du dictionnaire par rapport à notre dictionnaire de réference (manuel)
        * le :red[**score BLEU**] ("BiLingual Evaluation Understudy"), qui mesure la précision de notre traduction par rapport à celle de notre corpus référence.  
        """
    )
    #
    st.write("## **Paramètres :**\n")
    Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
    Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')
    Algo = st.radio('Algorithme :',('Manuel', 'KMeans','KNN','Random Forest',' Word Embedding'), horizontal=True)
    Metrique = ''
    if (Algo == 'KNN'):
        Metrique = st.radio('Metrique:',('minkowski', 'cosine', 'chebyshev', 'manhattan', 'euclidean'), horizontal=True)
    """
    save_dico = st.checkbox('Save dic ?')
    if save_dico:
        dic_name = st.text_input('Nom du fichier :','../data/dict_')
    """
    if (Lang=='en_fr'):
        df_data_src = df_data_en
        df_data_tgt = df_data_fr
        df_count_word_src = df_count_word_en
        df_count_word_tgt = df_count_word_fr
        nb_mots_src = nb_mots_en
        nb_mots_tgt = nb_mots_fr
    else:
        df_data_src = df_data_fr
        df_data_tgt = df_data_en
        df_count_word_src = df_count_word_fr
        df_count_word_tgt = df_count_word_en
        nb_mots_src = nb_mots_fr
        nb_mots_tgt = nb_mots_en

    # df_data_src.columns = ['Phrase']
    sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrases à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
    n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
    """
    load_dico = st.checkbox('Load dic ?')
    if load_dico:
        dic_name = st.text_input('Nom du fichier :','../data/dict_')
        df_dic = load_dic(dic_name)
        # st.dataframe(df_dic)
    else:
        df_dic = calcul_dic(Lang,Algo,Metrique)
    
    if save_dico:
        save_dic(dic_name, df_dic)
    """
    df_dic = calcul_dic(Lang,Algo,Metrique)
    df_dic_ref = calcul_dic(Lang,'Manuel',Metrique)
    st.write("## **Dictionnaire calculé et traduction mot à mot :**\n")
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.write("#### **Dictionnaire**")
        precision = int(round(accuracy(df_dic_ref,df_dic)*100, 0))
        st.write("<p style='text-align:center;background-color:red; color:white')>Précision = {:2d}%</p>".format(precision), unsafe_allow_html=True)
        display_dic(df_dic)
    with col2:
        st.write("#### **Traduction**")
        placeholder = st.empty()
        display_translation(n1, df_dic, Lang)   
