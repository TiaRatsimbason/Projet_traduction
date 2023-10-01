import streamlit as st
import pandas as pd
import numpy as np
import os
from sacrebleu import corpus_bleu
from transformers import pipeline
from translate import Translator
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import whisper
import io
# import wave
import wavio
from filesplit.merge import Merge
import tensorflow as tf
import string
import re
from tensorflow import keras
from tensorflow.keras import layers
# from keras_nlp.layers import TransformerEncoder
from tensorflow.keras.utils import plot_model
from PIL import Image
from gtts import gTTS
from extra_streamlit_components import tab_bar, TabBarItemData


title = "Traduction Sequence à Sequence"
sidebar_name = "Traduction Seq2Seq"

# !pip install transformers
# !pip install sentencepiece

@st.cache_data
def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

@st.cache_resource
def load_all_data():
    df_data_en = load_corpus('../data/preprocess_txt_en')
    df_data_fr = load_corpus('../data/preprocess_txt_fr')
    lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")
    translation_en_fr = pipeline('translation_en_to_fr', model="t5-base") 
    translation_fr_en = pipeline('translation_fr_to_en', model="Helsinki-NLP/opus-mt-fr-en")
    model_speech = whisper.load_model("base") 
    
    merge = Merge( "../data/rnn_en-fr_split",  "../data", "seq2seq_rnn-model-en-fr.h5").merge(cleanup=False)
    merge = Merge( "../data/rnn_fr-en_split",  "../data", "seq2seq_rnn-model-fr-en.h5").merge(cleanup=False)
    rnn_en_fr = keras.models.load_model("../data/seq2seq_rnn-model-en-fr.h5")
    rnn_fr_en = keras.models.load_model("../data/seq2seq_rnn-model-fr-en.h5")
    #transformer_en_fr = keras.models.load_model( "../data/transformer-model-en-fr.h5",
    #                                      custom_objects={"PositionalEmbedding": PositionalEmbedding, "TransformerDecoder": TransformerDecoder},)
    #transformer_fr_en = keras.models.load_model( "../data/transformer-model-fr-en.h5",
    #                                      custom_objects={"PositionalEmbedding": PositionalEmbedding, "TransformerDecoder": TransformerDecoder},)
    #transformer_en_fr.load_weights("../data/transformer-model-en-fr.weights.h5") 
    #transformer_fr_en.load_weights("../data/transformer-model-fr-en.weights.h5") 
    return df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech, rnn_en_fr, rnn_fr_en #, transformer_en_fr, transformer_fr_en

n1 = 0
df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech, rnn_en_fr, rnn_fr_en = load_all_data() 

# ===== Keras ====
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    lowercase=tf.strings.regex_replace(lowercase, "[à]", "a")
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

def load_vocab(file_path):
    with open(file_path, "r",  encoding="utf-8") as file:
        return file.read().split('\n')[:-1]


def decode_sequence_rnn(input_sentence, src, tgt):
    global translation_model

    vocab_size = 15000
    sequence_length = 50

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=custom_standardization,
        vocabulary = load_vocab("../data/vocab_"+src+".txt"),
    )

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
        vocabulary = load_vocab("../data/vocab_"+tgt+".txt"),
    )

    tgt_vocab = target_vectorization.get_vocabulary()
    tgt_index_lookup = dict(zip(range(len(tgt_vocab)), tgt_vocab))
    max_decoded_sentence_length = 50
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = translation_model.predict(
            [tokenized_input_sentence, tokenized_target_sentence], verbose=0)
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = tgt_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence[8:-6]

# ==== End Keras ====



def display_translation(n1, Lang):
    global df_data_src, df_data_tgt, placeholder
    
    placeholder = st.empty()
    with st.status(":sunglasses:", expanded=True):
        s = df_data_src.iloc[n1:n1+5][0].tolist()
        s_trad = []
        s_trad_ref = df_data_tgt.iloc[n1:n1+5][0].tolist()
        source = Lang[:2]
        target = Lang[-2:]
        for i in range(5):
            s_trad.append(decode_sequence_rnn(s[i], source, target))
            st.write("**"+source+"   :**  :blue["+ s[i]+"]")
            st.write("**"+target+"   :**  "+s_trad[-1])
            st.write("**ref. :** "+s_trad_ref[i])
            st.write("")
    with placeholder:
        st.write("<p style='text-align:center;background-color:red; color:white')>Score Bleu = "+str(int(round(corpus_bleu(s_trad,[s_trad_ref]).score,0)))+"%</p>", \
            unsafe_allow_html=True)
        
@st.cache_data        
def find_lang_label(lang_sel):
    global lang_tgt, label_lang
    return label_lang[lang_tgt.index(lang_sel)]

def run():

    global n1, df_data_src, df_data_tgt, translation_model, placeholder, model_speech
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en
    global lang_tgt, label_lang

    st.title(title)
    #
    st.write("## **Explications :**\n")

    st.markdown(
        """
        Enfin, nous avons réalisé une traduction :red[**Seq2Seq**] ("Sequence-to-Sequence") avec des :red[**réseaux neuronaux**].  
        La traduction Seq2Seq est une méthode d'apprentissage automatique qui permet de traduire des séquences de texte d'une langue à une autre en utilisant 
        un :red[**encodeur**] pour capturer le sens du texte source, un :red[**décodeur**] pour générer la traduction, et un :red[**vecteur de contexte**] pour relier les deux parties du modèle.
        """
    )

    lang_tgt   = ['en','fr','ab','aa','af','ak','sq','de','am','en','ar','an','hy','as','av','ae','ay','az','ba','bm','eu','bn','bi','be','bh','my','bs','br','bg','ks','ca','ch','ny','zh','si','ko','kw','co','ht','cr','hr','da','dz','gd','es','eo','et','ee','fo','fj','fi','fr','fy','gl','cy','lg','ka','el','kl','gn','gu','ha','he','hz','hi','ho','hu','io','ig','id','ia','iu','ik','ga','is','it','ja','jv','kn','kr','kk','km','kg','ki','rw','ky','rn','kv','kj','ku','lo','la','lv','li','ln','lt','lu','lb','mk','ms','ml','dv','mg','mt','gv','mi','mr','mh','mo','mn','na','nv','ng','nl','ne','no','nb','nn','nr','ie','oc','oj','or','om','os','ug','ur','uz','ps','pi','pa','fa','ff','pl','pt','qu','rm','ro','ru','se','sm','sg','sa','sc','sr','sh','sn','nd','sd','sk','sl','so','st','su','sv','sw','ss','tg','tl','ty','ta','tt','cs','ce','cv','te','th','bo','ti','to','ts','tn','tr','tk','tw','uk','ve','vi','cu','vo','wa','wo','xh','ii','yi','yo','za','zu']
    label_lang = ['Anglais','Français','Abkhaze','Afar','Afrikaans','Akan','Albanais','Allemand','Amharique','Anglais','Arabe','Aragonais','Arménien','Assamais','Avar','Avestique','Aymara','Azéri','Bachkir','Bambara','Basque','Bengali','Bichelamar','Biélorusse','Bihari','Birman','Bosnien','Breton','Bulgare','Cachemiri','Catalan','Chamorro','Chichewa','Chinois','Cingalais','Coréen','Cornique','Corse','Créolehaïtien','Cri','Croate','Danois','Dzongkha','Écossais','Espagnol','Espéranto','Estonien','Ewe','Féroïen','Fidjien','Finnois','Français','Frisonoccidental','Galicien','Gallois','Ganda','Géorgien','Grecmoderne','Groenlandais','Guarani','Gujarati','Haoussa','Hébreu','Héréro','Hindi','Hirimotu','Hongrois','Ido','Igbo','Indonésien','Interlingua','Inuktitut','Inupiak','Irlandais','Islandais','Italien','Japonais','Javanais','Kannada','Kanouri','Kazakh','Khmer','Kikongo','Kikuyu','Kinyarwanda','Kirghiz','Kirundi','Komi','Kuanyama','Kurde','Lao','Latin','Letton','Limbourgeois','Lingala','Lituanien','Luba','Luxembourgeois','Macédonien','Malais','Malayalam','Maldivien','Malgache','Maltais','Mannois','MaorideNouvelle-Zélande','Marathi','Marshallais','Moldave','Mongol','Nauruan','Navajo','Ndonga','Néerlandais','Népalais','Norvégien','Norvégienbokmål','Norvégiennynorsk','Nrebele','Occidental','Occitan','Ojibwé','Oriya','Oromo','Ossète','Ouïghour','Ourdou','Ouzbek','Pachto','Pali','Pendjabi','Persan','Peul','Polonais','Portugais','Quechua','Romanche','Roumain','Russe','SameduNord','Samoan','Sango','Sanskrit','Sarde','Serbe','Serbo-croate','Shona','Sindebele','Sindhi','Slovaque','Slovène','Somali','SothoduSud','Soundanais','Suédois','Swahili','Swati','Tadjik','Tagalog','Tahitien','Tamoul','Tatar','Tchèque','Tchétchène','Tchouvache','Télougou','Thaï','Tibétain','Tigrigna','Tongien','Tsonga','Tswana','Turc','Turkmène','Twi','Ukrainien','Venda','Vietnamien','Vieux-slave','Volapük','Wallon','Wolof','Xhosa','Yi','Yiddish','Yoruba','Zhuang','Zoulou']
    lang_src = {'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el':'modern greek', 'en': 'english', 'es': 'spanish', 'fr': 'french', \
                'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', \
                'th': 'thai', 'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}
    st.write("## **Paramètres :**\n")
    
    st.write("#### Choisissez le type de traduction:")
    # tab1, tab2, tab3 = st.tabs(["small vocab avec Keras et un GRU","Phrases à saisir", "Phrases à dicter"])
    chosen_id = tab_bar(data=[
        TabBarItemData(id="tab1", title="small vocab", description="avec Keras et un GRU"),
        TabBarItemData(id="tab2", title="Phrase personnelle", description="à saisir"),
        TabBarItemData(id="tab3", title="Phrase personnelle", description="à dicter")])
    
    TabContainerHolder = st.container()
    if chosen_id == "tab1":   
        Sens = TabContainerHolder.radio('Sens de la traduction:',('Anglais -> Français','Français -> Anglais'), horizontal=True)
        Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')

        if (Lang=='en_fr'):
            df_data_src = df_data_en
            df_data_tgt = df_data_fr
            translation_model = rnn_en_fr
        else:
            df_data_src = df_data_fr
            df_data_tgt = df_data_en
            translation_model = rnn_fr_en

        st.write("<center><h5>Architecture du modèle utilisé:</h5></center>", unsafe_allow_html=True)
        plot_model(translation_model, show_shapes=True, show_layer_names=True, show_layer_activations=True,rankdir='TB',to_file='../images/model_plot.png')
        st.image('../images/model_plot.png',use_column_width=True)

        sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrases à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
        n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
        display_translation(n1, Lang)
    elif chosen_id == "tab2":

        custom_sentence = st.text_area(label="Saisir le texte à traduire")
        l_tgt = st.selectbox("Choisir la langue cible pour Google Translate (uniquement):",lang_tgt, format_func = find_lang_label )
        st.button(label="Valider", type="primary")
        if custom_sentence!="":
            Lang_detected = lang_classifier (custom_sentence)[0]['label']
            st.write('Langue détectée : **'+lang_src.get(Lang_detected)+'**')
            audio_stream_bytesio_src = io.BytesIO()
            tts = gTTS(custom_sentence,lang=Lang_detected)
            tts.write_to_fp(audio_stream_bytesio_src)
            st.audio(audio_stream_bytesio_src)
            st.write("")
        else: Lang_detected=""
        col1, col2 = st.columns(2, gap="small") 
        with col1:
            st.write(":red[**Trad. t5-base & Helsinki**] *(Anglais/Français)*")
            audio_stream_bytesio_tgt = io.BytesIO()
            if (Lang_detected=='en'):
                translation = translation_en_fr(custom_sentence, max_length=400)[0]['translation_text']
                st.write("**fr :**  "+translation)
                st.write("")
                tts = gTTS(translation,lang='fr')
                tts.write_to_fp(audio_stream_bytesio_tgt)
                st.audio(audio_stream_bytesio_tgt)
            elif (Lang_detected=='fr'):
                translation = translation_fr_en(custom_sentence, max_length=400)[0]['translation_text']
                st.write("**en  :**  "+translation)
                st.write("")
                tts = gTTS(translation,lang='en')
                tts.write_to_fp(audio_stream_bytesio_tgt)
                st.audio(audio_stream_bytesio_tgt)
        with col2:
            st.write(":red[**Trad. Google Translate**]")
            try:
                translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
                if custom_sentence!="":
                    translation = translator.translate(custom_sentence)
                    st.write("**"+l_tgt+" :**  "+translation)
                    st.write("")
                    audio_stream_bytesio_tgt = io.BytesIO()
                    tts = gTTS(translation,lang=l_tgt)
                    tts.write_to_fp(audio_stream_bytesio_tgt)
                    st.audio(audio_stream_bytesio_tgt)
            except:
                st.write("Problème, essayer de nouveau..")

    elif chosen_id == "tab3":
        detection = st.toggle("Détection de langue ?")
        if not detection:
            l_src = st.selectbox("Choisissez la langue parlée :",lang_tgt, format_func = find_lang_label, index=1 )
        l_tgt = st.selectbox("Choisissez la langue cible  :",lang_tgt, format_func = find_lang_label )
        audio_bytes = audio_recorder (pause_threshold=1.0,  sample_rate=16000, text="Cliquez pour parler, puis attendre 2s..", \
                                      recording_color="#e8b62c", neutral_color="#1ec3bc", icon_size="6x",)
    
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            try:
                if detection:
                    # Create a BytesIO object from the audio stream
                    audio_stream_bytesio = io.BytesIO(audio_bytes)

                    # Read the WAV stream using wavio
                    wav = wavio.read(audio_stream_bytesio) 

                    # Extract the audio data from the wavio.Wav object
                    audio_data = wav.data

                    # Convert the audio data to a NumPy array
                    audio_input = np.array(audio_data, dtype=np.float32)
                    audio_input = np.mean(audio_input, axis=1)/32768
            
                    result = model_speech.transcribe(audio_input)
                    st.write("Langue détectée : "+result["language"])
                    Lang_detected = result["language"]
                    # Transcription Whisper (si result a été préalablement calculé)
                    custom_sentence = result["text"]
                else:
                    Lang_detected = l_src
                    # Transcription google
                    audio_stream = sr.AudioData(audio_bytes, 32000, 2) 
                    r = sr.Recognizer()
                    custom_sentence = r.recognize_google(audio_stream, language = Lang_detected)

                if custom_sentence!="":
                    # Lang_detected = lang_classifier (custom_sentence)[0]['label']
                    #st.write('Langue détectée : **'+Lang_detected+'**')
                    st.write("")
                    st.write("**"+Lang_detected+" :**  :blue["+custom_sentence+"]")
                    st.write("")
                    translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
                    translation = translator.translate(custom_sentence)
                    st.write("**"+l_tgt+" :**  "+translation)
                    st.write("")
                    audio_stream_bytesio_tgt = io.BytesIO()
                    tts = gTTS(translation,lang=l_tgt)
                    tts.write_to_fp(audio_stream_bytesio_tgt)
                    st.audio(audio_stream_bytesio_tgt)
                    st.write("Prêt pour la phase suivante..")
                    audio_bytes = False
            except KeyboardInterrupt:
                st.write("Arrêt de la reconnaissance vocale.")
            except:
                st.write("Problème, essayer de nouveau..")



