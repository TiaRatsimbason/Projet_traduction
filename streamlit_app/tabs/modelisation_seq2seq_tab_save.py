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
import wave
import wavio


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
    return df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech

n1 = 0
df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech = load_all_data() 
lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")

def display_translation(n1, Lang):
    global df_data_src, df_data_tgt, placeholder
    
    with st.status(":sunglasses:", expanded=True):
        s = df_data_src.iloc[n1:n1+5][0].tolist()
        s_trad = []
        s_trad_ref = df_data_tgt.iloc[n1:n1+5][0].tolist()
        source = Lang[:2]
        target = Lang[-2:]
        for i in range(5):
            s_trad.append(translation_model(s[i], max_length=500)[0]['translation_text'].lower())
            st.write("**"+source+"   :**  "+ s[i])
            st.write("**"+target+"   :**  "+s_trad[-1])
            st.write("**ref. :** "+s_trad_ref[i])
            st.write("")
    with placeholder:
        st.write("<p style='text-align:center;background-color:red; color:white')>Score Bleu = "+str(int(round(corpus_bleu(s_trad,[s_trad_ref]).score,0)))+"%</p>", \
            unsafe_allow_html=True)

def run():

    global n1, df_data_src, df_data_tgt, translation_model, placeholder, model_speech
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en

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
    #
    lang = { 'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el':'modern greek', 'en': 'english', 'es': 'spanish', 'fr': 'french', \
            'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', \
            'th': 'thai', 'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}

    st.write("## **Paramètres :**\n")
    
    choice = st.radio("Choisissez le type de traduction:",["small vocab","Phrases à saisir", "Phrases à dicter"], horizontal=True)

    if choice == "small vocab":
        Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
        Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')

        if (Lang=='en_fr'):
            df_data_src = df_data_en
            df_data_tgt = df_data_fr
            translation_model = translation_en_fr
        else:
            df_data_src = df_data_fr
            df_data_tgt = df_data_en
            translation_model = translation_fr_en

        sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrases à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
        n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
        placeholder = st.empty()
        display_translation(n1, Lang)
    elif choice == "Phrases à saisir":

        custom_sentence = st.text_area(label="Saisir le texte à traduire")
        l_tgt = st.selectbox("Choisir la langue cible pour Google Translate (uniquement):",['en','fr','af','am','ar','arn','as','az','ba','be','bg','bn','bo','br','bs','ca','co','cs','cy','da','de','dsb','dv','el','en','es','et','eu','fa','fi','fil','fo','fr','fy','ga','gd','gl','gsw','gu','ha','he','hi','hr','hsb','hu','hy','id','ig','ii','is','it','iu','ja','ka','kk','kl','km','kn','ko','kok','ckb','ky','lb','lo','lt','lv','mi','mk','ml','mn','moh','mr','ms','mt','my','nb','ne','nl','nn','no','st','oc','or','pa','pl','prs','ps','pt','quc','qu','rm','ro','ru','rw','sa','sah','se','si','sk','sl','sma','smj','smn','sms','sq','sr','sv','sw','syc','ta','te','tg','th','tk','tn','tr','tt','tzm','ug','uk','ur','uz','vi','wo','xh','yo','zh','zu'] )
        st.button(label="Valider", type="primary")
        if custom_sentence!="":
            Lang_detected = lang_classifier (custom_sentence)[0]['label']
            st.write('Langue détectée : **'+lang.get(Lang_detected)+'**')
            st.write("")
        else: Lang_detected=""
        col1, col2 = st.columns(2, gap="small") 
        with col1:
            st.write(":red[**Trad. t5-base & Helsinki**] *(Anglais/Français)*")
            if (Lang_detected=='en'):
                st.write("**fr :**  "+translation_en_fr(custom_sentence, max_length=400)[0]['translation_text'])
            elif (Lang_detected=='fr'):
                st.write("**en  :**  "+translation_fr_en(custom_sentence, max_length=400)[0]['translation_text'])
        with col2:
            st.write(":red[**Trad. Google Translate**]")
            translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
            if custom_sentence!="":
                st.write("**"+l_tgt+" :**  "+translator.translate(custom_sentence))

    elif choice == "Phrases à dicter":
        detection = st.toggle("Détection de langue ?")
        if not detection:
            l_src = st.selectbox("Choisissez la langue parlée :",['fr','en','es','de','it','nl'])
        l_tgt = st.selectbox("Choisissez la langue cible  :",['en','fr','af','am','ar','arn','as','az','ba','be','bg','bn','bo','br','bs','ca','co','cs','cy','da','de','dsb','dv','el','en','es','et','eu','fa','fi','fil','fo','fr','fy','ga','gd','gl','gsw','gu','ha','he','hi','hr','hsb','hu','hy','id','ig','ii','is','it','iu','ja','ka','kk','kl','km','kn','ko','kok','ckb','ky','lb','lo','lt','lv','mi','mk','ml','mn','moh','mr','ms','mt','my','nb','ne','nl','nn','no','st','oc','or','pa','pl','prs','ps','pt','quc','qu','rm','ro','ru','rw','sa','sah','se','si','sk','sl','sma','smj','smn','sms','sq','sr','sv','sw','syc','ta','te','tg','th','tk','tn','tr','tt','tzm','ug','uk','ur','uz','vi','wo','xh','yo','zh','zu'] )
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
                else:
                    Lang_detected = l_src


                # Transcription google
                audio_stream = sr.AudioData(audio_bytes, 32000, 2) 
                r = sr.Recognizer()
                custom_sentence = r.recognize_google(audio_stream, language = Lang_detected)
                # Transcription Whisper (si result a été préalablement calculé)
                # custom_sentence = result["text"])

                if custom_sentence!="":
                    # Lang_detected = lang_classifier (custom_sentence)[0]['label']
                    #st.write('Langue détectée : **'+Lang_detected+'**')
                    st.write("")
                    st.write("**"+Lang_detected+" :**  :blue["+custom_sentence+"]")
                    st.write("")
                    translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
                    st.write("**"+l_tgt+" :**  "+translator.translate(custom_sentence))
                    st.write("")
                    st.write("Prêt pour la phase suivante..")
                    audio_bytes = False
            except KeyboardInterrupt:
                st.write("Arrêt de la reconnaissance vocale.")
            except:
                st.write("Problème, essayer de nouveau..")



