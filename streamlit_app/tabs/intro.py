import streamlit as st



title = "Système de traduction pour lunettes connectées"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    # st.image("assets/tough-communication.gif",use_column_width=True)
    st.image("assets/miss-honey-glasses-off.gif",use_column_width=True)
    st.title(title)

    st.markdown("---")
    
    st.header("**Contexte**")

    st.markdown(
        """
        Les personnes malentendantes souffrent d’un problème auditif et se trouvent donc dans l’incapacité de communiquer aisément avec autrui.
        Par ailleurs, toute personne se trouvant dans un pays étranger dont il ne connaît pas la langue se trouve dans la situation d’une personne malentendante.
        Les lunettes connectées sont dotées de la technologie de reconnaissance vocale avec des algorithmes de deep learning en intelligence artificielle.
        Elles permettent de localiser la voix d’un interlocuteur puis d’afficher sur les verres la transcription textuelle en temps réel. A partir de cette transcription, il est possible d’:red[**afficher la traduction dans la langue du porteur de ces lunettes**].

        """
    )
    st.header("**Objectifs**")

    st.markdown(
        """
        L’objectif de ce projet est d’adapter un système de traduction au projet de lunettes connectées. Le système implémenté par ces lunettes permet de localiser, de transcrire la voix d’un interlocuteur et d’afficher la transcription sur des lunettes connectées. 
        Dans ce projet, notre groupe implémentera un :red[**système de traduction**] qui élargira l’utilisation de ces lunettes à un public plus vaste et permettra à deux individus ne pratiquant pas la même langue de pouvoir communiquer aisément.
        Ce projet concentrera ses efforts sur l'implémentation d’un système de traduction plutôt que sur la reconnaissance vocale. Celle-ci nous sera fournie.

        Il nous faut prendre en considération quelques contraintes d’usages final, et voir si nous pourrons les respecter : 

        -	Traduction en temps réel d’un dialogue oral -> optimisation sur la rapidité
        -	Dialogue courant sans expertise particulière (champs sémantique généraliste)
        -	Prise en compte de la vitesse de lecture de chacun, la traduction doit être synthétique et conserver l’idée clé sans biais. (tout public et/ou design inclusif)

        Il est souhaitable que le système puisse rapidement :red[**identifier si les phrases fournies sont exprimées dans une des langues connues**] par le système de traduction, et si c’est le cas, :red[**laquelle**].
        De plus, si le système de reconnaissance vocale n’est pas fiable, il est souhaitable de corriger la phrase en fonction des mots environnants ou des phrases préalablement entendues.
        Lors de la traduction, nous prendrons en compte le contexte défini par la phrase précédente ainsi que par le contexte des phrases préalablement traduites. 
        Nous évaluerons la qualité de nos résultats en les comparant avec des systèmes performants tels que “[Google translate](https://translate.google.fr/)” et “[Deepl](https://www.deepl.com/translator)”.
        Enfin, si le temps, nos compétences et les datasets existants, le permettent, nous intégreront une langue originale, non proposée par ces systèmes, telle qu’une langue régionale ou de l’argot.

        Le projet est enregistré sur [Github](https://github.com/DataScientest-Studio/AVR23_CDS_Reco_vocale/tree/main)

        """
    )