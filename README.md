![Alt Whaooh!](./streamlit_app/assets/miss-honey-glasses-off.gif)

## Introduction
This repository contains the code for our project **TRANSLATION SYSTEM FOR CONNECTED GLASSES**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The purpose of this project is to explore and implement exploration techniques.

This project is developed by the following team :
- Keyne Dupont ([GitHub](https://github.com/) / [LinkedIn](https://www.linkedin.com/in/keyne-dupont/))
- Tia Ratsimbason ([GitHub](https://github.com/) / [LinkedIn](https://www.linkedin.com/in/tia-ratsimbason-42110887/))
- Olivier Renouard ([GitHub](https://github.com/Demosthene-OR) / [LinkedIn](https://www.linkedin.com/in/olivier-renouard-b6b8a535/))

## Notebooks
You can browse and run the **[notebooks](./notebooks)**. 

To run the notebooks, you will need to install the dependencies (in a dedicated environment)
```
pip install -r requirements.txt
```

## Streamlit App
You can also run the **Streamlit App directly on the Cloud**: 

**[Translation System for Connected Glasses](https://demosthene-or-avr23-cds-translation.hf.space/)**

To run the **app locally** (be careful with the paths of the files in the app):
```shell
conda create --name avr23-cds-translation python=3.9
conda activate avr23-cds-translation
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
