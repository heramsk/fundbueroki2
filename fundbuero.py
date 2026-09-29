import streamlit as st
import tensorflow as tf
import numpy as np

# =====================================
# SEITENEINSTELLUNGEN
# =====================================

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🔎",
    layout="centered"
)

# =====================================
# DESIGN
# =====================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #eef3f8 100%);
    font-family: "Segoe UI", Arial, sans-serif;
}

.block-container {
    width: 100%;
    max-width: 1050px;
    padding: 25px 5%;
}

h1, h2, h3, p, label {
    color: #1e293b !important;
}

.title {
    text-align: center;
    font-family: Georgia, serif;
    font-size: clamp(32px, 6vw, 50px);
    font-weight: bold;
    letter-spacing: 5px;
    color: #203a5f;
    margin-bottom: 35px;
}

.page-title {
    font-family: Georgia, serif;
    font-size: clamp(26px, 5vw, 36px);
    font-weight: bold;
    color: #203a5f;
    margin-bottom: 25px;
}

.fund-card {
    box-sizing: border-box;
    width: 100%;
    background: white;
    border: 1px solid #d5e0ec;
    border-left: 6px solid #7696bd;
    border-radius: 18px;
    padding: 20px;
    margin: 18px 0;
    box-shadow: 0 6px 22px rgba(30, 50, 80, 0.08);
    overflow-wrap: anywhere;
}

.fund-name {
    font-size: clamp(19px, 4vw, 25px);
    font-weight: bold;
    color: #203a5f;
    margin-bottom: 10px;
}

.fund-info {
    color: #475569;
    font-size: 15px;
    line-height: 1.8;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border: 1px solid #c5d4e5;
    border-radius: 14px;
    background: white;
    color: #203a5f;
    font-weight: 600;
    box-shadow: 0 3px 10px rgba(30, 50, 80, 0.06);
}

.stButton > button:hover {
    background: #e8f0f8;
    border-color: #7696bd;
}

.stTextInput input,
.stTextArea textarea {
    border: 1px solid #cbd8e7;
    border-radius: 12px;
    background: white;
    color: #1e293b;
}

.stSelectbox div[data-baseweb="select"] > div {
    border: 1px solid #cbd8e7;
    border-radius: 12px;
    background: white;
}

section[data-testid="stFileUploader"] {
    border: 2px dashed #b8c9dd;
    border-radius: 15px;
    padding: 12px;
    background: #f8fbff;
}

.bottom-space {
    height: 45px;
}

@media (max-width: 600px) {
    .block-container {
        padding: 15px 12px;
    }

    .fund-card {
        border-left-width: 4px;
        border-radius: 14px;
        padding: 14px;
    }

    .stButton > button {
        min-height: 44px;
        font-size: 10px;
        padding: 4px 2px;
    }
}

section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# KI LADEN
# =====================================

@st.cache_resource
def lade_modell():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

@st.cache_data
def lade_labels():
    with open("labels.txt", "r", encoding="utf-8") as datei:
        return [zeile.strip() for zeile in datei.readlines()]

modell = lade_modell()
labels = lade_labels()

# =====================================
# FUNDSACHEN SPEICHERN
# =====================================

if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = [
        {
            "name": "Blaue Wasserflasche",
            "kategorie": "Flasche",
            "ort": "Schule",
            "beschreibung": "Blaue Wasserflasche wurde gefunden.",
            "bild": None
        },
        {
            "name": "Grauer Pullover",
            "kategorie": "Pulli",
            "ort": "Sporthalle",
            "beschreibung": "Grauer Pullover mit Kapuze.",
            "bild": None
        },
        {
            "name": "Schwarze Hose",
            "kategorie": "Hose",
            "ort": "Pausenhof",
            "beschreibung": "Schwarze Hose wurde gefunden.",
            "bild": None
        }
    ]

if "seite" not in st.session_state:
    st.session_state.seite = "HOME"

# =====================================
# KI-KATEGORIE ERKENNEN
# =====================================

def erkenne_kategorie(bild):
    bild = tf.keras.utils.load_img(
        bild,
        target_size=(224, 224)
    )

    bild_array = tf.keras.utils.img_to_array(bild)
    bild_array = np.asarray(bild_array, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1
    bild_array = np.expand_dims(bild_array, axis=0)

    vorhersage = modell.predict(bild_array, verbose=0)
    index = int(np.argmax(vorhersage[0]))
    sicherheit = float(vorhersage[0][index])

    label = labels[index]

    # Nummer am Anfang entfernen:
    # Beispiel: "0 Pulli" wird zu "Pulli"
    kategorie = label.split(" ", 1)[-1]

    return kategorie, sicherheit

# =====================================
# FUNDKARTE
# =====================================

def fundkarte_anzeigen(fund):
    st.markdown('<div class="fund-card">', unsafe_allow_html=True)

    if fund["bild"] is not None:
        st.image(fund["bild"], use_container_width=True)

    st.markdown(
        f'<div class="fund-name">{fund["name"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="fund-info">
        📦 <b>Kategorie:</b> {fund["kategorie"]}<br>
        📍 <b>Fundort:</b> {fund["ort"]}<br>
        📝 <b>Beschreibung:</b> {fund["beschreibung"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# HOMEPAGE
# =====================================

def homepage():
    st.markdown(
        '<div class="title">FUNDBÜRO</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Neu gefunden in Ihrer Nähe")

    for fund in reversed(st.session_state.fundstuecke):
        fundkarte_anzeigen(fund)

# =====================================
# SUCHE
# =====================================

def suche():
    st.markdown(
        '<div class="page-title">🔎 Suche</div>',
        unsafe_allow_html=True
    )

    suchtext = st.text_input(
        "Wonach suchst du?",
        placeholder="z. B. Flasche, Pulli oder Sporthalle"
    )

    if suchtext.strip():
        treffer = []

        for fund in st.session_state.fundstuecke:
            suchinhalt = (
                fund["name"] + " " +
                fund["kategorie"] + " " +
                fund["ort"] + " " +
                fund["beschreibung"]
            ).lower()

            if suchtext.lower() in suchinhalt:
                treffer.append(fund)

        if treffer:
            for fund in reversed(treffer):
                fundkarte_anzeigen(fund)
        else:
            st.warning("Keine passenden Fundstücke gefunden.")

# =====================================
# FUND EINSTELLEN
# =====================================

def fund_einstellen():
    st.markdown(
        '<div class="page-title">📦 Fund einstellen</div>',
        unsafe_allow_html=True
    )

    name = st.text_input(
        "Name des Gegenstands",
        placeholder="z. B. Gefundener Gegenstand"
    )

    ort = st.text_input(
        "Fundort",
        placeholder="z. B. Pausenhof oder Sporthalle"
    )

    beschreibung = st.text_area(
        "Beschreibung",
        placeholder="Weitere Informationen zum Fundstück ..."
    )

    bild = st.file_uploader(
        "Bild hochladen – die KI erkennt die Kategorie",
        type=["png", "jpg", "jpeg"]
    )

    erkannte_kategorie = None

    if bild is not None:
        st.image(bild, caption="Hochgeladenes Bild", use_container_width=True)

        with st.spinner("Die KI analysiert das Bild ..."):
            try:
                erkannte_kategorie, sicherheit = erkenne_kategorie(bild)

                st.success(
                    f"Erkannte Kategorie: {erkannte_kategorie} "
                    f"({sicherheit * 100:.1f}% Sicherheit)"
                )

                if sicherheit < 0.60:
                    st.warning(
                        "Die KI ist sich bei dieser Erkennung nicht ganz sicher."
                    )

            except Exception as fehler:
                st.error(
                    "Das Bild konnte nicht analysiert werden. "
                    "Überprüfe dein Modell und die Bildgröße."
                )
                st.code(str(fehler))

    if st.button("✨ Fundstück veröffentlichen"):
        if not name.strip() or not ort.strip():
            st.error("Bitte gib mindestens den Namen und den Fundort ein.")

        elif bild is None:
            st.error("Bitte lade zuerst ein Bild hoch.")

        elif erkannte_kategorie is None:
            st.error("Die Kategorie konnte nicht erkannt werden.")

        else:
            neues_fundstueck = {
                "name": name,
                "kategorie": erkannte_kategorie,
                "ort": ort,
                "beschreibung": beschreibung,
                "bild": bild.getvalue()
            }

            st.session_state.fundstuecke.append(neues_fundstueck)
            st.session_state.seite = "HOME"

            st.success("Fundstück erfolgreich veröffentlicht!")
            st.rerun()

# =====================================
# NACHRICHTEN
# =====================================

def nachrichten():
    st.markdown(
        '<div class="page-title">💬 Nachrichten</div>',
        unsafe_allow_html=True
    )

    st.info("Hier können später Nachrichten angezeigt werden.")

# =====================================
# PROFIL
# =====================================

def profil():
    st.markdown(
        '<div class="page-title">👤 Profil</div>',
        unsafe_allow_html=True
    )

    st.info("Hier können später Profildaten eingestellt werden.")

# =====================================
# SEITENSTEUERUNG
# =====================================

if st.session_state.seite == "HOME":
    homepage()

elif st.session_state.seite == "SUCHE":
    suche()

elif st.session_state.seite == "FUND EINSTELLEN":
    fund_einstellen()

elif st.session_state.seite == "NACHRICHTEN":
    nachrichten()

elif st.session_state.seite == "PROFIL":
    profil()

# =====================================
# UNTERE NAVIGATION
# =====================================

st.markdown('<div class="bottom-space"></div>', unsafe_allow_html=True)

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("⌂ HOME", use_container_width=True):
        st.session_state.seite = "HOME"
        st.rerun()

with col2:
    if st.button("🔎 SUCHE", use_container_width=True):
        st.session_state.seite = "SUCHE"
        st.rerun()

with col3:
    if st.button("＋ FUND", use_container_width=True):
        st.session_state.seite = "FUND EINSTELLEN"
        st.rerun()

with col4:
    if st.button("💬 CHAT", use_container_width=True):
        st.session_state.seite = "NACHRICHTEN"
        st.rerun()

with col5:
    if st.button("👤 PROFIL", use_container_width=True):
        st.session_state.seite = "PROFIL"
        st.rerun()
