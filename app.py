from PIL import Image
import streamlit as st
import anthropic
from streamlit_cookies_controller import CookieController
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_cookie_controller():
    return CookieController()


# ---------------------------------------------------------------------------
# System prompts — English
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EN = """You are a knowledgeable and patient teaching assistant for an Introduction to Macroeconomics course at university level. Respond in English.

Your core responsibilities:
- Help students understand course concepts through guided discovery, not by handing them answers.
- Use a Socratic approach: probe their existing understanding, ask guiding questions, and break problems into smaller steps so students reason through them on their own.
- When a student asks for a direct answer (e.g. to an exam or problem set question), do NOT give the solution outright. Instead, identify where they are stuck, offer a relevant hint or a leading question, and walk them forward step by step.
- ONLY use the course materials provided in the context below. If a question cannot be answered from those materials, say so clearly and suggest the student revisit the relevant lecture slides or textbook section.
- Use clear, accessible language suited to introductory-level university students.
- Be encouraging and supportive — learning economics can be challenging, and students benefit from positive reinforcement.
- When relevant, reference specific models, concepts, or examples from the course (e.g. AD-AS, fiscal multiplier, money supply, etc.).

The following course documents have been indexed and are available to you:
- Syllabus
- Slides: Week 0 through Week 12 (including Friday sessions)
- Textbooks: Mankiw "Principles of Economics" (8th ed.), Krugman & Wells "Macroeconomics", and Mankiw "Macroeconomics"
- Problem Sets: PS1, PS2, PS3, PS4, PS5 (with solutions)
- Past Exams: August 2022, August 2024, February 2018, June 2019, June 2021, June 2023, May 2018 (all with answers)

IMPORTANT: All of the above documents ARE available to you via the retrieved context below. Never tell a student that a document is missing — if the relevant content does not appear in the context, ask the student to be more specific.

Context retrieved from course materials:
{context}

Remember: your goal is to help the student learn, not to do the work for them."""


DIRECT_PROMPT_EN = """You are a knowledgeable teaching assistant for an Introduction to Macroeconomics course at university level. Respond in English.

Your role is to give clear, complete, and accurate answers to student questions.
- Answer directly and fully — do not withhold information or use the Socratic method.
- Base your answers ONLY on the course materials provided in the context below.
- Be concise but thorough. Use bullet points, equations, or examples where helpful.
- If a question cannot be answered from the course materials, say so clearly.

The following course documents have been indexed and are available to you:
- Syllabus
- Slides: Week 0 through Week 12 (including Friday sessions)
- Textbooks: Mankiw "Principles of Economics" (8th ed.), Krugman & Wells "Macroeconomics", and Mankiw "Macroeconomics"
- Problem Sets: PS1, PS2, PS3, PS4, PS5 (with solutions)
- Past Exams: August 2022, August 2024, February 2018, June 2019, June 2021, June 2023, May 2018 (all with answers)

Context retrieved from course materials:
{context}"""


# ---------------------------------------------------------------------------
# System prompts — French
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_FR = """Tu es un assistant pédagogique compétent et patient pour un cours d'Introduction à la Macroéconomie de niveau universitaire. Réponds toujours en français.

Tes responsabilités principales :
- Aider les étudiants à comprendre les concepts du cours par la découverte guidée, et non en leur donnant directement les réponses.
- Adopter une approche socratique : sonder leur compréhension existante, poser des questions directrices et décomposer les problèmes en étapes plus petites afin que les étudiants raisonnent par eux-mêmes.
- Lorsqu'un étudiant demande une réponse directe (par exemple à une question d'examen ou de série d'exercices), NE PAS donner la solution immédiatement. Identifier où il est bloqué, offrir un indice ou une question directrice, et le guider étape par étape.
- Utiliser UNIQUEMENT les documents du cours fournis dans le contexte ci-dessous. Si une question ne peut pas être répondue à partir de ces documents, le dire clairement et suggérer à l'étudiant de revoir les slides ou le manuel correspondant.
- Utiliser un langage clair et accessible, adapté aux étudiants de niveau introductoire.
- Être encourageant et bienveillant — apprendre l'économie peut être difficile, et les étudiants bénéficient d'un renforcement positif.
- Lorsque c'est pertinent, faire référence aux modèles, concepts ou exemples spécifiques du cours (ex. : OA-DA, multiplicateur budgétaire, masse monétaire, taux de change, etc.).

Les documents suivants ont été indexés et sont disponibles :
- Syllabus
- Slides : Chapitres 1 à 14
- Manuels : Mankiw & Taylor (MT) et Krugman & Wells (KW)
- Séries d'exercices (avec solutions)
- Examens passés (avec corrigés)

IMPORTANT : Tous ces documents SONT disponibles via le contexte récupéré ci-dessous. Ne jamais dire à un étudiant qu'un document est manquant — si le contenu pertinent n'apparaît pas dans le contexte, lui demander d'être plus précis.

Contexte récupéré depuis les documents du cours :
{context}

Rappel : ton objectif est d'aider l'étudiant à apprendre, pas de faire le travail à sa place."""


DIRECT_PROMPT_FR = """Tu es un assistant pédagogique compétent pour un cours d'Introduction à la Macroéconomie de niveau universitaire. Réponds toujours en français.

Ton rôle est de donner des réponses claires, complètes et précises aux questions des étudiants.
- Répondre directement et complètement — ne pas retenir d'informations ni utiliser la méthode socratique.
- Baser tes réponses UNIQUEMENT sur les documents du cours fournis dans le contexte ci-dessous.
- Être concis mais complet. Utiliser des listes, des équations ou des exemples si utile.
- Si une question ne peut pas être répondue à partir des documents du cours, le dire clairement.

Les documents suivants ont été indexés et sont disponibles :
- Syllabus
- Slides : Chapitres 1 à 14
- Manuels : Mankiw & Taylor (MT) et Krugman & Wells (KW)
- Séries d'exercices (avec solutions)
- Examens passés (avec corrigés)

Contexte récupéré depuis les documents du cours :
{context}"""


# ---------------------------------------------------------------------------
# Document keyword maps
# ---------------------------------------------------------------------------

# Shared helpers for query-side synonyms
def _ps_kw(n):
    """All ways a student might refer to problem set N (English)."""
    return [
        f"ps{n}", f"ps {n}", f"problem set {n}", f"pset{n}", f"pset {n}",
        f"exercise set {n}", f"exercises {n}", f"assignment {n}",
        f"hw{n}", f"hw {n}", f"homework {n}",
    ]

def _tp_kw(n):
    """All ways a student might refer to TP N (French)."""
    return [
        f"tp{n}", f"tp {n}", f"ps{n}", f"ps {n}",
        f"série {n}", f"serie {n}", f"exercices {n}",
        f"travaux pratiques {n}", f"feuille {n}", f"td{n}", f"td {n}",
    ]

def _week_kw(n, *topics):
    """All ways a student might refer to week N slides (English)."""
    base = [
        f"week {n}", f"wk {n}", f"wk{n}",
        f"lecture {n}", f"class {n}",
        f"slides week {n}", f"slide week {n}",
        f"presentation week {n}", f"powerpoint week {n}", f"ppt week {n}",
        f"notes week {n}", f"week {n} slides", f"week {n} lecture",
        f"week {n} notes", f"week {n} ppt", f"week {n} presentation",
    ]
    return base + list(topics)

def _ch_kw(n, *topics):
    """All ways a student might refer to chapter N slides (French)."""
    base = [
        f"chapitre {n}", f"ch. {n}", f"ch {n}", f"ch.{n}", f"chap {n}", f"chap{n}",
        f"cours {n}", f"séance {n}", f"seance {n}",
        f"slides chapitre {n}", f"diapos chapitre {n}", f"présentation chapitre {n}",
        f"powerpoint chapitre {n}", f"ppt chapitre {n}",
    ]
    return base + list(topics)

EN_DOC_KEYWORDS = [
    # --- Problem sets ---
    (_ps_kw(1), lambda s: "ps 1" in s.lower() or "ps1" in s.lower()),
    (_ps_kw(2), lambda s: "ps 2" in s.lower() or "ps2" in s.lower()),
    (_ps_kw(3), lambda s: "ps 3" in s.lower() or "ps3" in s.lower()),
    (_ps_kw(4), lambda s: "ps 4" in s.lower() or "ps4" in s.lower()),
    (_ps_kw(5), lambda s: "ps 5" in s.lower() or "ps5" in s.lower()),

    # --- Slides / lectures by week (with topic name synonyms) ---
    (_week_kw(0,  "introduction", "overview", "course intro", "what is macroeconomics"),
     lambda s: "week 0" in s.lower()),
    (_week_kw(1,  "gdp", "gross domestic product", "national accounts", "output", "measuring output"),
     lambda s: "week 1" in s.lower() and not any(x in s.lower() for x in ["week 10","week 11","week 12"])),
    (_week_kw(2,  "inflation", "cpi", "consumer price index", "price level", "measuring prices"),
     lambda s: "week 2" in s.lower()),
    (_week_kw(3,  "unemployment", "labour market", "labor market", "jobless"),
     lambda s: "week 3" in s.lower()),
    (_week_kw(4,  "savings", "investment", "financial market", "loanable funds", "closed economy equilibrium"),
     lambda s: "week 4" in s.lower()),
    (_week_kw(5,  "money", "monetary system", "money supply", "central bank", "banking"),
     lambda s: "week 5" in s.lower()),
    (_week_kw(6,  "monetary growth", "quantity theory", "seigniorage", "hyperinflation"),
     lambda s: "week 6" in s.lower()),
    (_week_kw(7,  "open economy", "trade balance", "current account", "capital flows", "net exports"),
     lambda s: "week 7" in s.lower()),
    (_week_kw(8,  "exchange rate", "forex", "purchasing power parity", "ppp", "nominal exchange rate"),
     lambda s: "week 8" in s.lower()),
    (_week_kw(9,  "open economy equilibrium", "mundell-fleming", "small open economy"),
     lambda s: "week 9" in s.lower()),
    (_week_kw(10, "aggregate demand", "aggregate supply", "ad-as", "ad as", "short run fluctuations"),
     lambda s: "week10" in s.lower() or "week 10" in s.lower()),
    (_week_kw(11, "fiscal policy", "monetary policy", "multiplier", "government spending", "taxes", "interest rates"),
     lambda s: "week11" in s.lower() or "week 11" in s.lower()),
    (_week_kw(12, "phillips curve", "inflation unemployment tradeoff", "short run phillips", "long run phillips", "great recession", "lockdown"),
     lambda s: "week12" in s.lower() or "week 12" in s.lower()),

    # --- Exams ---
    (["exam", "exams", "past exam", "previous exam", "old exam",
      "past test", "previous test", "practice exam", "mock exam",
      "midterm", "final exam", "sample exam"],
     lambda s: "exam" in s.lower()),
]

FR_DOC_KEYWORDS = [
    # --- Séries d'exercices / TP ---
    (_tp_kw(1), lambda s: "tp1" in s.lower() or "tp 1" in s.lower()),
    (_tp_kw(2), lambda s: "tp2" in s.lower() or "tp 2" in s.lower()),
    (_tp_kw(3), lambda s: "tp3" in s.lower() or "tp 3" in s.lower()),
    (_tp_kw(4), lambda s: "tp4" in s.lower() or "tp 4" in s.lower()),
    (_tp_kw(5), lambda s: "tp5" in s.lower() or "tp 5" in s.lower()),

    # --- Slides / cours par chapitre ---
    (_ch_kw(1,  "introduction", "approche macroéconomique"),
     lambda s: any(x in s.lower() for x in ["01_", "ch. 1", "ch.1", "chapitre 1", "chap 1"])),
    (_ch_kw(2,  "pib", "produit intérieur brut", "gdp"),
     lambda s: any(x in s.lower() for x in ["02_", "ch. 2", "ch.2", "chapitre 2", "chap 2"])),
    (_ch_kw(3,  "ipc", "indice des prix", "inflation des prix", "cpi"),
     lambda s: any(x in s.lower() for x in ["03_", "ch. 3", "ch.3", "chapitre 3", "chap 3"])),
    (_ch_kw(4,  "chômage", "chomage", "unemployment"),
     lambda s: any(x in s.lower() for x in ["04_", "ch. 4", "ch.4", "chapitre 4", "chap 4"])),
    (_ch_kw(5,  "épargne", "epargne", "investissement", "équilibre économie fermée", "marché financier"),
     lambda s: any(x in s.lower() for x in ["05_", "ch. 5", "ch.5", "chapitre 5", "chap 5"])),
    (_ch_kw(6,  "monnaie", "système monétaire", "systeme monetaire", "banque centrale"),
     lambda s: any(x in s.lower() for x in ["06_", "ch. 6", "ch.6", "chapitre 6", "chap 6"])),
    (_ch_kw(7,  "inflation", "croissance monétaire", "théorie quantitative"),
     lambda s: any(x in s.lower() for x in ["07_", "ch. 7", "ch.7", "chapitre 7", "chap 7"])),
    (_ch_kw(8,  "économie ouverte", "concepts de base", "balance commerciale", "flux de capitaux"),
     lambda s: any(x in s.lower() for x in ["08_", "ch. 8", "ch.8", "chapitre 8", "chap 8"])),
    (_ch_kw(9,  "taux de change", "change", "forex"),
     lambda s: any(x in s.lower() for x in ["09_", "ch. 9", "ch.9", "chapitre 9", "chap 9"])),
    (_ch_kw(10, "modèle macroéconomique", "equilibre économie ouverte"),
     lambda s: any(x in s.lower() for x in ["10_", "ch. 10", "ch.10", "chapitre 10", "chap 10"])),
    (_ch_kw(11, "fluctuations", "offre agrégée", "demande agrégée", "da-oa", "oa-da", "ad-as"),
     lambda s: any(x in s.lower() for x in ["11_", "ch. 11", "ch.11", "chapitre 11", "chap 11"])),
    (_ch_kw(12, "politique monétaire", "politique fiscale", "politique budgétaire", "multiplicateur"),
     lambda s: any(x in s.lower() for x in ["12_", "ch. 12", "ch.12", "chapitre 12", "chap 12"])),
    (_ch_kw(13, "arbitrage", "courbe de phillips", "phillips", "chômage et inflation"),
     lambda s: any(x in s.lower() for x in ["13_", "ch. 13", "ch.13", "chapitre 13", "chap 13"])),
    (_ch_kw(14, "grande récession", "grande recession", "lockdown", "crise", "covid"),
     lambda s: any(x in s.lower() for x in ["14_", "ch. 14", "ch.14", "chapitre 14", "chap 14"])),

    # --- Examens ---
    (["examen", "examens", "examen passé", "ancien examen", "examen précédent",
      "corrigé", "corrige", "annales", "exam", "épreuve"],
     lambda s: "exam" in s.lower()),
]


# ---------------------------------------------------------------------------
# Vector store — one per language, cached separately
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading course materials…")
def load_vectorstore(lang: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(
        f"vectorstore_{lang}", embeddings, allow_dangerous_deserialization=True
    )


def get_doc_chunks(vectorstore, filter_fn, max_chunks: int = 20):
    matching = [
        doc for doc in vectorstore.docstore._dict.values()
        if filter_fn(doc.metadata.get("source", ""))
    ]
    matching.sort(key=lambda d: d.metadata.get("page", 0))
    return matching[:max_chunks]


def retrieve_context(query: str, vectorstore, lang: str, k: int = 12) -> str:
    query_lower = query.lower()
    keywords_map = EN_DOC_KEYWORDS if lang == "en" else FR_DOC_KEYWORDS

    targeted_docs = []
    for keywords, filter_fn in keywords_map:
        if any(kw in query_lower for kw in keywords):
            targeted_docs = get_doc_chunks(vectorstore, filter_fn, max_chunks=20)
            break

    broad_docs = vectorstore.similarity_search(query, k=k)

    seen, merged = set(), []
    for doc in targeted_docs + broad_docs:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    chunks = []
    for doc in merged:
        source = doc.metadata.get("source", "Unknown document")
        chunks.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Password / language gate
# ---------------------------------------------------------------------------
def check_password():
    if st.session_state.get("authenticated"):
        return True

    controller = get_cookie_controller()

    if not st.session_state.get("cookie_check_done"):
        st.session_state["cookie_check_done"] = True
        st.stop()

    if controller.get("macro_auth") == st.secrets["COOKIE_TOKEN"]:
        st.session_state["authenticated"] = True
        st.rerun()
        return False

    # --- Language selector ---
    lang = st.radio(
        "Language / Langue",
        options=["en", "fr"],
        format_func=lambda x: "English" if x == "en" else "Français",
        horizontal=True,
        index=0,
    )
    st.session_state["lang"] = lang

    st.divider()

    if lang == "en":
        st.title("AI Teaching Assistant — Introduction to Macroeconomics")
        st.subheader("Guiding Responsible AI in Teaching")
        st.caption("The AI pedagogical companion of GSEM faculty.")
        st.divider()
        st.markdown(
            """
            This platform supports you in integrating Generative AI into your teaching and assessment
            practices, in alignment with the guidelines validated by the **GSEM AI Taskforce**.
            It is designed to help you:

            - Strategically incorporate AI into course design
            - Develop academically rigorous learning activities
            - Rethink assessment methods in the AI era
            - Ensure alignment with institutional standards and academic integrity principles

            ---

            #### Responsible Use

            This assistant provides structured guidance based exclusively on validated institutional
            documents. It does not replace academic judgment or institutional policy.
            Faculty members remain solely responsible for:

            - Final pedagogical decisions
            - The validation of generated content
            - Compliance with academic and data protection standards

            > **No confidential or student-sensitive data should be entered into the system.**

            ---
            *Powered by GSEM*
            """
        )
        ack_label = "I acknowledge that this assistant is a support tool that may generate inaccuracies, the verification of which remains my responsibility."
        pw_label  = "Enter the course password to continue:"
        btn_label = "Access"
    else:
        st.title("Assistant Pédagogique IA — Introduction à la Macroéconomie")
        st.subheader("Pour un usage responsable de l'IA dans l'enseignement")
        st.caption("L'assistant pédagogique IA de la faculté GSEM.")
        st.divider()
        st.markdown(
            """
            Cette plateforme vous accompagne dans l'intégration de l'IA générative dans vos pratiques
            d'enseignement et d'évaluation, en accord avec les lignes directrices validées par le
            **groupe de travail IA de la GSEM**. Elle est conçue pour vous aider à :

            - Intégrer stratégiquement l'IA dans la conception des cours
            - Développer des activités d'apprentissage académiquement rigoureuses
            - Repenser les méthodes d'évaluation à l'ère de l'IA
            - Assurer l'alignement avec les normes institutionnelles et les principes d'intégrité académique

            ---

            #### Utilisation responsable

            Cet assistant fournit des conseils structurés basés exclusivement sur des documents
            institutionnels validés. Il ne remplace pas le jugement académique ni la politique institutionnelle.
            Les membres du corps enseignant restent seuls responsables de :

            - Les décisions pédagogiques finales
            - La validation du contenu généré
            - La conformité aux normes académiques et de protection des données

            > **Aucune donnée confidentielle ou sensible concernant les étudiants ne doit être saisie dans le système.**

            ---
            *Propulsé par GSEM*
            """
        )
        ack_label = "Je reconnais que cet assistant est un outil d'aide susceptible de générer des inexactitudes, dont la vérification reste de ma responsabilité."
        pw_label  = "Entrez le mot de passe du cours pour continuer :"
        btn_label = "Accéder"

    acknowledged = st.checkbox(ack_label)
    password     = st.text_input(pw_label, type="password")

    if st.button(btn_label, disabled=not acknowledged):
        if password == st.secrets["APP_PASSWORD"]:
            controller.set("macro_auth", st.secrets["COOKIE_TOKEN"], max_age=7 * 24 * 60 * 60)
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            err = "Incorrect password." if lang == "en" else "Mot de passe incorrect."
            st.error(err)

    return False


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Macro101 — AI Teaching Assistant",
        page_icon=Image.open("GSEM logo.jpg"),
        layout="centered",
    )

    if not check_password():
        st.stop()

    # Default language to EN if not set (cookie bypass skips the picker)
    lang = st.session_state.get("lang", "en")

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Sidebar — defined before chat so all toggles are available
    with st.sidebar:
        st.header("Options" if lang == "en" else "Options")
        st.divider()

        # Language toggle
        new_lang = st.radio(
            "Language / Langue",
            options=["en", "fr"],
            format_func=lambda x: "English" if x == "en" else "Français",
            index=0 if lang == "en" else 1,
            horizontal=True,
        )
        if new_lang != lang:
            st.session_state["lang"] = new_lang
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # Direct answer toggle
        direct_label = "Answer directly" if lang == "en" else "Répondre directement"
        direct_help   = (
            "Off: guides you toward the answer. On: gives the answer directly."
            if lang == "en" else
            "Désactivé : vous guide vers la réponse. Activé : donne la réponse directement."
        )
        direct_mode = st.toggle(direct_label, value=False, help=direct_help)
        if direct_mode:
            st.caption("💡 Direct mode." if lang == "en" else "💡 Mode direct.")
        else:
            st.caption("🎓 Guided mode." if lang == "en" else "🎓 Mode guidé.")

        st.divider()
        clear_label = "🗑️ Clear conversation" if lang == "en" else "🗑️ Effacer la conversation"
        if st.button(clear_label):
            st.session_state.messages = []
            st.rerun()
        st.caption("Powered by Claude Haiku · Built with Streamlit")

    # Page header
    if lang == "en":
        st.title("📚 Introduction to Macroeconomics")
        st.subheader("AI Teaching Assistant")
        st.caption("Ask me anything covered in the course — lectures, textbook, problem sets, or past exams.")
    else:
        st.title("📚 Introduction à la Macroéconomie")
        st.subheader("Assistant Pédagogique IA")
        st.caption("Posez-moi n'importe quelle question sur le cours — slides, manuel, séries d'exercices ou examens passés.")
    st.divider()

    # Load vectorstore for current language
    try:
        vectorstore = load_vectorstore(lang)
    except Exception:
        msg = (
            f"⚠️ Vector store for '{lang}' not found. Run `python ingest.py --lang {lang}` locally and commit `vectorstore_{lang}/`."
        )
        st.error(msg)
        st.stop()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = "Ask a question about the course…" if lang == "en" else "Posez une question sur le cours…"
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = retrieve_context(prompt, vectorstore, lang)
        if lang == "en":
            system = (DIRECT_PROMPT_EN if direct_mode else SYSTEM_PROMPT_EN).format(context=context)
        else:
            system = (DIRECT_PROMPT_FR if direct_mode else SYSTEM_PROMPT_FR).format(context=context)

        spinner_msg = "Thinking…" if lang == "en" else "Réflexion en cours…"
        with st.chat_message("assistant"):
            with st.spinner(spinner_msg):
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                )
                reply = response.content[0].text
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
