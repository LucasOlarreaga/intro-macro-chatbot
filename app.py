from datetime import datetime, timedelta
from PIL import Image
import streamlit as st
import anthropic
import extra_streamlit_components as stx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_cookie_manager():
    return stx.CookieManager(key="global_cookie_manager")

# ---------------------------------------------------------------------------
# System prompt — edit this to adjust the assistant's personality/behaviour
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable and patient teaching assistant for an Introduction to Macroeconomics course at university level.

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
- Slides: Week 0 through Week 12 (including Friday sessions, which every few weeks adds content to the previous week's slides)
- Textbooks: Mankiw "Principles of Economics" (8th ed.), Krugman & Wells "Macroeconomics", and Mankiw "Macroeconomics"
- Problem Sets: PS1, PS2, PS3, PS4, PS5 (with solutions)
- Past Exams: August 2022, August 2024, February 2018, June 2019, June 2021, June 2023, May 2018 (all with answers)

IMPORTANT: All of the above documents ARE available to you via the retrieved context below. Never tell a student that a document is missing or that you don't have access to it — if the relevant content does not appear in the context, ask the student to be more specific so a better search can be performed.

Context retrieved from course materials:
{context}

Remember: your goal is to help the student learn, not to do the work for them."""

DIRECT_PROMPT = """You are a knowledgeable and patient teaching assistant for an Introduction to Macroeconomics course at university level.

Your core responsibilities:
- Help students understand course concepts through guided discovery.
- Use a Socratic approach: probe their existing understanding, ask guiding questions, and break problems into smaller steps so students reason through them on their own.
- When a student asks for a direct answer (e.g. to an exam or problem set question), do can give the solution outright. But, try to identify where they are stuck, and walk them forward step by step.
- ONLY use the course materials provided in the context below. If a question cannot be answered from those materials, say so clearly and suggest the student revisit the relevant lecture slides or textbook section.
- Use clear, accessible language suited to introductory-level university students.
- Be encouraging and supportive — learning economics can be challenging, and students benefit from positive reinforcement.
- When relevant, reference specific models, concepts, or examples from the course (e.g. AD-AS, fiscal multiplier, money supply, etc.).

The following course documents have been indexed and are available to you:
- Syllabus
- Slides: Week 0 through Week 12 (including Friday sessions, which every few weeks adds content to the previous week's slides)
- Textbooks: Mankiw "Principles of Economics" (8th ed.), Krugman & Wells "Macroeconomics", and Mankiw "Macroeconomics"
- Problem Sets: PS1, PS2, PS3, PS4, PS5 (with solutions)
- Past Exams: August 2022, August 2024, February 2018, June 2019, June 2021, June 2023, May 2018 (all with answers)

IMPORTANT: All of the above documents ARE available to you via the retrieved context below. Never tell a student that a document is missing or that you don't have access to it — if the relevant content does not appear in the context, ask the student to be more specific so a better search can be performed.

Context retrieved from course materials:
{context}"""

# ---------------------------------------------------------------------------
# Load vector store (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading course materials…")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )


DOC_KEYWORDS = [
    (["ps1", "ps 1", "problem set 1"],  lambda s: "ps1" in s.lower() or "ps 1" in s.lower()),
    (["ps2", "ps 2", "problem set 2"],  lambda s: "ps2" in s.lower() or "ps 2" in s.lower()),
    (["ps3", "ps 3", "problem set 3"],  lambda s: "ps3" in s.lower() or "ps 3" in s.lower()),
    (["ps4", "ps 4", "problem set 4"],  lambda s: "ps4" in s.lower() or "ps 4" in s.lower()),
    (["ps5", "ps 5", "problem set 5"],  lambda s: "ps5" in s.lower() or "ps 5" in s.lower()),
    (["week 0"],  lambda s: "week 0" in s.lower()),
    (["week 1"],  lambda s: "week 1" in s.lower() and not any(x in s.lower() for x in ["week 10","week 11","week 12"])),
    (["week 2"],  lambda s: "week 2" in s.lower()),
    (["week 3"],  lambda s: "week 3" in s.lower()),
    (["week 4"],  lambda s: "week 4" in s.lower()),
    (["week 5"],  lambda s: "week 5" in s.lower()),
    (["week 6"],  lambda s: "week 6" in s.lower()),
    (["week 7"],  lambda s: "week 7" in s.lower()),
    (["week 8"],  lambda s: "week 8" in s.lower()),
    (["week 9"],  lambda s: "week 9" in s.lower()),
    (["week 10"], lambda s: "week10" in s.lower() or "week 10" in s.lower()),
    (["week 11"], lambda s: "week11" in s.lower() or "week 11" in s.lower()),
    (["week 12"], lambda s: "week12" in s.lower() or "week 12" in s.lower()),
    (["exam", "past exam", "previous exam", "old exam"], lambda s: "exam" in s.lower()),
]


def get_doc_chunks(vectorstore, filter_fn, max_chunks: int = 20):
    """Directly scan the docstore and return chunks from matching documents."""
    matching = [
        doc for doc in vectorstore.docstore._dict.values()
        if filter_fn(doc.metadata.get("source", ""))
    ]
    # Sort by page number so questions appear in their natural order
    matching.sort(key=lambda d: d.metadata.get("page", 0))
    return matching[:max_chunks]


def retrieve_context(query: str, vectorstore, k: int = 12) -> str:
    query_lower = query.lower()

    # Detect specific document mentions and pull chunks directly from that doc
    targeted_docs = []
    for keywords, filter_fn in DOC_KEYWORDS:
        if any(kw in query_lower for kw in keywords):
            targeted_docs = get_doc_chunks(vectorstore, filter_fn, max_chunks=20)
            break

    # Always also do a broad semantic search
    broad_docs = vectorstore.similarity_search(query, k=k)

    # Merge: targeted first, then semantic, deduplicated
    seen = set()
    merged = []
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
# Password gate
# ---------------------------------------------------------------------------
def check_password():
    # Already authenticated this session — fastest path, no rendering needed
    if st.session_state.get("authenticated"):
        return True

    cookie_manager = get_cookie_manager()

    # CookieManager reads cookies via JS and needs one render cycle to load.
    # On the very first run, show a blank page and stop so the component can
    # initialize. It will automatically trigger a second render with values.
    if not st.session_state.get("cookie_check_done"):
        st.session_state["cookie_check_done"] = True
        st.stop()

    # Second render onwards: cookie values are available
    if cookie_manager.get("macro_auth") == st.secrets["COOKIE_TOKEN"]:
        st.session_state["authenticated"] = True
        st.rerun()
        return False

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

    acknowledged = st.checkbox(
        "I acknowledge that this assistant is a support tool that may generate inaccuracies, "
        "the verification of which remains my responsibility."
    )
    password = st.text_input("Enter the course password to continue:", type="password")

    if st.button("Access", disabled=not acknowledged):
        if password == st.secrets["APP_PASSWORD"]:
            # Set cookie valid for 7 days
            expires = datetime.now() + timedelta(days=7)
            cookie_manager.set(
                "macro_auth",
                st.secrets["COOKIE_TOKEN"],
                expires_at=expires,
                key="set_auth_cookie",
            )
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

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

    st.title("📚 Introduction to Macroeconomics")
    st.subheader("AI Teaching Assistant")
    st.caption(
        "Ask me anything covered in the course — lectures, textbook, problem sets, or past exams. "
        "I'll guide you through the material rather than just giving you the answer!"
    )
    st.divider()

    # Load resources
    try:
        vectorstore = load_vectorstore()
    except Exception:
        st.error(
            "⚠️ Vector store not found. Please run `python ingest.py` locally first "
            "and commit the `vectorstore/` folder to your repository."
        )
        st.stop()

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Sidebar — must be defined before chat input so direct_mode is available
    with st.sidebar:
        st.header("Options")
        st.divider()
        direct_mode = st.toggle(
            "Answer directly",
            value=False,
            help="Off: guides you toward the answer (default). On: gives the answer directly.",
        )
        if direct_mode:
            st.caption("💡 Direct mode — full answers provided.")
        else:
            st.caption("🎓 Guided mode — Socratic approach.")
        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        st.caption("Powered by Claude Haiku · Built with Streamlit")

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the course…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant context from the vector store
        context = retrieve_context(prompt, vectorstore)
        system = (DIRECT_PROMPT if direct_mode else SYSTEM_PROMPT).format(context=context)

        # Call Claude
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
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
