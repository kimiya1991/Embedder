from __future__ import annotations

import pandas as pd
import streamlit as st

from lab import (
    METRIC_GUIDE,
    all_embedders,
    compare_texts,
    evaluate_embedder,
    get_embedder,
    install_sentence_transformers,
    load_dataset,
    persist_and_apply,
    public_key_status,
    reset_embedders,
    search_chunks,
)

st.set_page_config(page_title="Embedder Lab", layout="wide")
st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow-y: auto !important;
      }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #f6f7f8 !important;
        color: #1c2430 !important;
      }
      [data-testid="stHeader"] { border-bottom: 1px solid #dde2e8; }
      section[data-testid="stSidebar"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        background: #e8edf1 !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow: visible !important;
        background: #e8edf1 !important;
        color: #1c2430 !important;
      }
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] h1 {
        color: #1c2430 !important;
      }
      [data-testid="stExpander"] {
        background: #fff !important;
        border: 1px solid #d5dde5 !important;
        border-radius: 10px !important;
      }
      .stButton > button {
        background: #1f4e5f !important;
        color: #fff !important;
        border: 0 !important;
        border-radius: 8px !important;
      }
      .stButton > button:hover { background: #173d4a !important; color: #fff !important; }
      h1, h2, h3 { font-family: Georgia, serif; color: #1c2430 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_SELECTED = ["hash-baseline", "minilm", "bge-small"]


def configured_embedders():
    return [item for item in all_embedders() if item.is_configured()]


def pick_embedder(label: str, key: str) -> str | None:
    ready = configured_embedders()
    if not ready:
        st.warning("No embedder is ready. Open API keys if you want a cloud model.")
        return None
    names = [item.name for item in ready]
    ids = {item.name: item.id for item in ready}
    chosen = st.pills(label, names, selection_mode="single", default=names[0], key=key)
    return ids[chosen] if chosen else ids[names[0]]


data = load_dataset()
catalog = all_embedders()

for item in catalog:
    box_key = f"pick-{item.id}"
    if box_key not in st.session_state:
        st.session_state[box_key] = item.id in DEFAULT_SELECTED and item.is_configured()

st.sidebar.markdown("**Retrieval lab**")
st.sidebar.title("Embedder test bench")
st.sidebar.caption(
    "Check models to benchmark. Cloud models need a key: open the **API keys** tab, paste, Save keys, then check the model."
)

for item in catalog:
    status = "ready" if item.is_configured() else "need key"
    with st.sidebar.expander(f"{item.name} · {status}", expanded=False):
        st.checkbox("Use in benchmark", disabled=not item.is_configured(), key=f"pick-{item.id}")
        st.write(item.explanation)
        if item.env_var:
            st.caption(f"Needs `{item.env_var}`" + ("" if item.is_configured() else " — not set yet."))
        if not item.is_configured() and item.id == "mpnet":
            st.caption("Not installed yet. This downloads CPU torch and can take several minutes.")
            if st.button("Install sentence-transformers", key="install-mpnet"):
                with st.spinner("Installing sentence-transformers and CPU torch…"):
                    ok, log = install_sentence_transformers()
                if ok:
                    st.success("Installed. MPNet is ready — check Use in benchmark.")
                    st.rerun()
                else:
                    st.error("Install failed.")
                    st.code(log)
        elif not item.is_configured():
            st.caption(item.missing_reason())

selected_ids = [item.id for item in catalog if st.session_state.get(f"pick-{item.id}")]
tab_bench, tab_search, tab_embed, tab_metrics, tab_doc, tab_keys = st.tabs(
    ["Benchmark", "Search file", "Use embedder", "Metrics", "Sample file", "API keys"]
)

with tab_bench:
    st.subheader("Benchmark")
    st.write("Embeds `sample.txt`, ranks chunks for labeled questions, and reports Recall@k, MRR, nDCG, and STS.")
    if selected_ids:
        st.caption("Selected: " + ", ".join(get_embedder(item_id).name for item_id in selected_ids))
    if st.button("Run selected", type="primary", disabled=not selected_ids):
        rows = []
        progress = st.progress(0.0, text="Starting…")
        for index, embedder_id in enumerate(selected_ids, start=1):
            embedder = get_embedder(embedder_id)
            progress.progress(index / len(selected_ids), text=f"Running {embedder.name}…")
            try:
                rows.append(evaluate_embedder(embedder, data["chunks"], data["queries"], data["pairs"]))
            except Exception as exc:  # noqa: BLE001
                rows.append({"model": embedder.name, "ok": False, "error": str(exc)})
        progress.empty()
        st.session_state["results"] = rows

    results = st.session_state.get("results") or []
    if not results:
        st.caption("No results yet. Hash baseline is instant; MiniLM/BGE download once.")
    else:
        table_rows = []
        for result in results:
            if not result.get("ok", True) or "recall_at_1" not in result:
                table_rows.append({"model": result.get("embedder", {}).get("name") or result.get("model"), "error": result.get("error")})
                continue
            table_rows.append(
                {
                    "model": result["embedder"]["name"],
                    "Recall@1": result["recall_at_1"],
                    "Recall@3": result["recall_at_3"],
                    "MRR": result["mrr"],
                    "nDCG@5": result["ndcg_at_5"],
                    "STS ρ": result["sts_spearman"],
                    "Anisotropy": result["anisotropy"],
                    "ms/chunk": result["latency_ms_per_chunk"],
                    "dim": result["dimensions"],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        for result in results:
            if not result.get("queries"):
                continue
            with st.expander(f"{result['embedder']['name']} query breakdown"):
                for item in result["queries"]:
                    top = ", ".join(f"{hit['chunk_id']} ({hit['score']:.2f})" for hit in item["top"])
                    mark = "hit@1" if item["hit_at_1"] else f"expected {', '.join(item['relevant_ids'])}"
                    st.write(f"**{item['query']}** → {top} · {mark}")

with tab_search:
    st.subheader("Search the sample file")
    chosen = pick_embedder("Embedder", "search-model")
    query = st.text_input("Query", "How do plants make sugar from light?")
    if st.button("Search") and chosen and query.strip():
        for hit in search_chunks(get_embedder(chosen), data["chunks"], query):
            st.markdown(f"**{hit['title']}** · `{hit['score']:.3f}`")
            st.write(hit["text"])

with tab_embed:
    st.subheader("Use an embedder")
    chosen = pick_embedder("Embedder", "embed-model")
    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("Text A", "Photosynthesis converts sunlight into chemical energy.")
    with col_b:
        text_b = st.text_area("Text B", "Black holes trap light behind an event horizon.")
    if st.button("Compare cosine") and chosen:
        compared = compare_texts(get_embedder(chosen), text_a, text_b)
        st.metric("cosine", f"{compared['cosine']:.4f}")
        st.caption(f"{compared['dimensions']} dimensions")
    embed_input = st.text_area("Embed this text", "Vector databases retrieve similar chunks for RAG.")
    if st.button("Embed") and chosen:
        embedder = get_embedder(chosen)
        vectors = embedder.embed([embed_input])
        st.write(f"{embedder.name} · {int(vectors.shape[1])}d")
        st.code(", ".join(f"{float(x):.5f}" for x in vectors[0][:24]) + " …")

with tab_metrics:
    st.subheader("What to measure")
    st.write("For RAG, start with cosine plus Recall@k and MRR. STS checks whether similar sentences sit close.")
    for metric in METRIC_GUIDE:
        st.markdown(f"**{metric['name']}** · {metric['role']}")
        st.write(metric["summary"])

with tab_doc:
    st.subheader(data["file"])
    st.write(data["summary"])
    st.write(data["how_it_is_used"])
    for item in data["queries"]:
        st.caption(f"“{item.query}” → {', '.join(item.relevant_ids)}")
    for chunk in data["chunks"]:
        st.markdown(f"### {chunk.title}")
        st.write(chunk.text)

with tab_keys:
    st.subheader("Cloud / local keys")
    st.write("Paste a key and save. It is written to `.env` on this machine.")
    drafts = {}
    for field in public_key_status():
        st.markdown(f"**{field['label']}** · `{field['name']}`")
        st.caption(f"Unlocks {field['unlocks']} · {'set ' + field['masked'] if field['set'] else 'empty'} · [get a key]({field['docs']})")
        drafts[field["name"]] = st.text_input(
            field["name"],
            type="password",
            label_visibility="collapsed",
            placeholder="Leave blank to keep the current value",
            key=f"key-{field['name']}",
        )
    if st.button("Save keys"):
        updated = persist_and_apply(drafts)
        reset_embedders()
        if updated:
            st.success("Saved " + ", ".join(updated))
            st.rerun()
        else:
            st.info("Nothing to save.")
