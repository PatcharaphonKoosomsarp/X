import os, re, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tkinter as tk
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def load_texts_from_folder(folder: str) -> List[Tuple[str, str]]:
    return [(n, open(os.path.join(folder, n), encoding="utf-8").read()) 
            for n in sorted(os.listdir(folder)) if n.lower().endswith(".txt")]
def preprocess_text(text: str) -> List[str]:
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in word_tokenize(text.lower()) if t.isalpha() and t not in set(stopwords.words("english"))]
def build_tfidf_corpus(docs: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    vec = TfidfVectorizer(token_pattern=r"\b\w+\b")
    X = vec.fit_transform([" ".join(d) for d in docs])
    return X.toarray(), list(vec.get_feature_names_out())
def plot_bar(labels, values, title, ylabel): 
        plt.bar(labels, values); 
        plt.title(title); 
        plt.ylabel(ylabel); 
        plt.xticks(rotation=45); 
        plt.tight_layout(); 
        plt.show()
def plot_grouped_bars(labels, series, title, ylabel):
    w = 0.8 / max(1, len(series)); idx = np.arange(len(labels))
    for i, (k, v) in enumerate(series.items()): plt.bar(idx + (i - (len(series)-1)/2)*w, v, w, label=k)
    plt.xticks(idx, labels, rotation=45); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.show()
def plot_similarity_heatmap(M, labels): 
        sns.heatmap(M, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=labels, yticklabels=labels); 
        plt.title("Document Cosine Similarity (TF-IDF)"); plt.tight_layout(); plt.show()
def compile_patterns() -> Dict[str, re.Pattern]:
    return {n: re.compile(p) for n, p in {
        "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "year": r"\b(?:19\d{2}|20\d{2})\b", 
        "integer": r"\b\d+\b", 
        "decimal": r"\b\d+\.\d+\b",
        "email": r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", 
        "url": r"\b(?:https?://|www\.)\S+\b",
        "refer": r"\[(\d+)\]", 
        "word": r"\b[A-Za-z]+\b"
    }.items()}
def regex_counts_for_text(text, patterns): return {n: len(p.findall(text)) for n, p in patterns.items()}
def print_similarity_series(labels, series):
    for k, v in series.items():
        print(f"\nSimilarity ({k}):"); [print(f" - {l}: {x:.3f}") for l, x in zip(labels, v)]
def print_similarity_matrices(labels, matrices):
    for k, M in matrices.items():
        print(f"\nMatrix ({k}):")
        for i, l in enumerate(labels): print(f"{l:<12} " + " ".join(f"{x:8.3f}" for x in M[i]))
def print_top_similarity(labels, series):
    for k, v in series.items():
        if v: print(f"Top ({k}): {labels[np.argmax(v)]} ({max(v):.3f})")
def l1_normalize_rows(A): s = A.sum(1, keepdims=True); s[s==0]=1; return A/s
def weighted_jaccard_scores(D, q): 
    return [np.minimum(D[i], q).sum()/max(1e-9, np.maximum(D[i], q).sum()) for i in range(D.shape[0])]
def weighted_jaccard_matrix(A): 
    return np.array([[np.minimum(A[i],A[j]).sum()/max(1e-9,np.maximum(A[i],A[j]).sum()) 
                      for j in range(A.shape[0])] for i in range(A.shape[0])])
def regex_search_menu(texts):
    patterns = compile_patterns(); labels = [fn for fn, _ in texts]
    pat = input("Choose pattern: date/year/integer/decimal/email/url/refer/word/all/custom\npattern = ").strip().lower()
    if pat not in {*patterns, "all", "custom"}: print("Invalid."); return
    p = re.compile(input("Enter regex: ")) if pat=="custom" else patterns.get(pat)
    sub = input("1) One file 2) All files\nchoice = ").strip()
    if sub=="1":
        for i, l in enumerate(labels): print(f"[{i}] {l}")
        idx = int(input("Select file index: "))
        fname, text = texts[idx]
        if pat=="all":
            counts = regex_counts_for_text(text, patterns)
            plot_bar(list(counts), list(counts.values()), f"Regex in {fname}", "Matches")
            for n, p in patterns.items(): 
                print(f"- {n} ({len(list(p.finditer(text)))}): {', '.join(m.group(0) for m in p.finditer(text))}")
        else:
            ms = list(p.finditer(text))
            plot_bar([pat], [len(ms)], f"{pat} in {fname}", "Matches")
            print(f"\nMatches ({len(ms)}): {', '.join(m.group(0) for m in ms)}")
    elif sub=="2":
        if pat=="all":
            series = {n:[] for n in patterns}
            for fname, text in texts:
                counts = regex_counts_for_text(text, patterns)
                for n in series: series[n].append(counts[n])
            plot_grouped_bars(labels, series, "Regex counts", "Matches")
            for fname, text in texts:
                print(f"\nFile: {fname}")
                for n, ptn in patterns.items():
                    matches = [m.group(0) for m in ptn.finditer(text)]
                    print(f"  Pattern '{n}' ({len(matches)}): {', '.join(matches) if matches else '-'}")
        else:
            counts = [len(list(p.finditer(text))) for _, text in texts]
            plot_bar(labels, counts, f"{pat} matches", "Matches")
            for fname, text in texts:
                matches = [m.group(0) for m in p.finditer(text)]
                print(f"\nFile: {fname} ({len(matches)}): {', '.join(matches) if matches else '-'}")
    else: print("Invalid.")
def query_similarity_menu(texts):
    labels = [fn for fn, _ in texts]
    docs = [preprocess_text(t) for _, t in texts]
    query = preprocess_text(input("Enter your query: "))
    vec = TfidfVectorizer(token_pattern=r"\b\w+\b")
    X = vec.fit_transform([" ".join(d) for d in docs] + [" ".join(query)])
    D, q = X[:-1].toarray(), X[-1].toarray()[0]
    method = input("Choose method: cosine/euclidean/jaccard/manhattan/all\nmethod = ").strip().lower()
    series = {}
    if method in ("cosine","all"): 
        series["cosine"] = list(cosine_similarity(q.reshape(1,-1), D)[0])
    if method in ("euclidean","all"): 
        series["euclidean"] = list(1/(1+pairwise_distances(D, q.reshape(1,-1), metric="euclidean").ravel()))
    if method in ("manhattan","all"):
        D1, q1 = l1_normalize_rows(D), l1_normalize_rows(q.reshape(1,-1)).ravel()
        series["manhattan"] = list(1/(1+pairwise_distances(D1, q1.reshape(1,-1), metric="manhattan").ravel()))
    if method in ("jaccard","all"): series["jaccard"] = weighted_jaccard_scores(D, q)
    if not series: print("Invalid."); return
    print_top_similarity(labels, series); print_similarity_series(labels, series)
    sub = input("2.1) Bar chart 2.2) Heatmap\nchoice = ").strip()
    if sub in ("1","2.1"): plot_grouped_bars(labels, series, "Query vs Documents", "Similarity")
    elif sub in ("2.2","2"):
        data = np.array([series[n] for n in series])
        sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=list(series)); 
        plt.title("Query vs Documents"); plt.tight_layout(); plt.show()
    else: print("Invalid.")
def docs_similarity_menu(texts):
    labels = [fn for fn, _ in texts]
    docs = [preprocess_text(t) for _, t in texts]
    matrix, _ = build_tfidf_corpus(docs)
    method = input("Choose method: cosine/euclidean/jaccard/manhattan/all\nmethod = ").strip().lower()
    matrices = {}
    if method in ("cosine","all"): matrices["cosine"] = cosine_similarity(matrix)
    if method in ("euclidean","all"): matrices["euclidean"] = 1/(1+pairwise_distances(matrix, metric="euclidean"))
    if method in ("manhattan","all"):
        L1 = l1_normalize_rows(matrix)
        matrices["manhattan"] = 1/(1+pairwise_distances(L1, metric="manhattan"))
    if method in ("jaccard","all"): matrices["jaccard"] = weighted_jaccard_matrix(matrix)
    if not matrices: print("Invalid."); return
    print_similarity_matrices(labels, matrices)
    names = list(matrices)
    fig, axes = plt.subplots(nrows=len(names), ncols=1, figsize=(max(7,len(labels)*1.1), max(4,4.5*len(names))))
    if len(names)==1: axes=[axes]
    for ax, name in zip(axes, names):
        sns.heatmap(matrices[name], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Pairwise similarity ({name})")
        ax.set_xlabel("Documents"); ax.set_ylabel("Documents")
    plt.subplots_adjust(hspace=0.8)
    if len(names)>1:
        root=tk.Tk(); root.title("Pairwise similarity")
        container=tk.Frame(root); container.pack(fill=tk.BOTH, expand=True)
        c=tk.Canvas(container); sb=tk.Scrollbar(container, orient=tk.VERTICAL, command=c.yview)
        c.configure(yscrollcommand=sb.set); sb.pack(side=tk.RIGHT, fill=tk.Y); c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner=tk.Frame(c); c.create_window((0,0), window=inner, anchor="nw")
        FigureCanvasTkAgg(fig, master=inner).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        inner.update_idletasks(); c.configure(scrollregion=c.bbox("all"))
        c.bind_all("<MouseWheel>", lambda e: c.yview_scroll(int(-1*(e.delta/120)), "units"))
        root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy())); root.mainloop(); plt.close(fig)
    else: plt.tight_layout(); plt.show(); plt.close(fig)
def main():
    folder = os.path.join(os.getcwd(), "txtfolder")
    if not os.path.isdir(folder): raise SystemExit("Folder 'txtfolder' not found.")
    texts = load_texts_from_folder(folder)
    if not texts: raise SystemExit("No .txt files found.")
    while True:
        print("\nMenu:\n1) Regex search\n2) User query similarity\n3) All txt files similarity\n4) Exit")
        choice = input("Select 1-4: ").strip()
        if choice=="1": regex_search_menu(texts)
        elif choice=="2": query_similarity_menu(texts)
        elif choice=="3": docs_similarity_menu(texts)
        elif choice=="4": print("Bye."); break
        else: print("Invalid.")
if __name__ == "__main__": main()