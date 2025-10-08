
# classifier_pipeline.py
# Reproducible pipeline for multi-label classification of companies into an insurance taxonomy.
# Requires: pandas, scikit-learn, numpy, pickle
# Usage: python classifier_pipeline.py

import pandas as pd, numpy as np, re, string, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(s):
    s = str(s).lower()
    s = re.sub(r'[\t\r\n]', ' ', s)
    s = re.sub(r'[{}]'.format(re.escape(string.punctuation.replace('|',''))), ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_for_match(s):
    s = preprocess_text(s)
    tokens = set([t for t in re.split(r'\W+', s) if t])
    return tokens

def build_and_run(csv_path, tax_path, out_path='/mnt/data/ml_insurance_challenge_annotated.csv', sim_threshold=0.28):
    df = pd.read_csv(csv_path)
    xls = pd.read_excel(tax_path, sheet_name=None)
    tax_df = list(xls.values())[0].copy()

    # Combine fields to a single combined text
    # adapt these names to your CSV's actual columns if needed
    for c in df.columns:
        df[c] = df[c].astype(str).fillna('')
    combined = df.get('description', '').astype(str) + ' | ' + df.get('business_tags', '').astype(str) + ' | ' + df.get('sector', '').astype(str) + ' | ' + df.get('category', '').astype(str) + ' | ' + df.get('niche', '').astype(str)
    df['__combined_text'] = combined
    df['__combined_text_proc'] = df['__combined_text'].apply(preprocess_text)

    # Taxonomy
    tax_cols = {c.lower(): c for c in tax_df.columns}
    label_col = next((tax_cols[k] for k in ('label','taxonomy','name','term','category','insurance_label') if k in tax_cols), tax_df.columns[0])
    desc_col = next((tax_cols[k] for k in ('description','desc','definition') if k in tax_cols), None)
    tax_df['_label_name'] = tax_df[label_col].astype(str).fillna('')
    tax_df['_label_desc'] = tax_df[desc_col].astype(str).fillna('') if desc_col else ''
    extra_cols = [c for c in tax_df.columns if c not in [label_col, desc_col]]
    tax_df['_extra'] = tax_df[extra_cols].astype(str).agg(' | '.join, axis=1) if extra_cols else ''
    tax_df['_label_text'] = (tax_df['_label_name'] + ' | ' + tax_df['_label_desc'] + ' | ' + tax_df['_extra']).str.strip()
    tax_df['_label_text_proc'] = tax_df['_label_text'].apply(preprocess_text)

    # TF-IDF fit on combined corpus of labels + companies
    all_texts = pd.concat([tax_df['_label_text_proc'], df['__combined_text_proc']], ignore_index=True)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
    vectorizer.fit(all_texts)
    X_labels = normalize(vectorizer.transform(tax_df['_label_text_proc']))
    X_companies = normalize(vectorizer.transform(df['__combined_text_proc']))
    sim = cosine_similarity(X_companies, X_labels)

    # Rule-based token mapping
    label_tokens_to_idx = {}
    for idx, row in tax_df.reset_index().iterrows():
        tokens = tokenize_for_match(row['_label_text_proc'])
        for t in tokens:
            label_tokens_to_idx.setdefault(t, set()).add(idx)

    label_names = tax_df['_label_name'].astype(str).tolist()
    def rule_based_labels_for_text(text):
        tokens = tokenize_for_match(text)
        hits = {}
        for t in tokens:
            if t in label_tokens_to_idx:
                for li in label_tokens_to_idx[t]:
                    hits[li] = hits.get(li, 0) + 1
        return sorted(hits.items(), key=lambda x: x[1], reverse=True)

    def predict_labels_for_company(i, sim_row, top_k=3, sim_threshold=sim_threshold):
        chosen = []
        for j, score in enumerate(sim_row):
            if score >= sim_threshold:
                chosen.append((j, float(score)))
        if len(chosen) == 0:
            top_idx = np.argsort(sim_row)[-top_k:][::-1]
            chosen = [(int(j), float(sim_row[j])) for j in top_idx]
        else:
            chosen = sorted(chosen, key=lambda x: x[1], reverse=True)[:top_k]
        rb = rule_based_labels_for_text(preprocess_text(df.loc[i, '__combined_text']))
        for j, cnt in rb[:2]:
            if j not in [c[0] for c in chosen]:
                chosen.append((j, 0.25 + 0.05*cnt))
        chosen = sorted(chosen, key=lambda x: x[1], reverse=True)
        seen = set()
        final = []
        for j, s in chosen:
            if j not in seen:
                seen.add(j)
                final.append((j, s))
        return [(label_names[j], round(float(s), 4)) for j, s in final]

    predicted = []
    predicted_scores = []
    for i in range(df.shape[0]):
        preds = predict_labels_for_company(i, sim[i], top_k=3, sim_threshold=sim_threshold)
        predicted.append([p[0] for p in preds])
        predicted_scores.append([p[1] for p in preds])

    df['insurance_label'] = ['|'.join(p) if len(p)>0 else '' for p in predicted]
    df['insurance_label_list'] = predicted
    df['insurance_label_scores'] = predicted_scores

    # fallback for any empty
    no_label_mask = df['insurance_label'].str.strip() == ''
    if no_label_mask.any():
        for i in df[no_label_mask].index:
            top_j = np.argmax(sim[i])
            df.at[i, 'insurance_label'] = label_names[top_j]
            df.at[i, 'insurance_label_list'] = [label_names[top_j]]
            df.at[i, 'insurance_label_scores'] = [float(sim[i, top_j])]

    df.to_csv(out_path, index=False)
    # save artifacts
    with open(out_path.replace('.csv','_artifacts.pkl'), 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'label_names': label_names, 'tax_df': tax_df, 'sim_threshold': sim_threshold}, f)
    return out_path

if __name__ == '__main__':
    CSV = '/mnt/data/ml_insurance_challenge.csv'
    TAX = '/mnt/data/insurance_taxonomy.xlsx'
    OUT = '/mnt/data/ml_insurance_challenge_annotated.csv'
    build_and_run(CSV, TAX, OUT)
