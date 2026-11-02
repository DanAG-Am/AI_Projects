import json
import pickle
import re
from datetime import datetime
from importlib.util import find_spec
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack, issparse, save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


DATASETS = {
    "3classes": ("originalTraining_3classes.txt", "originalTest_3classes.txt", ["negative", "neutral", "positive"]),
    "binary": ("originalTraining_binary.txt", "originalTest_binary.txt", ["negative", "positive"]),
}
MAX_FEATURES = [5000, 10000]
ENSEMBLE_COMPONENTS = ("bow", "one_hot", "tfidf", "ngrams", "ensemble")
ENSEMBLES = ["combo_" + "+".join(parts) for size in range(2, len(ENSEMBLE_COMPONENTS) + 1) for parts in combinations(ENSEMBLE_COMPONENTS, size)]
NB_ALPHAS = [0.1, 0.5, 1.0]
LR_C_VALUES = [0.5, 1.0, 2.0]
CLASSICAL_C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
NN_CONFIGS = [
    {"name": "course_64_32_16", "hidden": (64, 32, 16), "dropout": 0.2, "learning_rate": 0.001, "batch_size": 200, "epochs": 14},
    {"name": "wide_128_64_32", "hidden": (128, 64, 32), "dropout": 0.35, "learning_rate": 0.0005, "batch_size": 128, "epochs": 18},
]


def _read(path, labels):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\r\n").split("\t", 3)
            if len(parts) == 4 and parts[2].casefold() in labels and parts[3].strip():
                rows.append({"tweet_id": parts[0].strip(), "label": parts[2].casefold(), "text": parts[3].strip()})
    return rows


def _filter_duplicates(train, test):
    records, parent, seen = train + test, list(range(len(train) + len(test))), {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(records):
        keys = ["text:" + " ".join(row["text"].casefold().split())]
        if row["tweet_id"]:
            keys.append("id:" + row["tweet_id"])
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = [find(i) for i in range(len(records))]
    labels_by_group = {}
    for row, group in zip(records, groups):
        labels_by_group.setdefault(group, set()).add(row["label"])
    conflicts = {group for group, labels in labels_by_group.items() if len(labels) > 1}
    train_groups = set(groups[:len(train)])
    clean_train = [row for i, row in enumerate(train) if groups[i] not in conflicts]
    clean_test, used = [], set()
    for i in range(len(train), len(records)):
        group = groups[i]
        if group not in train_groups | conflicts | used:
            clean_test.append(records[i])
            used.add(group)
    return clean_train, clean_test


def _groups(rows):
    group_map, output = {}, []
    for row in rows:
        keys = (("id:" + row["tweet_id"],) if row["tweet_id"] else ()) + ("text:" + " ".join(row["text"].casefold().split()),)
        group = next((group_map[key] for key in keys if key in group_map), len(group_map))
        for key in keys:
            group_map[key] = group
        output.append(group)
    return output


def _vectorizer(name, max_features):
    vectorizers = {
        "bow": CountVectorizer(lowercase=False, max_features=max_features, min_df=2),
        "one_hot": CountVectorizer(lowercase=False, binary=True, max_features=max_features, min_df=2),
        "tfidf": TfidfVectorizer(lowercase=False, max_features=max_features, min_df=2),
        "ngrams": TfidfVectorizer(lowercase=False, ngram_range=(1, 3), max_features=max_features, min_df=2),
        "ensemble": FeatureUnion([
            ("word", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=max_features, min_df=2)),
            ("char", TfidfVectorizer(lowercase=False, analyzer="char", ngram_range=(3, 5), max_features=max_features, min_df=2)),
        ]),
    }
    return vectorizers[name]


def _text_vectors(representation, max_features, train_texts, valid_texts):
    parts = representation.removeprefix("combo_").split("+") if representation.startswith("combo_") else [representation]
    train_vectors, valid_vectors = [], []
    for part in parts:
        vectorizer = _vectorizer(part, max_features)
        train_vectors.append(normalize(vectorizer.fit_transform(train_texts), norm="l2", copy=False))
        valid_vectors.append(normalize(vectorizer.transform(valid_texts), norm="l2", copy=False))
    if len(parts) == 1:
        return train_vectors[0], valid_vectors[0]
    return hstack(train_vectors, format="csr"), hstack(valid_vectors, format="csr")


def _fit_text_representation(representation, max_features, train_texts):
    if representation.startswith("combo_"):
        parts = representation.removeprefix("combo_").split("+")
        vectorizers = [_vectorizer(part, max_features) for part in parts]
        train_vectors = [
            normalize(vectorizer.fit_transform(train_texts), norm="l2", copy=False)
            for vectorizer in vectorizers
        ]
        x_train = train_vectors[0] if len(train_vectors) == 1 else hstack(train_vectors, format="csr")
        return x_train, {"kind": "combined", "parts": parts, "vectorizers": vectorizers}
    if representation in {"tfidf_word", "tfidf_char", "tfidf_word_char"}:
        vectorizer = _classical_vectorizer(representation)
        return vectorizer.fit_transform(train_texts), {"kind": "single", "vectorizer": vectorizer, "normalize": False}
    vectorizer = _vectorizer(representation, max_features)
    x_train = normalize(vectorizer.fit_transform(train_texts), norm="l2", copy=False)
    return x_train, {"kind": "single", "vectorizer": vectorizer, "normalize": True}


def _transform_text_representation(bundle, texts):
    if bundle["kind"] == "single":
        vectors = bundle["vectorizer"].transform(texts)
        if bundle["normalize"]:
            vectors = normalize(vectors, norm="l2", copy=False)
        return vectors
    vectors = [
        normalize(vectorizer.transform(texts), norm="l2", copy=False)
        for vectorizer in bundle["vectorizers"]
    ]
    return vectors[0] if len(vectors) == 1 else hstack(vectors, format="csr")


def _word2vec(texts, preprocess, size=100, window=5, min_count=1, epochs=10):
    from gensim.models import Word2Vec
    sentences = [preprocess(text).split() for text in texts]
    model = Word2Vec(vector_size=size, window=window, min_count=min_count, workers=1, sg=1, seed=42)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model.wv, sentences


def _glove():
    import gensim.downloader as api
    return api.load("glove-twitter-25")


def _token_vectors(sentences, keyed_vectors):
    return [np.array([keyed_vectors[word] for word in sentence if word in keyed_vectors], dtype=np.float32).reshape(-1, keyed_vectors.vector_size) for sentence in sentences]


def _pool_tokens(token_vectors, mode, normalize=False, max_tokens=16):
    dim = token_vectors[0].shape[1]
    output = np.zeros((len(token_vectors), dim if mode == "mean" else dim * max_tokens), dtype=np.float32)
    for i, vectors in enumerate(token_vectors):
        if len(vectors):
            if mode == "mean":
                output[i] = vectors.mean(axis=0)
            else:
                used = min(len(vectors), max_tokens)
                output[i, :used * dim] = vectors[:used].reshape(-1)
    if normalize:
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        output = output / np.maximum(norms, 1e-12)
    return output


def _bert(texts, batch_size=32, max_length=18):
    from transformers import BertModel, BertTokenizer
    name = "bert-base-uncased"
    tokenizer, model = BertTokenizer.from_pretrained(name), BertModel.from_pretrained(name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            hidden = model(**batch).last_hidden_state
            for row, mask in zip(hidden, batch["attention_mask"]):
                length = int(mask.sum().item())
                vectors.append(row[1:max(1, length - 1)].cpu().numpy().astype(np.float16))
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return vectors


def _roberta(train, test, batch_size=32, max_length=96):
    from transformers import AutoModel, AutoTokenizer
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    texts = train + test
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            hidden, mask = model(**batch).last_hidden_state, batch["attention_mask"].unsqueeze(-1)
            output.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy().astype(np.float32))
    vectors = np.concatenate(output) if output else np.empty((0, model.config.hidden_size), dtype=np.float32)
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return vectors[:len(train)], vectors[len(train):]


def _save_pooled(name, train_tokens, test_tokens, folder, manifest):
    for mode in ("mean", "concat"):
        for normalized in (False, True):
            suffix = f"{name}_{mode}_{'l2' if normalized else 'raw'}"
            train_vectors = _pool_tokens(train_tokens, mode, normalized)
            test_vectors = _pool_tokens(test_tokens, mode, normalized)
            np.save(folder / f"{suffix}_train.npy", train_vectors)
            np.save(folder / f"{suffix}_test.npy", test_vectors)
            manifest.append({"dataset": "binary", "representation": suffix, "train_shape": train_vectors.shape, "test_shape": test_vectors.shape})


def _search_embedding_pooling(texts, labels, groups, preprocess, glove_vectors, bert_tokens):
    rows = []
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    glove_tokens = _token_vectors([preprocess(text).split() for text in texts], glove_vectors)
    for fold, (train_ids, valid_ids) in enumerate(cv.split(texts, labels, groups), 1):
        fold_texts = [texts[i] for i in train_ids]
        for size, window in ((50, 3), (100, 5)):
            word_vectors, word_sentences = _word2vec(fold_texts, preprocess, size=size, window=window)
            models = {
                "word2vec": (_token_vectors(word_sentences, word_vectors), _token_vectors([preprocess(texts[i]).split() for i in valid_ids], word_vectors), size, window),
                "glove_twitter_25": ([glove_tokens[i] for i in train_ids], [glove_tokens[i] for i in valid_ids], np.nan, np.nan),
                "bert_base_uncased": ([bert_tokens[i] for i in train_ids], [bert_tokens[i] for i in valid_ids], np.nan, np.nan),
            }
            for representation, (train_tokens, valid_tokens, vector_size, context_window) in models.items():
                if representation != "word2vec" and size != 50:
                    continue
                for mode in ("mean", "concat"):
                    for normalized in (False, True):
                        x_train = _pool_tokens(train_tokens, mode, normalized)
                        x_valid = _pool_tokens(valid_tokens, mode, normalized)
                        for c in (0.5, 1.0, 2.0):
                            model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42).fit(x_train, [labels[i] for i in train_ids])
                            rows.append({"representation": representation, "pooling": mode, "normalized": normalized, "vector_size": vector_size, "window": context_window, "C": c, "fold": fold, "f1_macro": f1_score([labels[i] for i in valid_ids], model.predict(x_valid), average="macro")})
    scores = pd.DataFrame(rows)
    return scores.groupby(["representation", "pooling", "normalized", "vector_size", "window", "C"], dropna=False)["f1_macro"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)


def search_hyperparameters(train_texts, labels, groups, folds=3):
    """Compara representaciones y valores de max_features, alpha y C solo con train."""
    rows = []
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    for representation in ("bow", "tfidf", "ngrams"):
        for max_features in MAX_FEATURES:
            for fold, (train_ids, valid_ids) in enumerate(cv.split(train_texts, labels, groups), 1):
                vectorizer = _vectorizer(representation, max_features)
                x_train = vectorizer.fit_transform([train_texts[i] for i in train_ids])
                x_valid = vectorizer.transform([train_texts[i] for i in valid_ids])
                y_train, y_valid = [labels[i] for i in train_ids], [labels[i] for i in valid_ids]
                for alpha in NB_ALPHAS:
                    model = MultinomialNB(alpha=alpha).fit(x_train, y_train)
                    prediction = model.predict(x_valid)
                    rows.append({"representation": representation, "max_features": max_features, "algorithm": "MultinomialNB", "alpha": alpha, "C": np.nan, "fold": fold, "accuracy": accuracy_score(y_valid, prediction), "f1_macro": f1_score(y_valid, prediction, average="macro")})
                for c in LR_C_VALUES:
                    model = LogisticRegression(C=c, class_weight="balanced", max_iter=1800, random_state=42).fit(x_train, y_train)
                    prediction = model.predict(x_valid)
                    rows.append({"representation": representation, "max_features": max_features, "algorithm": "LogisticRegression", "alpha": np.nan, "C": c, "fold": fold, "accuracy": accuracy_score(y_valid, prediction), "f1_macro": f1_score(y_valid, prediction, average="macro")})
    scores = pd.DataFrame(rows)
    keys = ["representation", "max_features", "algorithm", "alpha", "C"]
    summary = scores.groupby(keys, dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary.sort_values(["accuracy_mean", "f1_macro_mean"], ascending=False)


def _new_nn(input_size, config, n_classes, device):
    layers, size = [], input_size
    for width in config["hidden"]:
        layers.extend((torch.nn.Linear(size, width), torch.nn.ReLU(), torch.nn.Dropout(config["dropout"])))
        size = width
    return torch.nn.Sequential(*layers, torch.nn.Linear(size, n_classes)).to(device)


def _fit_nn(x_train, y_train, config, n_classes, device, seed):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    model = _new_nn(x_train.shape[1], config, n_classes, device)
    counts = np.bincount(y_train, minlength=n_classes)
    weights = len(y_train) / np.maximum(counts, 1) / n_classes
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(seed)
    for _ in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), config["batch_size"]):
            ids = order[start:start + config["batch_size"]]
            values = x_train[ids].toarray() if issparse(x_train) else np.asarray(x_train[ids])
            features = torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=device)
            targets = torch.as_tensor(y_train[ids], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
    return model


def _nn_predict(x_train, y_train, x_valid, config, n_classes, device, seed):
    model = _fit_nn(x_train, y_train, config, n_classes, device, seed)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, x_valid.shape[0], config["batch_size"]):
            ids = np.arange(start, min(start + config["batch_size"], x_valid.shape[0]))
            values = x_valid[ids].toarray() if issparse(x_valid) else np.asarray(x_valid[ids])
            features = torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=device)
            predictions.extend(model(features).argmax(dim=1).cpu().numpy())
    return np.asarray(predictions)


def search_neural_networks(preprocess, output_dir="vector_outputs", folds=3):
    """Compara dos MLP sobre las representaciones con CV; evalúa la mejor en test."""
    root, output_dir = Path(__file__).parent, Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cv, cv_rows, test_rows = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42), [], []
    dense_names = ("word2vec_mean_l2", "glove_twitter_25_mean_l2", "bert_base_uncased_mean_l2")
    sparse_names = ("bow", "one_hot", "tfidf", "ngrams", "ensemble")
    for dataset, (train_file, test_file, allowed) in DATASETS.items():
        train, test = _filter_duplicates(_read(root / train_file, allowed), _read(root / test_file, allowed))
        labels = sorted(allowed)
        label_ids = {label: i for i, label in enumerate(labels)}
        y_train = np.asarray([label_ids[row["label"]] for row in train])
        y_test = np.asarray([label_ids[row["label"]] for row in test])
        texts, test_texts = [preprocess(row["text"]) for row in train], [preprocess(row["text"]) for row in test]
        groups, folder = _groups(train), output_dir / dataset
        splits = list(cv.split(texts, y_train, groups))
        representations = [(name, features) for name in sparse_names for features in MAX_FEATURES]
        if dataset == "binary":
            representations += [(name, None) for name in dense_names if (folder / f"{name}_train.npy").exists()]

        for representation, max_features in representations:
            is_sparse = representation in sparse_names
            if not is_sparse:
                dense_train = np.load(folder / f"{representation}_train.npy", mmap_mode="r")
                dense_test = np.load(folder / f"{representation}_test.npy", mmap_mode="r")
            for config in NN_CONFIGS:
                for fold, (train_ids, valid_ids) in enumerate(splits, 1):
                    if is_sparse:
                        fold_train = [texts[i] for i in train_ids]
                        fold_valid = [texts[i] for i in valid_ids]
                        if representation.startswith("combo_"):
                            x_train, x_valid = _text_vectors(representation, max_features, fold_train, fold_valid)
                        else:
                            vectorizer = _vectorizer(representation, max_features)
                            x_train = normalize(vectorizer.fit_transform(fold_train), norm="l2", copy=False)
                            x_valid = normalize(vectorizer.transform(fold_valid), norm="l2", copy=False)
                    elif representation == "word2vec_mean_l2":
                        word_vectors, sentences = _word2vec([train[i]["text"] for i in train_ids], preprocess)
                        x_train = _pool_tokens(_token_vectors(sentences, word_vectors), "mean", True)
                        valid_tokens = _token_vectors([preprocess(train[i]["text"]).split() for i in valid_ids], word_vectors)
                        x_valid = _pool_tokens(valid_tokens, "mean", True)
                    else:
                        x_train, x_valid = dense_train[train_ids], dense_train[valid_ids]
                    prediction = _nn_predict(x_train, y_train[train_ids], x_valid, config, len(labels), device, 42 + fold)
                    score = f1_score(y_train[valid_ids], prediction, average="macro")
                    cv_rows.append({"dataset": dataset, "representation": representation, "max_features": max_features, **config, "fold": fold, "accuracy": accuracy_score(y_train[valid_ids], prediction), "f1_macro": score})
        dataset_rows = pd.DataFrame([row for row in cv_rows if row["dataset"] == dataset])
        keys = ["representation", "max_features", "name", "hidden", "dropout", "learning_rate", "batch_size", "epochs"]
        summary = dataset_rows.groupby(keys, dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
            for column in summary.columns
        ]
        for representation in dataset_rows["representation"].unique():
            best = summary[summary["representation"] == representation].sort_values("f1_macro_mean", ascending=False).iloc[0]
            config = next(item for item in NN_CONFIGS if item["name"] == best["name"])
            if representation in sparse_names:
                if representation.startswith("combo_"):
                    x_train, x_test = _text_vectors(representation, int(best["max_features"]), texts, test_texts)
                else:
                    vectorizer = _vectorizer(representation, int(best["max_features"]))
                    x_train = normalize(vectorizer.fit_transform(texts), norm="l2", copy=False)
                    x_test = normalize(vectorizer.transform(test_texts), norm="l2", copy=False)
            else:
                filename = "roberta" if representation == "roberta" else representation
                x_train = np.load(folder / f"{filename}_train.npy", mmap_mode="r")
                x_test = np.load(folder / f"{filename}_test.npy", mmap_mode="r")
            prediction = _nn_predict(x_train, y_train, x_test, config, len(labels), device, 2026)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, labels=np.arange(len(labels)), zero_division=0)
            row = {"dataset": dataset, "representation": representation, "max_features": best["max_features"], "configuration": config["name"], "cv_accuracy": best["accuracy_mean"], "cv_macro_f1": best["f1_macro_mean"], "test_accuracy": accuracy_score(y_test, prediction), "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction), "test_macro_f1": f1_score(y_test, prediction, average="macro")}
            for i, label in enumerate(labels):
                row.update({f"precision_{label}": precision[i], f"recall_{label}": recall[i], f"f1_{label}": f1[i]})
            test_rows.append(row)

    cv_results = pd.DataFrame(cv_rows)
    summary = cv_results.groupby(["dataset", "representation", "max_features", "name", "hidden", "dropout", "learning_rate", "batch_size", "epochs"], dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    test_results = pd.DataFrame(test_rows)
    cv_results.to_csv(output_dir / "neural_network_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "neural_network_hyperparameter_search.csv", index=False)
    test_results.to_csv(output_dir / "neural_network_test_results.csv", index=False)
    return summary, test_results


def search_ensemble_combinations(preprocess, output_dir="vector_outputs", folds=3):
    """Evalúa las 26 combinaciones con regresión logística y las 3 mejores con MLP."""
    root, output_dir = Path(__file__).parent, Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_rows, test_rows = [], []

    for dataset, (train_file, test_file, allowed) in DATASETS.items():
        train, test = _filter_duplicates(_read(root / train_file, allowed), _read(root / test_file, allowed))
        labels = sorted(allowed)
        label_ids = {label: i for i, label in enumerate(labels)}
        y_train = np.asarray([label_ids[row["label"]] for row in train])
        y_test = np.asarray([label_ids[row["label"]] for row in test])
        texts = [preprocess(row["text"]) for row in train]
        test_texts = [preprocess(row["text"]) for row in test]
        splits = list(cv.split(texts, y_train, _groups(train)))

        for representation in ENSEMBLES:
            for max_features in MAX_FEATURES:
                for fold, (train_ids, valid_ids) in enumerate(splits, 1):
                    x_train, x_valid = _text_vectors(representation, max_features, [texts[i] for i in train_ids], [texts[i] for i in valid_ids])
                    for c in LR_C_VALUES:
                        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1800, random_state=42).fit(x_train, y_train[train_ids])
                        prediction = model.predict(x_valid)
                        score = f1_score(y_train[valid_ids], prediction, average="macro")
                        cv_rows.append({"dataset": dataset, "representation": representation, "max_features": max_features, "algorithm": "LogisticRegression", "C": c, "name": "", "fold": fold, "accuracy": accuracy_score(y_train[valid_ids], prediction), "f1_macro": score})

        results = pd.DataFrame([row for row in cv_rows if row["dataset"] == dataset])
        lr_summary = results.groupby(["representation", "max_features", "algorithm", "C"], dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
        lr_summary.columns = [
            "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
            for column in lr_summary.columns
        ]
        top_combinations = lr_summary.sort_values("f1_macro_mean", ascending=False).drop_duplicates("representation").head(3)
        for _, candidate in top_combinations.iterrows():
            representation, max_features = candidate["representation"], int(candidate["max_features"])
            for config in NN_CONFIGS:
                for fold, (train_ids, valid_ids) in enumerate(splits, 1):
                    x_train, x_valid = _text_vectors(representation, max_features, [texts[i] for i in train_ids], [texts[i] for i in valid_ids])
                    prediction = _nn_predict(x_train, y_train[train_ids], x_valid, config, len(labels), device, 42 + fold)
                    score = f1_score(y_train[valid_ids], prediction, average="macro")
                    cv_rows.append({"dataset": dataset, "representation": representation, "max_features": max_features, "algorithm": "MLP", "C": np.nan, **config, "fold": fold, "accuracy": accuracy_score(y_train[valid_ids], prediction), "f1_macro": score})

        results = pd.DataFrame([row for row in cv_rows if row["dataset"] == dataset])
        keys = ["representation", "max_features", "algorithm", "C", "name", "hidden", "dropout", "learning_rate", "batch_size", "epochs"]
        summary = results.groupby(keys, dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
            for column in summary.columns
        ]
        best = summary.sort_values("f1_macro_mean", ascending=False).iloc[0]
        if best["algorithm"] == "LogisticRegression":
            x_train, x_test = _text_vectors(best["representation"], int(best["max_features"]), texts, test_texts)
            model = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=1800, random_state=42).fit(x_train, y_train)
            prediction = model.predict(x_test)
        else:
            config = next(item for item in NN_CONFIGS if item["name"] == best["name"])
            x_train, x_test = _text_vectors(best["representation"], int(best["max_features"]), texts, test_texts)
            prediction = _nn_predict(x_train, y_train, x_test, config, len(labels), device, 2026)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, labels=np.arange(len(labels)), zero_division=0)
        row = {"dataset": dataset, "representation": best["representation"], "max_features": best["max_features"], "algorithm": best["algorithm"], "C": best["C"], "configuration": best["name"], "cv_accuracy": best["accuracy_mean"], "cv_macro_f1": best["f1_macro_mean"], "test_accuracy": accuracy_score(y_test, prediction), "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction), "test_macro_f1": f1_score(y_test, prediction, average="macro")}
        for i, label in enumerate(labels):
            row.update({f"precision_{label}": precision[i], f"recall_{label}": recall[i], f"f1_{label}": f1[i]})
        test_rows.append(row)

    cv_results = pd.DataFrame(cv_rows)
    keys = ["dataset", "representation", "max_features", "algorithm", "C", "name", "hidden", "dropout", "learning_rate", "batch_size", "epochs"]
    summary = cv_results.groupby(keys, dropna=False).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    test_results = pd.DataFrame(test_rows)
    cv_results.to_csv(output_dir / "ensemble_combo_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "ensemble_combo_search.csv", index=False)
    test_results.to_csv(output_dir / "ensemble_combo_test_results.csv", index=False)
    return summary, test_results


def _classical_vectorizer(representation):
    vectorizers = {
        "tfidf_word": TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "tfidf_char": TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        "tfidf_word_char": FeatureUnion([
            ("word", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]),
    }
    return vectorizers[representation]


def _classical_model(c, class_weight):
    return LogisticRegression(C=c, class_weight=class_weight, solver="liblinear", max_iter=3000, random_state=42)


def _bert_tensors(tokenizer, texts, labels=None, max_length=48):
    from torch.utils.data import TensorDataset
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if labels is not None:
        tensors.append(torch.as_tensor(labels, dtype=torch.long))
    return TensorDataset(*tensors)


def _fit_bert_classifier(texts, labels, tokenizer, device, epochs, seed, batch_size=32, validation_texts=None, validation_labels=None):
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    dataset = _bert_tensors(tokenizer, texts, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    history = []
    for _ in range(epochs):
        model.train()
        for input_ids, attention_mask, targets in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if validation_texts is not None and validation_labels is not None:
            prediction = _bert_predictions(model, tokenizer, validation_texts, device)
            history.append({
                "accuracy": accuracy_score(validation_labels, prediction),
                "f1_macro": f1_score(validation_labels, prediction, average="macro"),
            })
    return model, history


def _bert_predictions(model, tokenizer, texts, device, batch_size=64):
    from torch.utils.data import DataLoader

    dataset = _bert_tensors(tokenizer, texts)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for input_ids, attention_mask in loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
    return np.asarray(predictions)


def fine_tune_bert_binary(preprocess, output_dir="vector_outputs", folds=2, epochs=2):
    """Ajusta BERT base para sentimiento binario y selecciona épocas con CV agrupada."""
    from transformers import AutoTokenizer

    root, output_dir = Path(__file__).parent, Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file, test_file, allowed = DATASETS["binary"]
    train, test = _filter_duplicates(_read(root / train_file, allowed), _read(root / test_file, allowed))
    labels = sorted(allowed)
    label_ids = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([label_ids[row["label"]] for row in train])
    y_test = np.asarray([label_ids[row["label"]] for row in test])
    texts = [preprocess(row["text"]) for row in train]
    test_texts = [preprocess(row["text"]) for row in test]
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cv_rows = []

    for fold, (train_ids, valid_ids) in enumerate(cv.split(texts, y_train, _groups(train)), 1):
        model, history = _fit_bert_classifier(
            [texts[i] for i in train_ids],
            y_train[train_ids],
            tokenizer,
            device,
            epochs,
            42 + fold,
            validation_texts=[texts[i] for i in valid_ids],
            validation_labels=y_train[valid_ids],
        )
        for epoch, metrics in enumerate(history, 1):
            cv_rows.append({
                "fold": fold,
                "epoch": epoch,
                **metrics,
            })
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    cv_results = pd.DataFrame(cv_rows)
    summary = cv_results.groupby("epoch").agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(["accuracy_mean", "f1_macro_mean"], ascending=False)
    best_epoch = int(summary.iloc[0]["epoch"])
    model, _ = _fit_bert_classifier(texts, y_train, tokenizer, device, best_epoch, 2026)
    prediction = _bert_predictions(model, tokenizer, test_texts, device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, labels=np.arange(len(labels)), zero_division=0)
    test_result = {
        "dataset": "binary",
        "representation": "bert_base_finetuned",
        "algorithm": "BertForSequenceClassification",
        "epochs": best_epoch,
        "cv_accuracy_mean": summary.iloc[0]["accuracy_mean"],
        "cv_accuracy_std": summary.iloc[0]["accuracy_std"],
        "cv_macro_f1_mean": summary.iloc[0]["f1_macro_mean"],
        "test_accuracy": accuracy_score(y_test, prediction),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "test_macro_f1": f1_score(y_test, prediction, average="macro"),
    }
    for i, label in enumerate(labels):
        test_result.update({f"precision_{label}": precision[i], f"recall_{label}": recall[i], f"f1_{label}": f1[i]})
    test_results = pd.DataFrame([test_result])
    cv_results.to_csv(output_dir / "bert_finetune_binary_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "bert_finetune_binary_search.csv", index=False)
    test_results.to_csv(output_dir / "bert_finetune_binary_test_results.csv", index=False)
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return summary, test_results


def search_classical_models(preprocess, output_dir="vector_outputs", folds=3):
    """Compara regresión logística sobre TF-IDF de palabras y caracteres."""
    root, output_dir = Path(__file__).parent, Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file, test_file, allowed = DATASETS["binary"]
    train, test = _filter_duplicates(_read(root / train_file, allowed), _read(root / test_file, allowed))
    labels = sorted(allowed)
    label_ids = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([label_ids[row["label"]] for row in train])
    y_test = np.asarray([label_ids[row["label"]] for row in test])
    texts = [preprocess(row["text"]) for row in train]
    test_texts = [preprocess(row["text"]) for row in test]
    groups = _groups(train)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(cv.split(texts, y_train, groups))
    cv_rows = []

    for representation in ("tfidf_word", "tfidf_char", "tfidf_word_char"):
        fold_vectors = []
        for train_ids, valid_ids in splits:
            vectorizer = _classical_vectorizer(representation)
            x_train = vectorizer.fit_transform([texts[i] for i in train_ids])
            x_valid = vectorizer.transform([texts[i] for i in valid_ids])
            fold_vectors.append((train_ids, valid_ids, x_train, x_valid))

        algorithm = "LogisticRegression"
        for c in CLASSICAL_C_VALUES:
            for class_weight in (None, "balanced"):
                for fold, (train_ids, valid_ids, x_train, x_valid) in enumerate(fold_vectors, 1):
                    model = _classical_model(c, class_weight).fit(x_train, y_train[train_ids])
                    prediction = model.predict(x_valid)
                    cv_rows.append({
                        "representation": representation,
                        "algorithm": algorithm,
                        "C": c,
                        "class_weight": class_weight or "none",
                        "fold": fold,
                        "accuracy": accuracy_score(y_train[valid_ids], prediction),
                        "f1_macro": f1_score(y_train[valid_ids], prediction, average="macro"),
                    })

    cv_results = pd.DataFrame(cv_rows)
    keys = ["representation", "algorithm", "C", "class_weight"]
    summary = cv_results.groupby(keys).agg({"accuracy": ["mean", "std"], "f1_macro": ["mean", "std"]}).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(["accuracy_mean", "f1_macro_mean"], ascending=False)
    best = summary.iloc[0]
    vectorizer = _classical_vectorizer(best["representation"])
    x_train = vectorizer.fit_transform(texts)
    x_test = vectorizer.transform(test_texts)
    model = _classical_model(best["C"], None if best["class_weight"] == "none" else best["class_weight"])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, labels=np.arange(len(labels)), zero_division=0)
    test_result = {
        "dataset": "binary",
        "representation": best["representation"],
        "algorithm": best["algorithm"],
        "C": best["C"],
        "class_weight": best["class_weight"],
        "cv_accuracy_mean": best["accuracy_mean"],
        "cv_accuracy_std": best["accuracy_std"],
        "cv_macro_f1_mean": best["f1_macro_mean"],
        "test_accuracy": accuracy_score(y_test, prediction),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "test_macro_f1": f1_score(y_test, prediction, average="macro"),
    }
    for i, label in enumerate(labels):
        test_result.update({f"precision_{label}": precision[i], f"recall_{label}": recall[i], f"f1_{label}": f1[i]})
    test_results = pd.DataFrame([test_result])
    cv_results.to_csv(output_dir / "classical_binary_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "classical_binary_search.csv", index=False)
    test_results.to_csv(output_dir / "classical_binary_test_results.csv", index=False)
    return summary, test_results


def generate_all_vectors(preprocess, output_dir="vector_outputs", include_roberta=True, include_embeddings=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, searches, embedding_search = [], [], pd.DataFrame()
    for name, (train_file, test_file, labels_allowed) in DATASETS.items():
        train, test = _filter_duplicates(_read(train_file, labels_allowed), _read(test_file, labels_allowed))
        x_train = [preprocess(row["text"]) for row in train]
        x_test = [preprocess(row["text"]) for row in test]
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train)[["tweet_id", "label"]].to_csv(folder / "train_metadata.csv", index=False)
        pd.DataFrame(test)[["tweet_id", "label"]].to_csv(folder / "test_metadata.csv", index=False)

        search = search_hyperparameters(x_train, [row["label"] for row in train], _groups(train))
        search.insert(0, "dataset", name)
        searches.append(search)
        search.to_csv(folder / "hyperparameter_search.csv", index=False)

        for rep in ("bow", "one_hot", "tfidf", "ngrams", "ensemble"):
            vectorizer = _vectorizer(rep, 20_000)
            a, b = vectorizer.fit_transform(x_train), vectorizer.transform(x_test)
            save_npz(folder / f"{rep}_train.npz", a)
            save_npz(folder / f"{rep}_test.npz", b)
            manifest.append({"dataset": name, "representation": rep, "train_shape": a.shape, "test_shape": b.shape})
        if include_roberta:
            a, b = _roberta([r["text"] for r in train], [r["text"] for r in test])
            np.save(folder / "roberta_train.npy", a)
            np.save(folder / "roberta_test.npy", b)
            manifest.append({"dataset": name, "representation": "roberta", "train_shape": a.shape, "test_shape": b.shape})
        if name == "binary" and include_embeddings:
            glove_vectors = _glove()
            bert_vectors = _bert([r["text"] for r in train + test])
            bert_train, bert_test = bert_vectors[:len(train)], bert_vectors[len(train):]
            word_vectors, word_sentences = _word2vec([r["text"] for r in train], preprocess)
            _save_pooled("word2vec", _token_vectors(word_sentences, word_vectors), _token_vectors([preprocess(text).split() for text in [r["text"] for r in test]], word_vectors), folder, manifest)
            _save_pooled("glove_twitter_25", _token_vectors([text.split() for text in x_train], glove_vectors), _token_vectors([preprocess(r["text"]).split() for r in test], glove_vectors), folder, manifest)
            _save_pooled("bert_base_uncased", bert_train, bert_test, folder, manifest)
            embedding_search = _search_embedding_pooling([r["text"] for r in train], [r["label"] for r in train], _groups(train), preprocess, glove_vectors, bert_train)
            embedding_search.to_csv(folder / "embedding_hyperparameter_search.csv", index=False)
            del bert_vectors, bert_train, bert_test, word_vectors, word_sentences
    results = pd.DataFrame(manifest)
    results.to_csv(output_dir / "vector_manifest.csv", index=False)
    pd.concat(searches).to_csv(output_dir / "hyperparameter_search.csv", index=False)
    if not embedding_search.empty:
        embedding_search.to_csv(output_dir / "embedding_hyperparameter_search.csv", index=False)
    return results, pd.concat(searches, ignore_index=True), embedding_search


def _candidate_rows(frame, dataset, family, algorithm, representation, parameters):
    rows = []
    for _, source in frame.iterrows():
        if source.get(representation) == "roberta":
            continue
        accuracy_mean = source.get("accuracy_mean")
        f1_mean = source.get("f1_macro_mean")
        if pd.isna(accuracy_mean) or pd.isna(f1_mean):
            continue
        params = {}
        for key in parameters:
            if key not in source.index:
                continue
            value = source[key]
            if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            params[key] = value
        rows.append({
            "dataset": source.get("dataset", dataset),
            "model_family": family,
            "algorithm": source.get(algorithm, algorithm) if isinstance(algorithm, str) else algorithm(source),
            "representation": source.get(representation, representation) if isinstance(representation, str) else representation(source),
            "max_features": source.get("max_features"),
            "parameters": params,
            "cv_accuracy_mean": float(accuracy_mean),
            "cv_accuracy_std": float(source.get("accuracy_std", 0.0)),
            "cv_macro_f1_mean": float(f1_mean),
            "cv_macro_f1_std": float(source.get("f1_macro_std", 0.0)),
        })
    return rows


def _candidate_config(candidate):
    values = candidate["parameters"]
    return {
        "name": values.get("name", "saved_mlp"),
        "hidden": tuple(values.get("hidden", ())),
        "dropout": float(values.get("dropout", 0.0)),
        "learning_rate": float(values.get("learning_rate", 0.001)),
        "batch_size": int(values.get("batch_size", 128)),
        "epochs": int(values.get("epochs", 1)),
    }


def _representation_description(representation, max_features):
    features = "sin límite de vocabulario explícito" if max_features is None or pd.isna(max_features) else f"máximo {int(max_features)} características"
    if representation.startswith("combo_"):
        parts = representation.removeprefix("combo_").split("+")
        return f"Ensamble concatenado de {', '.join(parts)}; {features}; normalización L2 por bloque"
    descriptions = {
        "bow": "CountVectorizer de unigramas, min_df=2",
        "one_hot": "CountVectorizer binario de unigramas, min_df=2",
        "tfidf": "TF-IDF de unigramas, min_df=2",
        "ngrams": "TF-IDF de unigramas a trigramas, min_df=2",
        "ensemble": "FeatureUnion de TF-IDF word (1-2) y character (3-5)",
        "tfidf_word": "TF-IDF de palabras (unigramas y bigramas), sublinear_tf, min_df=2",
        "tfidf_char": "TF-IDF de caracteres char_wb (3-5), sublinear_tf, min_df=2",
        "tfidf_word_char": "FeatureUnion de TF-IDF word (1-2) + char_wb (3-5), sublinear_tf, min_df=2",
        "word2vec_mean_l2": "Word2Vec, promedio de tokens y normalización L2",
        "glove_twitter_25_mean_l2": "GloVe Twitter 25d, promedio de tokens y normalización L2",
        "bert_base_uncased_mean_l2": "BERT base uncased, promedio de tokens y normalización L2",
        "bert_base_finetuned": "BERT base uncased fine-tuned, max_length=48",
    }
    description = descriptions.get(representation, representation)
    if max_features is None or pd.isna(max_features):
        return description
    return f"{description}; {features}"


def _fit_candidate_for_export(candidate, preprocess, output_dir):
    dataset = candidate["dataset"]
    train_file, test_file, allowed = DATASETS[dataset]
    train, test = _filter_duplicates(_read(Path(__file__).parent / train_file, allowed), _read(Path(__file__).parent / test_file, allowed))
    labels = sorted(allowed)
    label_ids = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([label_ids[row["label"]] for row in train])
    y_test = np.asarray([label_ids[row["label"]] for row in test])
    texts = [preprocess(row["text"]) for row in train]
    test_texts = [preprocess(row["text"]) for row in test]
    family, algorithm = candidate["model_family"], candidate["algorithm"]
    representation = candidate["representation"]
    parameters = json.loads(json.dumps(candidate["parameters"], default=lambda value: value.item() if isinstance(value, np.generic) else list(value) if isinstance(value, tuple) else str(value)))
    raw_max_features = candidate.get("max_features")
    metadata_max_features = None if raw_max_features is None or pd.isna(raw_max_features) else int(raw_max_features)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if family == "bert_finetune":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model, _ = _fit_bert_classifier(texts, y_train, tokenizer, device, int(parameters.get("epochs", parameters.get("epoch"))), 2026)
        prediction = _bert_predictions(model, tokenizer, test_texts, device)
        payload = {
            "format_version": 1,
            "metadata": {
                "dataset": dataset,
                "model_family": family,
                "algorithm": "BertForSequenceClassification",
                "representation": "bert_base_finetuned",
                "max_length": 48,
                "model_name": "bert-base-uncased",
                "epochs": int(parameters.get("epochs", parameters.get("epoch"))),
                "labels": labels,
            },
            "model_state_dict": {
                key: value.detach().cpu().to(torch.float16) if value.is_floating_point() else value.detach().cpu()
                for key, value in model.cpu().state_dict().items()
            },
        }
        del model, tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()
    elif family == "mlp" or algorithm == "MLP":
        max_features = candidate.get("max_features")
        feature_count = 10000 if max_features is None or pd.isna(max_features) else int(max_features)
        if representation in {"word2vec_mean_l2", "glove_twitter_25_mean_l2", "bert_base_uncased_mean_l2"}:
            data_folder = Path(output_dir) / dataset
            train_vectors = np.load(data_folder / f"{representation}_train.npy", mmap_mode="r")
            test_vectors = np.load(data_folder / f"{representation}_test.npy", mmap_mode="r")
            vectorizer_bundle = None
        else:
            train_vectors, vectorizer_bundle = _fit_text_representation(
                representation,
                feature_count,
                texts,
            )
            test_vectors = _transform_text_representation(vectorizer_bundle, test_texts)
        config = _candidate_config(candidate)
        model = _fit_nn(train_vectors, y_train, config, len(labels), device, 2026)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, test_vectors.shape[0], config["batch_size"]):
                ids = np.arange(start, min(start + config["batch_size"], test_vectors.shape[0]))
                values = test_vectors[ids].toarray() if issparse(test_vectors) else np.asarray(test_vectors[ids])
                features = torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=device)
                predictions.extend(model(features).argmax(dim=1).cpu().numpy())
        prediction = np.asarray(predictions)
        payload = {
            "format_version": 1,
            "metadata": {
                "dataset": dataset,
                "model_family": family,
                "algorithm": "MLP",
                "representation": representation,
                "max_features": metadata_max_features,
                "parameters": parameters,
                "labels": labels,
                "dense_vectors_source": str(Path(output_dir) / dataset / f"{representation}_train.npy") if vectorizer_bundle is None else None,
            },
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.cpu().state_dict().items()
            },
        }
        if vectorizer_bundle is not None:
            vectorizer_bytes = pickle.dumps(vectorizer_bundle, protocol=pickle.HIGHEST_PROTOCOL)
            payload["vectorizer_bundle"] = torch.tensor(list(vectorizer_bytes), dtype=torch.uint8)
    else:
        max_features = candidate.get("max_features")
        feature_count = 10000 if max_features is None or pd.isna(max_features) else int(max_features)
        if family == "traditional":
            vectorizer = _vectorizer(representation, feature_count)
            x_train = vectorizer.fit_transform(texts)
            x_test = vectorizer.transform(test_texts)
            vectorizer_bundle = {"kind": "single", "vectorizer": vectorizer, "normalize": False}
        else:
            x_train, vectorizer_bundle = _fit_text_representation(
                representation,
                feature_count,
                texts,
            )
            x_test = _transform_text_representation(vectorizer_bundle, test_texts)
        class_weight = parameters.get("class_weight")
        c = float(parameters.get("C", 1.0))
        if algorithm == "MultinomialNB":
            model = MultinomialNB(alpha=float(parameters["alpha"]))
        else:
            if "class_weight" in parameters:
                logistic_class_weight = None if class_weight == "none" else class_weight
            else:
                logistic_class_weight = "balanced"
            model = LogisticRegression(
                C=c,
                class_weight=logistic_class_weight,
                max_iter=3000,
                random_state=42,
            )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        bundle_bytes = pickle.dumps({"estimator": model, "vectorizer_bundle": vectorizer_bundle}, protocol=pickle.HIGHEST_PROTOCOL)
        payload = {
            "format_version": 1,
            "metadata": {
                "dataset": dataset,
                "model_family": family,
                "algorithm": algorithm,
                "representation": representation,
                "max_features": metadata_max_features,
                "parameters": parameters,
                "labels": labels,
                "serialization": "pickle bytes in uint8 tensor; load only trusted files",
            },
            "pipeline_pickle": torch.tensor(list(bundle_bytes), dtype=torch.uint8),
        }
    return payload, accuracy_score(y_test, prediction), f1_score(y_test, prediction, average="macro")


def run_complete_experiments(preprocess, output_dir="vector_outputs", top_n=20):
    """Ejecuta búsquedas PLN y exporta hasta top_n modelos por conjunto de datos."""
    root = Path(__file__).parent
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if find_spec("gensim") is None:
        raise ModuleNotFoundError(
            "La ejecución consolidada requiere gensim para validar Word2Vec dentro de cada fold. "
            "Selecciona un entorno del notebook que tenga gensim instalado antes de correr la celda."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / "top20_models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    binary_folder = output_dir / "binary"
    cached_embeddings = [
        binary_folder / f"{representation}_{split}.npy"
        for representation in ("word2vec_mean_l2", "glove_twitter_25_mean_l2", "bert_base_uncased_mean_l2")
        for split in ("train", "test")
    ]
    embedding_search_path = output_dir / "embedding_hyperparameter_search.csv"
    missing_embeddings = [str(path) for path in cached_embeddings if not path.is_file()]
    if missing_embeddings or not embedding_search_path.is_file():
        missing = missing_embeddings + ([] if embedding_search_path.is_file() else [str(embedding_search_path)])
        raise FileNotFoundError(
            "Faltan resultados/representaciones de embeddings precalculados requeridos para incluir "
            f"todas las pruebas: {', '.join(missing)}. Genéralos primero con generate_all_vectors "
            "y una instalación de gensim disponible."
        )

    vector_manifest, hyperparameter_results, _ = generate_all_vectors(
        preprocess,
        output_dir=output_dir,
        include_roberta=False,
        include_embeddings=False,
    )
    embedding_results = pd.read_csv(embedding_search_path)
    nn_cv_results, nn_test_results = search_neural_networks(preprocess, output_dir=output_dir)
    ensemble_cv_results, ensemble_test_results = search_ensemble_combinations(preprocess, output_dir=output_dir)
    classical_cv_results, classical_test_results = search_classical_models(preprocess, output_dir=output_dir)
    bert_cv_results, bert_test_results = fine_tune_bert_binary(preprocess, output_dir=output_dir)

    candidates = []
    candidates += _candidate_rows(
        hyperparameter_results,
        None,
        "traditional",
        "algorithm",
        "representation",
        ["alpha", "C", "max_features"],
    )
    candidates += _candidate_rows(
        nn_cv_results,
        None,
        "mlp",
        "MLP",
        "representation",
        ["name", "hidden", "dropout", "learning_rate", "batch_size", "epochs"],
    )
    candidates += _candidate_rows(
        ensemble_cv_results,
        None,
        "ensemble",
        "algorithm",
        "representation",
        ["C", "name", "hidden", "dropout", "learning_rate", "batch_size", "epochs", "max_features"],
    )
    candidates += _candidate_rows(
        classical_cv_results,
        "binary",
        "classical",
        "algorithm",
        "representation",
        ["C", "class_weight"],
    )
    candidates += _candidate_rows(
        bert_cv_results.assign(dataset="binary", representation="bert_base_finetuned", algorithm="BertForSequenceClassification"),
        "binary",
        "bert_finetune",
        "algorithm",
        "representation",
        ["epoch"],
    )

    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        raise RuntimeError("No se encontraron candidatos con métricas de validación para exportar.")
    candidate_frame = candidate_frame[
        (candidate_frame["representation"] != "roberta")
        & (candidate_frame["algorithm"] != "LinearSVC")
    ].copy()
    candidate_frame = candidate_frame.sort_values(
        ["cv_accuracy_mean", "cv_macro_f1_mean"],
        ascending=False,
    ).copy()
    candidate_frame["rank"] = candidate_frame.groupby("dataset", sort=False).cumcount() + 1
    candidate_frame = candidate_frame[candidate_frame["rank"] <= top_n].reset_index(drop=True)

    output_rows = []
    for _, candidate_series in candidate_frame.iterrows():
        candidate = candidate_series.to_dict()
        parameters = candidate["parameters"]
        artifact_name = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            f"{candidate['rank']:02d}_{candidate['dataset']}_{candidate['algorithm']}_{candidate['representation']}",
        )[:180] + ".pt"
        payload, test_accuracy, test_macro_f1 = _fit_candidate_for_export(candidate, preprocess, output_dir)
        artifact_path = model_dir / artifact_name
        torch.save(payload, artifact_path)
        output_rows.append({
            "rank": candidate["rank"],
            "dataset": candidate["dataset"],
            "representation": candidate["representation"],
            "text_representation_configuration": _representation_description(candidate["representation"], candidate["max_features"]),
            "model_family": candidate["model_family"],
            "algorithm": candidate["algorithm"],
            "max_features": candidate["max_features"],
            "hyperparameters": json.dumps(parameters, ensure_ascii=False, default=str, sort_keys=True),
            "cv_accuracy_mean": candidate["cv_accuracy_mean"],
            "cv_accuracy_std": candidate["cv_accuracy_std"],
            "cv_macro_f1_mean": candidate["cv_macro_f1_mean"],
            "cv_macro_f1_std": candidate["cv_macro_f1_std"],
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "model_file": str(artifact_path.relative_to(root)),
        })

    model_table = pd.DataFrame(output_rows)
    model_table.to_csv(output_dir / "top20_models.csv", index=False)
    experiment_summary = {
        "vector_manifest": vector_manifest,
        "hyperparameter_results": hyperparameter_results,
        "embedding_results": embedding_results,
        "neural_network_cv": nn_cv_results,
        "neural_network_test": nn_test_results,
        "ensemble_cv": ensemble_cv_results,
        "ensemble_test": ensemble_test_results,
        "classical_cv": classical_cv_results,
        "classical_test": classical_test_results,
        "bert_cv": bert_cv_results,
        "bert_test": bert_test_results,
    }
    return experiment_summary, model_table
