import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


nlp = spacy.load('de_core_news_md')

# df_sonst = pd.read_csv("sonstiges.csv", sep=";", encoding="utf-8")
# df_classified = pd.read_csv("random_sample.csv", sep=";", encoding="utf-8")

# df_sonst_filtered = df_sonst[~df_sonst['id'].isin(df_classified['id'])]
# df_sonst_filtered = df_sonst_filtered.dropna(subset=["further_remarks"])


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)
    mytokens = [tok.lemma_.lower().strip() for tok in mytokens if tok.lemma_ != '-PRON-' and not tok.is_stop and not tok.is_punct and not tok.is_space]
    return mytokens


def spacy_tokenizer_text(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = spacy_tokenizer(sentence)
    return " ".join(mytokens)


def get_dense_vector_without_stopwords(sentence):
    doc = nlp(sentence)
    filtered_tokens = [token.vector for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    if not filtered_tokens:
        return np.zeros(doc.vector.shape)
    return np.mean(filtered_tokens, axis=0)


def descriptive_statistics(df):
    ret = {
    # Count the number of unique values in the 'further_remarks' column
        'unique_values': df['further_remarks'].nunique(),

        # Count the number of rows in the DataFrame
        'total_rows': len(df)
    }

    df["lemma_remarks"] = df["further_remarks"].apply(spacy_tokenizer)

    # Count occurrences of each word in the 'lemma_remarks' column
    word_counts = Counter()
    for tokens in df["lemma_remarks"]:
        word_counts.update(tokens)
    # Sort by frequency in descending order
    word_counts = dict(
        sorted(
            word_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    ret['keywords'] = [word for word, count in word_counts.items() if count > 14]

    ret['filtered_rows'] = df[
        df["lemma_remarks"].apply(
            lambda tokens: any(
                keyword in tokens for keyword in ret['keywords']
            )
        )
    ]

    all_remarks = df["further_remarks"].dropna().astype(str).tolist()

    # Count occurrences of each string
    remarks_counts = Counter(all_remarks)
    print(remarks_counts)

    # Sort by frequency in descending order
    ret['sorted_remarks_counts'] = remarks_counts.most_common(10)

    ret['rows_with_more_than_10'] = df[
        df["further_remarks"].isin(
            [remark for remark, count in remarks_counts.items() if count > 10]
        )
    ]
    print(ret['rows_with_more_than_10'])

    return ret


def train_model(df):
    tfidf_vector = TfidfVectorizer(
         tokenizer=spacy_tokenizer,
         token_pattern=None,
         lowercase=True,
         norm='l2',
         use_idf=True,
         smooth_idf=True,
    )

    # classifier = LogisticRegression(multi_class='ovr', solver='liblinear')
    # classifier = LogisticRegression(
    #     solver='newton-cg',
    #     warm_start=True,
    #     max_iter=500,
    #     n_jobs=-1,
    # )

    classifier = MultinomialNB(
        alpha=1.0,
        fit_prior=True,
        class_prior=None
    )

    pipe = Pipeline(
        [   ('vectorizer', tfidf_vector),
            ('classifier', classifier)
        ]
    )

    df["further_remarks"] = df["further_remarks"].fillna("").astype(str)
    # df["further_remarks"] = df["further_remarks"].astype(str)
    # df["id"] = df["id"]


    dense_x_df = pd.DataFrame(
        df["further_remarks"].apply(
            lambda sentence: (sentence, get_dense_vector_without_stopwords(sentence))
        ).tolist(),
        columns=["sentence", "dense_vector"]
    )
    dense_y = df["category"].tolist()

    x = df["further_remarks"].tolist()
    y = df["category"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2
    )
    print(y_train)

    pipe.fit(x_train, y_train)

    predicted = pipe.predict_proba(x_test)
    y_pred = pipe.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("Predicted:", predicted)
    print("Predicted shape:", predicted.shape)
    print("Predicted argmax:", np.argmax(predicted, axis=1))

    ret = []
    for i, (sentence, category, prob) in enumerate(zip(x_test, y_test, predicted)):
        ret.append(
            {
                "sentence": sentence,
                "assign_cat": category,
                "predicted_cat": np.argmax(prob),
                "prob": prob[0],
            }
        )
    return ret

# print("Logistic Regression Accuracy:",
#       metrics.accuracy_score(y_test, predicted))
# print(
#     "Logistic Regression Precision:",
#     metrics.precision_score(
#         y_test,
#         predicted,
#         average='macro',
#         zero_division=0
#     )
# )
# print(
#     "Logistic Regression Recall:",
#     metrics.recall_score(
#         y_test,
#         predicted,
#         average='macro',
#         zero_division=0
#     )
# )

# predicted = pipe.predict(df_sonst_filtered["further_remarks"])
# df_sonst_filtered["activity_id"] = predicted
# print(df_sonst_filtered)

# df_sonst_filtered.to_csv(
#     "sonstiges_classified.csv",
#     sep=";",
#     index=False,
#     encoding="utf-8"
# )