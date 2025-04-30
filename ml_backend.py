import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.svm import SVC
import numpy as np


nlp = spacy.load('de_core_news_md')

df_sonst = pd.read_csv("sonstiges.csv", sep=";", encoding="utf-8")
df_classified = pd.read_csv("random_sample.csv", sep=";", encoding="utf-8")

df_sonst_filtered = df_sonst[~df_sonst['id'].isin(df_classified['id'])]
df_sonst_filtered = df_sonst_filtered.dropna(subset=["further_remarks"])


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


df_sonst_filtered["lemma_remarks"] = df_sonst_filtered["further_remarks"].apply(spacy_tokenizer)

# Flatten the list of tokens in "lemma_remarks" and count occurrences
all_tokens = [token for tokens in df_sonst_filtered["lemma_remarks"] for token in tokens]
word_counts = Counter(all_tokens)

# Sort by frequency in descending order
sorted_word_counts = word_counts.most_common()

# Print the most common words
print("Most common words in lemma_remarks:")
for word, count in sorted_word_counts:
    if count > 10:
        print(f"{word}: {count}")

keywords = [word for word, count in word_counts.items() if count > 14]

filtered_rows = df_sonst_filtered[
    df_sonst_filtered["lemma_remarks"].apply(lambda tokens: any(keyword in tokens for keyword in keywords))
]

print(len(filtered_rows))

# Print the filtered rows
# for index, row in filtered_rows.iterrows():
#     print(f"ID: {row['id']}, Further Remarks: {row['further_remarks']}")

all_remarks = df_sonst_filtered["further_remarks"].dropna().astype(str).tolist()

# Count occurrences of each string
remarks_counts = Counter(all_remarks)

# Sort by frequency in descending order
sorted_remarks_counts = remarks_counts.most_common()

# Print the most common strings
# print("Most common strings in further_remarks:")
# for remark, count in sorted_remarks_counts:  # Adjust the number to show more or fewer results
#     print(f"{remark}: {count}")

rows_with_more_than_10 = df_sonst_filtered[
    df_sonst_filtered["further_remarks"].isin(
        [remark for remark, count in remarks_counts.items() if count > 10]
    )
]

# Print the filtered rows
print("Rows with 'further_remarks' occurring more than 10 times:")
print(len(rows_with_more_than_10))


bow_vector = CountVectorizer(
    tokenizer=spacy_tokenizer,
    ngram_range=(1, 1)
)


tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer, token_pattern=None)

# classifier = LogisticRegression(multi_class='ovr', solver='liblinear')
classifier = LogisticRegression(
    solver='newton-cg',
    warm_start=True,
    max_iter=500,
    n_jobs=-1,
)

svc = SVC(kernel='linear', probability=True, random_state=42)

pipe = Pipeline([  # ('vectorizer', bow_vector),
                 ('classifier', classifier)])


df_classified["further_remarks"] = df_classified["further_remarks"].astype(str)
df_classified["activity_id"] = df_classified["activity_id"].astype(str)


dense_x_df = pd.DataFrame(
    df_classified["further_remarks"].apply(
        lambda sentence: (sentence, get_dense_vector_without_stopwords(sentence))
    ).tolist(),
    columns=["sentence", "dense_vector"]
)
dense_y = df_classified["activity_id"].tolist()


x_train, x_test, y_train, y_test = train_test_split(
    dense_x_df,
    dense_y,
    test_size=0.2
)

pipe.fit(x_train["dense_vector"].to_list(), y_train)

predicted = pipe.predict_proba(x_test["dense_vector"].to_list())

for i, (sentence, category, prob) in enumerate(zip(x_test["sentence"], y_test, predicted)):
    print(f"Sample {i+1}:")
    print(f"Predicted Category: {np.argmax(prob)}")
    print(f"True Category: {category}")
    print(f"Sentence: {sentence}")
    print(f"Class 0 Probability: {prob[0]:.4f}")
    print("-" * 50)

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