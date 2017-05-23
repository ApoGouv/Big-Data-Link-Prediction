# Submit #45

## Scored: 0.97115,

## Classifier used:
    classifier = RF(n_jobs=1, n_estimators=500, criterion="entropy", max_features="log2", max_depth=10)

## Features used:
- **# Calculate feature #1 - number of overlapping words in title**
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
- **# Calculate feature #2 - temporal distance (time) between the papers**
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
- **# Calculate feature #3 - number of common authors**
    *we also cleared authors from parentheses*
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
- **# Calculate feature #4 - number of common words in journal**
    comm_journ_test.append(len(set(source_journal).intersection(set(target_journal))))
- **# Calculate feature #5 - number of common abstract words**
    comm_abstr_test.append(len(set(source_abstr).intersection(set(target_abstr))))
- **# Calculate feature #6a - abstract cosine similarity**
    cos_sim_abstract.append(
        cosine_similarity(features_TFIDF_Abstract[index_source], features_TFIDF_Abstract[index_target]))
- **# Calculate feature #6b - title cosine similarity**
    cos_sim_title.append(cosine_similarity(features_TFIDF_Title[index_source], features_TFIDF_Title[index_target]))
- **# Calculate feature #6c - author cosine similarity**
    cos_sim_author.append(cosine_similarity(features_TFIDF_Author[index_source], features_TFIDF_Author[index_target]))
- **# Calculate feature #6d - journal cosine similarity**
    cos_sim_journal.append(
        cosine_similarity(features_TFIDF_Journal[index_source], features_TFIDF_Journal[index_target]))
- **# Calculate feature #7: common neighbours**
    com_neigh.append(len(gAdjList[index_source].intersection(gAdjList[index_target])))
- **# Calculate feature #8: preferential attachment**
    pref_attach.append(int(degrees[index_source] * degrees[index_target]))