# Submit #23

## Scored: 0.86713,

## Classifier used:
    classifier = RF(n_jobs=1, n_estimators=200, max_features=5, max_depth=10, min_samples_leaf=100)

## Features used:
- **# Calculate feature #1 - number of overlapping words in title**
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
- **# Calculate feature #2 - temporal distance (time) between the papers**
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
- **# Calculate feature #3 - number of common authors**
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
- **# Calculate feature #4 - number of common words in journal**
    comm_journ_test.append(len(set(source_journal).intersection(set(target_journal))))
- **# Calculate feature #5 - number of common abstract words**
    comm_abstr_test.append(len(set(source_abstr).intersection(set(target_abstr))))
- **# Calculate feature #6 - cosine similarity**
    cos_sim_test.append(cosine_similarity(features_TFIDF[index_source], features_TFIDF[index_target]))