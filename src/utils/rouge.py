import re

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def get_rouge_score_for_sum(split_gold_summary, split_proposed_summary):
    split_gold_summary = [(re.sub(r'[^\w\s]', '', i)).lower() for i in split_gold_summary]
    split_proposed_summary = [(re.sub(r'[^\w\s]', '', i)).lower() for i in split_proposed_summary]

    gold_sum_1_gram = [_get_ngrams(1, i.split()) for i in split_gold_summary]
    if gold_sum_1_gram:
      gold_sum_1_gram = set.union(*map(set, gold_sum_1_gram))

    gold_sum_2_gram = [_get_ngrams(2, i.split()) for i in split_gold_summary]
    if gold_sum_2_gram:
      gold_sum_2_gram = set.union(*map(set, gold_sum_2_gram))

    proposed_sum_1_gram = [_get_ngrams(1, i.split()) for i in split_proposed_summary]
    if proposed_sum_1_gram:
      proposed_sum_1_gram = set.union(*map(set, proposed_sum_1_gram))

    proposed_sum_2_gram = [_get_ngrams(2, i.split()) for i in split_proposed_summary]
    if proposed_sum_2_gram:
      proposed_sum_2_gram = set.union(*map(set, proposed_sum_2_gram))

    rouge_1_f = cal_rouge(gold_sum_1_gram, proposed_sum_1_gram)["f"]
    rouge_2_f = cal_rouge(gold_sum_2_gram, proposed_sum_2_gram)["f"]
    
    return rouge_1_f, rouge_2_f
    