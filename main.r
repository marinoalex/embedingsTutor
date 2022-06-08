library(tidytext)
library(dplyr)
library(widyr)
library(irlba)
library(broom)
source("myFyunctions.R")


load(url("https://cbail.github.io/Elected_Official_Tweets.Rdata"))
elected_no_retweets <- elected_official_tweets %>%
  filter(is_retweet == F) %>%
  select(c("text"))

#create tweet id
elected_no_retweets$postID<-row.names(elected_no_retweets)
head(elected_no_retweets)

#create context window with length 8
tidy_skipgrams <- elected_no_retweets %>%
  unnest_tokens(ngram, text, token = "ngrams", n = 8) %>%
  mutate(ngramID = row_number()) %>% 
  tidyr::unite(skipgramID, postID, ngramID) %>%
  unnest_tokens(word, ngram)

#calculate unigram probabilities (used to normalize skipgram probabilities later)
unigram_probs <- elected_no_retweets %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  mutate(p = n / sum(n))


#calculate probabilities
skipgram_probs <- tidy_skipgrams %>%
  pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>%
  dplyr::mutate(p = n / sum(n))

#normalize probabilities
normalized_prob <- skipgram_probs %>%
  filter(n > 20) %>%
  rename(word1 = item1, word2 = item2) %>%
  left_join(unigram_probs %>%
              select(word1 = word, p1 = p),
            by = "word1") %>%
  left_join(unigram_probs %>%
              select(word2 = word, p2 = p),
            by = "word2") %>%
  mutate(p_together = p / p1 / p2)


normalized_prob[2005:2010,]


normalized_prob %>% 
  filter(word1 == "trump") %>%
  arrange(-p_together)

pmi_matrix <- normalized_prob %>%
  mutate(pmi = log10(p_together)) %>%
  cast_sparse(word1, word2, pmi)

library(irlba)

pmi_matrix@x[is.na(pmi_matrix@x)] <- 0
#run SVD
pmi_svd <- irlba(pmi_matrix, 256, maxit = 500)
#next we output the word vectors:
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)

pres_synonym <- search_synonyms(word_vectors,word_vectors["president",])

pres_synonym