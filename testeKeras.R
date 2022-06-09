library(reticulate)
library(purrr)
library(text2vec) 
library(dplyr)
library(Rtsne)
library(ggplot2)
library(plotly)
library(stringr)
library("keras")
source("myFyunctions.R")

load(url("https://cbail.github.io/Elected_Official_Tweets.Rdata"))
elected_no_retweets <- elected_official_tweets %>%
  filter(is_retweet == F) %>%
  select(c("screen_name", "text"))


# We want to use original tweets, not retweets:
elected_no_retweets <- elected_official_tweets %>%
  filter(is_retweet == F) %>%
  select(c("screen_name", "text"))

# Many tweets contain URLs, which we don't want considered in the model:
elected_no_retweets$text <- str_replace_all(string = elected_no_retweets$text,
                                            pattern = "https.+",
                                            replacement = "")



tokenizer <- text_tokenizer(num_words = 20000)
tokenizer %>% fit_text_tokenizer(elected_no_retweets$text)




# Number of Dimensions in the embedding vector.
embedding_size <- 128  
# Size of context window
skip_window <- 5       
# Number of negative examples to sample for each word.
num_sampled <- 1       
input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)


embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, 
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)

output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)

model %>% compile(loss = "binary_crossentropy", optimizer = "rmsprop")
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)



model %>%
  fit(
    skipgrams_generator(elected_no_retweets$text, tokenizer, skip_window, negative_samples), 
    steps_per_epoch = 100, epochs = 4
  )


embedding_matrix <- get_weights(model)[[1]]

words <- data_frame(
  word = names(tokenizer$word_index), 
  id = as.integer(unlist(tokenizer$word_index))
)


words <- words %>%
  filter(id <= tokenizer$num_words) %>%
  arrange(id)


row.names(data.frame(embedding_matrix)) <- c("UNK", words$word)


find_similar_words <- function(word, embedding_matrix, n = 5) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}

find_similar_words("republican", embedding_matrix)


tsne <- Rtsne(embedding_matrix[2:500,], perplexity = 50, pca = FALSE)

tsne_plot <- tsne$Y %>%
  as.data.frame() %>%
  mutate(word = row.names(embedding_matrix)[2:500]) %>%
  ggplot(aes(x = V1, y = V2, label = word)) + 
  geom_text(size = 3)