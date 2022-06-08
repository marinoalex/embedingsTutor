library(reticulate)
library(purrr)
library(text2vec) 
library(dplyr)
library(Rtsne)
library(ggplot2)
library(plotly)
library(stringr)



elected_no_retweets <- elected_official_tweets %>%
  filter(is_retweet == F) %>%
  select(c("screen_name", "text"))


elected_no_retweets$text <- str_replace_all(string = elected_no_retweets$text,
                                            pattern = "https.+",
                                            replacement = "")


tokenizer <- text_tokenizer(num_words = 20000)

tokenizer %>% fit_text_tokenizer(elected_no_retweets$text)

embedding_size <- 128  

skip_window <- 5  
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
