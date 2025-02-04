if (!require("readxl")) install.packages("readxl")          
if (!require("tm")) install.packages("tm")          
if (!require("topicmodels")) install.packages("topicmodels")    
if (!require("tidyverse")) install.packages("tidyverse")      
if (!require("tidytext")) install.packages("tidytext")      
if (!require("reshape2")) install.packages("reshape2") 

library(readxl)
library(tm)
library(topicmodels)
library(tidyverse)
library(tidytext)
library(reshape2)

data <- read_xlsx("News.xlsx")
news <- data$details 
top_word_count <- 8
topic_number <- 12

corpus <- Corpus(VectorSource(news))
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.95)
non_empty_docs <- rowSums(as.matrix(dtm)) > 0
dtm <- dtm[non_empty_docs, ]

tf_idf <- weightTfIdf(dtm)

lda_model <- LDA(dtm, k = topic_number, control = list(seed = 123))

topics <- tidy(lda_model, matrix = "beta") %>%
          group_by(topic) %>%
          slice_max(beta, n = top_word_count, with_ties = FALSE) %>% 
          ungroup()

topic_proportions <- tidy(lda_model, matrix = "gamma") %>%
                     mutate(document = as.numeric(document))

p_beta <- topics %>%
  ggplot(aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(
    title = "Top Terms for Each Topic",
    x = "Terms",
    y = "Beta"
  )

p_gamma <- topic_proportions %>%
  ggplot(aes(x = factor(topic), y = gamma, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ document, scales = "free") +
  labs(
    title = "Topic Proportions for Each Document",
    x = "Topics",
    y = "Proportion (Gamma)"
  )

print(p_gamma)
print(p_beta)

ggsave(
  filename = "topic_proportions.png",
  plot = p_gamma,
  width = 20,   
  height = 10, 
  dpi = 400    
)
ggsave(
  filename = "top_terms.png",
  plot = p_beta,
  width = 12,  
  height = 8, 
  dpi = 400     
)
