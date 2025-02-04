if (!require("rvest")) install.packages("rvest")          
if (!require("dplyr")) install.packages("dplyr")          
if (!require("hunspell")) install.packages("hunspell")    
if (!require("stringr")) install.packages("stringr")      
if (!require("stringi")) install.packages("stringi")      
if (!require("purrr")) install.packages("purrr")          
if (!require("textclean")) install.packages("textclean")  
if (!require("stopwords")) install.packages("stopwords")  
if (!require("textstem")) install.packages("textstem")    
if (!require("writexl")) install.packages("writexl")    

library(rvest)       
library(dplyr)       
library(hunspell)    
library(stringr)     
library(stringi)     
library(purrr)       
library(stopwords)   
library(textstem)    
library(writexl)
library(textclean)

clean_text <- function(text) {
  cleaned_text <- textclean::replace_html(text) 
  cleaned_text <- stringi::stri_trans_general(cleaned_text, "latin-ascii")
  cleaned_text <- stringr::str_replace_all(cleaned_text, "\\s+", " ") 
  cleaned_text <- stringr::str_replace_all(cleaned_text, "\\d+", "") 
  cleaned_text <- stringr::str_trim(cleaned_text) 
  return(cleaned_text)
}

NLPpreprocess_text <- function(text) {
  text <- clean_text(text) 
  text <- stringr::str_replace_all(text, "[[:punct:]]", "") 
  tokens <- unlist(strsplit(text, "\\s+"))          
  tokens <- tolower(tokens)                       
  stopwords <- tolower(stop_words$word)                     
  stopwords <- stringr::str_replace_all(stopwords, "[[:punct:]]", "")
  tokens <- tokens[!(tokens %in% stopwords)]
  tokens <- textstem::lemmatize_words(tokens)
  tokens <- textclean::replace_contraction(tokens)
  tokens <- textclean::replace_emoji(tokens)
  tokens <- textclean::replace_emoticon(tokens)
  
  tokens <- purrr::map_chr(tokens, ~ {
    misspelled <- hunspell::hunspell(.x)
    if (length(misspelled[[1]]) > 0) {
      suggestions <- hunspell::hunspell_suggest(misspelled[[1]])
      if (length(suggestions[[1]]) > 0) {
        return(suggestions[[1]][1])
      } else {
        return(.x)
      }
    } else {
      return(.x)
    }
  })
  tokens <- stringi::stri_trans_general(tokens, "latin-ascii") 
  tokens <- tokens[tokens != ""]                           
  preprocessed_text <- paste(tokens, collapse = " ")  
  return(preprocessed_text)
}

link <- "https://edition.cnn.com/world"
newsLinks <- read_html(link) %>% 
              html_nodes("a.container__link.container__link--type-article.container_lead-plus-headlines__link") %>% 
              html_attr("href") %>% 
              paste("https://edition.cnn.com", ., sep="")
newsLinks <- unique(newsLinks)

results_df <- data.frame(link = character(), details = character(), stringsAsFactors = FALSE)
for (links in newsLinks) {
  details <- read_html(links) %>% 
             html_nodes(".vossi-paragraph") %>% 
             rvest::html_text() %>%
             purrr::map_chr(NLPpreprocess_text) %>%
             paste(collapse = " ") 
  
  results_df <- rbind(results_df, data.frame(link = links, 
                                            details = details, 
                                            stringsAsFactors = FALSE))
}

write_xlsx(results_df, "News.xlsx")
getwd()