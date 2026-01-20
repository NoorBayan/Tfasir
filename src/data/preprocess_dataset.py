from datasets import load_dataset 
import re
import sys

def get_raw_data(path="ccdv/arxiv-summarization", extension=None):
  if (extension == None):
    dataset = load_dataset(path)
  elif (extension == "json" or extension=="csv" or extension=="parquet"):
    data_files = {
      "train": path + "/train." + extension,
      "validation": path + "/validation." + extension,
      "test": path + "/test." + extension,
    }

    dataset = load_dataset(extension, data_files=data_files)
  else:
    sys.exit("Invalid file format. Please use json, csv or parquet files.")

  return dataset

def split_into_sentences(text, alphabets, prefixes, suffixes, starters, acronyms, websites, digits):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def split_raw_data_into_sentences(raw_data, max_no_of_sentences):
  alphabets= "([A-Za-z])"
  prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
  suffixes = "(Inc|Ltd|Jr|Sr|Co)"
  starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
  acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
  websites = "[.](com|net|org|io|gov)"
  digits = "([0-9])"

  split_raw_data = []

  for data in raw_data:
    article_split = split_into_sentences(data["article"], alphabets, prefixes, suffixes, starters, acronyms, websites, digits)
    abstract_split = split_into_sentences(data["abstract"], alphabets, prefixes, suffixes, starters, acronyms, websites, digits)
     
    if (len(article_split) < max_no_of_sentences):
      for i in range(len(article_split), max_no_of_sentences):
        article_split.append("")
    else:
      article_split = article_split[0:max_no_of_sentences]
      
    if (len(abstract_split) < max_no_of_sentences):
      for i in range(len(abstract_split), max_no_of_sentences):
        abstract_split.append("")
    else:
      abstract_split = abstract_split[0:max_no_of_sentences]
    
    split_raw_data_item = {}
    split_raw_data_item["article"] = article_split
    split_raw_data_item["abstract"] = abstract_split
    split_raw_data.append(split_raw_data_item)

  return split_raw_data


