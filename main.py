from lstm_rnn import text_generator


path_to_text = "./text.txt"
start_string = "how dare thee"
predicatbility = 0.5
num_generate = 1000
nlp_ai = text_generator(path_to_text)
text_gen = nlp_ai.main(start_string,predicatbility,num_generate)
print(f"\n \n GENERATED TEXT: \n \n {text_gen} \n")
