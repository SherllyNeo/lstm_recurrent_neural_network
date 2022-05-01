from lstm_rnn import text_generator


path_to_text = "./text.txt"
start_string = "a"
predicatbility = 1
num_generate = 50
nlp_ai = text_generator("text.txt")
text_gen = nlp_ai.main(start_string,predicatbility,num_generate)
print(f"\n \n GENERATED TEXT: \n \n {text_gen} \n")
