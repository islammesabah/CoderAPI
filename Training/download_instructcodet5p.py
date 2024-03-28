from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = 'Salesforce/instructcodet5p-16b'

# save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./saved_models/Salesforce/instructcodet5p-16b", from_pt=True)

# save model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.tie_weights()
model.save_pretrained("./saved_models/Salesforce/instructcodet5p-16b", from_pt=True)
