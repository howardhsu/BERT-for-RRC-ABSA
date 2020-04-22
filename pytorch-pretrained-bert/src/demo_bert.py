import torch

from transformers import *
# from pytorch_pretrained_bert.modeling import *

texts = [
    "bill [MASK] is the president of united states . His wife is [MASK] clinton",
    "this is a beautiful [MASK], its [MASK] is clean .", 
    "I love this [MASK] , its [MASK] is fast .",
    "The [MASK] is good , given its [MASK] is not that good .",
    "I like this [MASK] , its super [MASK] and easy to [MASK] . But its [MASK] is not [MASK] .",
    "the keyboard is great, [MASK], [MASK] and [MASK] ."
]

model_path_or_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path_or_name)
model = torch.load("path to model.pt") # MODEL_CLASS.from_pretrained(model_path_or_name)

for text in texts:
    tokens = tokenizer.tokenize(text)

    input_ids = torch.tensor([tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens))])

    with torch.no_grad():
        output_prediction = model(input_ids)[0].argmax(dim=-1)            
        # check output based on different types of predictions.
        # output_tokens = tokenizer.convert_ids_to_tokens(output_prediction[0])
        # print(" ".join(output_tokens))

    print("\n")