from datetime import datetime
import json
import os

from OpenAIAPI import OpenAIAPI

OUTPUT_DIRECTORY = 'outputs'

def cache_gpt_outputs(
    output_dir = OUTPUT_DIRECTORY,
):
    with open("datasets/dataset_v3.json", "r") as f:
        content = f.read()
    dataset = json.loads(content)
    
    oai = OpenAIAPI()
    prompt_template = "Context: {}\n\nSentence: {}\n\nIs the sentence supported by the context above? Answer Yes or No: "

    for idx in range(len(dataset)):
        sentences = dataset[idx]['gpt3_sentences']
        text_samples = dataset[idx]['gpt3_text_samples']
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(text_samples):
                outpath = "{}/{}_{}_{}.txt".format(output_dir, idx, sent_i, sample_i)
                exist = os.path.isfile(outpath)
                if exist:
                    print("idx {} - sentence {} sample {}: already exists".format(idx, sent_i, sample_i))
                    continue

                prompt = prompt_template.format(sample.replace("\n", " "), sentence)
                
                messages= [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                
                _, nl_response = oai.request(messages=messages)
                
                print("ChatGPT-3.5-turbo:", nl_response)
                with open(outpath, "w") as f:
                    f.write(nl_response)
                print("[{}] {} wrote: {}".format(str(datetime.now()), idx, outpath))
                
if __name__ == '__main__':
    cache_gpt_outputs()
    