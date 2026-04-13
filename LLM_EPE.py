# LLM_EPE.py

import os
from openai import AzureOpenAI
import pandas as pd
import json


#conda activate env_01

# Available models
# o3-mini
# gpt-4o-mini-audio-preview
# gpt-4o-mini-realtime-preview
# o1
# o1-mini
# gpt-4o
# gpt-4o-mini
# gpt-4o-audio-preview
# gpt-4o-realtime-preview
# o1-preview
# gpt-4
# gpt-4-32k
# text-embedding-3-large
# text-embedding-3-small
# tts
# tts-hd
# whisper
# dall-e-3
# dall-e-2
# text-embedding-ada-002
# davinci-002
# gpt-35-turbo-16k
# gpt-35-turbo-instruct
# gpt-35-turbo
# babbage-002

def send_prompt(prompt, model):
    messages = prompt

    completion = client.chat.completions.create(  
    model=model,  
    messages=messages,
    max_tokens=800,  
    temperature=0,
    seed=123,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False  
    )

    # use for o3-mini
    # completion = client.chat.completions.create(  
    # model=model,  
    # messages=messages,
    # max_completion_tokens=100000,
    # seed=123,
    # stop=None,  
    # stream=False  
    # )

    return completion.choices[0].message.content


def clean_prompt(model, report):
    clean_report_system_prompt = {"role":"system","content":"You are an AI assitant helping to remove certain data from a radiology MRI text report. You will be asked to remove certain data from the following radiology report:"}
    
    report_input = {"role":"user", "content":f"{report}"}

    clean_report_user_prompt = {"role": "user", "content":"Clean the provided report by performing the following operations: \n1. In the body of the report, only remove the text 'with visualized gross EPE','without visualized gross EPE','suspicion for EPE', 'with gross EPE', 'without gross EPE'. \n2. Remove ALL text following the word **IMPRESSION** (inclusive of the word 'IMPRESSION') until the end of the report.\n3. Return only the cleaned report content in your output, without any additional commentary or formatting.\n# Output Format\n- A plain text version of the cleaned report, containing only the modified text without any formatting or extra explanations."}

    messages = [clean_report_system_prompt, report_input, clean_report_user_prompt]
    
    completion = client.chat.completions.create(  
    model=model,  
    messages=messages,
    max_tokens=800,  
    temperature=0,
    seed=123,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False
    )  

    # use for o3-mini
    # completion = client.chat.completions.create(  
    # model=model,  
    # messages=messages,
    # max_completion_tokens=100000,
    # seed=123,
    # stop=None,  
    # stream=False  
    # )

    return completion.choices[0].message.content


def classify_report(model,report):
    # System prompt matches that used in the LLM multi-shot learning function

    classify_report_system_prompt = {"role":"system","content":"You are an AI assistant helping an abdominal radiologist who is reviewing prostate MRI scans for patients with prostate cancer. You will be provided a text report from a prostate MRI, and will be asked to analyze this radiology report:"}
   
    report_input = {"role":"user", "content":f"{report}"}

    output_format_structure = {'Mehralivand Radiologic MRI EPE Scoring System':'[Calculate the mEPE score]','PI-RADS score':'[Calculate a PI-RADS score]','EPE_status': 'present" or "absent','justification':'[Brief explanation summarizing the key findings from the radiology report that support the determination]', 'keywords':'[comma separated list of keywords suggesting presence or absence of EPE]','confidence_level':'[confidence level ranging from 0 to 100%]'}
    json_string = json.dumps(output_format_structure)

    classify_report_user_prompt = {"role":"user", "content":f"Determine if extraprostatic extension (EPE) is present or absent in the provided radiology report by analyzing the objective data described. Focus on key findings that directly indicate the presence or absence of EPE, such as explicit mentions of capsular involvement, seminal vesicle invasion, or protrusions beyond the prostate. Steps:\n1. Review the radiology report in detail. \n2. Identify any explicit mention of extraprostatic extension (EPE) or descriptors suggesting it, including phrases such as 'capsular abutment', 'capsular invasion', 'posterior protrusion', or 'seminal vesicle involvement', 'capsular bulging', 'capsular asymmetry', 'capsular irregularity', and more phrases as described in the published literature.   \n3. Based on the findings in the prostate MRI report, determine if EPE is present or absent. \n4. Avoid injecting subjective interpretations beyond the available data. \n5. Utilize the PI-RADS score in the decision making process. \n#Output Format\n{json_string}"}


    messages = [classify_report_system_prompt, report_input, classify_report_user_prompt]
    
    completion = client.chat.completions.create(  
    model=model,  
    messages=messages,
    max_tokens=800,  
    temperature=0,
    seed=123,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False  
    )

    # use for o3-mini
    # completion = client.chat.completions.create(  
    # model=model,  
    # messages=messages,
    # max_completion_tokens=100000,
    # seed=123,
    # stop=None,  
    # stream=False  
    # )

    return completion.choices[0].message.content


def multishotlearn_llm(model, input_file):
    # This function is used to perform mult-shot learning with the LLM
    # 5 positive cases, and 5 negative cases are fed into the LLM
    # feed in same system prompt every time to maintain state

    learning_report_system_prompt = {"role":"system","content":"You are an AI assistant helping an abdominal radiologist who is reviewing prostate MRI scans for patients with prostate cancer. You will be provided a text report from a prostate MRI, and will be asked to analyze this radiology report:"}
        
    
    learning_positive_case = {"role":"user", "content":"The following report is an example of a MRI report for a patient who was POSITIVE for extraprostatic extension on their prostatectomy pathology specimen."}
    learning_negative_case = {"role":"user", "content":"The following report is an example of a MRI report for a patient who was NEGATIVE for extraprostatic extension on their prostatectomy pathology specimen."}
    user_prompt2 = {"role":"user", "content":" Use the given report as a learning example, and identify features in the report suggesting the presence or absence of EPE. No response is needed."}

    with open(input_file, 'a') as f:
        #Currently only csv files are supported
        fname = input_file
        if fname[-3:]=='csv':
            df = pd.read_csv(fname)
            reports = df['report'].to_list()
            EPE_list = df['EPE'].to_list()
            i=0
            for report in reports:
                response=''
                report_input = {"role":"user", "content":f"{report}"}
                #print(EPE_list[i])
                if EPE_list[i] == 1:
                    # EPE positive case
                    messages = [learning_report_system_prompt, learning_positive_case, report_input, user_prompt2]
                    response = send_prompt(messages, model)
                elif EPE_list[i] == 0:
                    # EPE negative case
                    messages = [learning_report_system_prompt, learning_negative_case, report_input]
                    response = send_prompt(messages, model)
                i=i+1
            return 1
        else:
            print("Error, file is not a CSV")
            return 0

#decorator to avoid division by zero
def test(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except: pass
    return inner

@test
def stats_divide(num_val, denom_val):
    return num_val/denom_val



if __name__ == '__main__':

    # Initialize Azure OpenAI client with key-based authentication

    #version = "2024-05-01-preview" #used for gpt-4o mini, 35 turbo, 4o, and 4
    version = "2024-12-01-preview" #used for o3 mini

    #user defined Microsoft Azure endpoint for connection
    endpoint_url = "https//My-end-point.com"
    endpoint = os.getenv("ENDPOINT_URL", endpoint_url)  
    
    # change deployment name as needed
    #model = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini") 
    #model = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")
    #model = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    model = os.getenv("DEPLOYMENT_NAME", "gpt-4")
    #model = os.getenv("DEPLOYMENT_NAME", "o3-mini")
    #model = os.getenv("DEPLOYMENT_NAME", "o1")

    
    
    


    #Open data files
    #Currently only csv files are supported
    training_file = 'prostatectomy_data_training.csv' # 10 learning examples
    data_file = 'Prostatectomy_full.csv' #all cases

    #Iteratre over rows
    #For row, check if patient is valid for use
        # Use case
            # open API
            # Clean report
            # Multi-shot learn
            # classify
            # Store result
            # close API
        # Skip case

    with open(data_file, 'a') as f:
        fname = data_file
        if fname[-3:]=='csv':

            df = pd.read_csv(fname)
            # make sure indexes pair with number of rows
            df = df.reset_index()
            #drop invalid rows
            df = df.drop(df[df.valid <1].index)

            accession_numbers = df['accession'].to_list()
            valid_data = df['valid'].to_list()
            reports = df['report'].to_list()
            EPE_list = df['EPE'].to_list()
            training_list = df['training_data'].to_list()
            classification_list = []
            cleaned_reports = []
            EPE_score = []
            PIRADS_score = []
            justification_list = []
            confidence_list = []
            keywords_list = []

            i=0

            for report in reports:
                
                # open API instance per report for independent results
                client = AzureOpenAI(
                azure_endpoint = endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version= version
                )

                if valid_data[i] == 1 and training_list[i] == 0:
                    # This is a valid case, and not used for multi-shot learning

                    # Clean report
                    clean_report = clean_prompt(model, report)
                    cleaned_reports.append(clean_report)

                    # Multi-shot learn
                    print("Performing Multi-shot learning")
                    response = multishotlearn_llm(model, training_file)

                    # Classify report and store results
                    print("Classifying report for accession: "+str(accession_numbers[i])+"\n")
                    report_classification = classify_report(model, clean_report)
                    my_dict = json.loads(report_classification)
                    #print(my_dict)
                    #print(my_dict["EPE_status"])

                    EPE_score.append(my_dict["Mehralivand Radiologic MRI EPE Scoring System"])
                    PIRADS_score.append(my_dict["PI-RADS score"])
                    justification_list.append(my_dict["justification"])
                    confidence_list.append(my_dict["confidence_level"])
                    keywords_list.append(my_dict["keywords"])

                    if my_dict["EPE_status"] == 'present':
                        classification_list.append('present')
                    elif my_dict["EPE_status"] == 'absent':
                        classification_list.append('absent')
                    else:
                        # error case
                        classification_list.append('error')

                else:
                    # skip case
                    print("skipped")
                    EPE_score.append("-1")
                    PIRADS_score.append("-1")
                    justification_list.append("-1")
                    confidence_list.append("-1")
                    classification_list.append("-1")
                    cleaned_reports.append("-1")
                    keywords_list.append("-1")
                
                i=i+1
                client.close()

        else:
            print("Error, input file is not a CSV")


    # Write results to file
    
    output_df = pd.DataFrame({"accession":accession_numbers, "valid":valid_data, "epe":EPE_list, "llm_classification":classification_list, "epe_score":EPE_score, "pirads":PIRADS_score, "justification":justification_list, "keywords":keywords_list, "confidence":confidence_list, "cleaned_report":cleaned_reports})
    
    #output_df = output_df.astype(str)
    output_name = str('output_'+ model + '.csv')
    output_df['epe'] = output_df['epe'].astype(int)
    output_df.to_csv(output_name, mode="a", index=False)


    print(model)
    
    # Close Azure API
    client.close()
