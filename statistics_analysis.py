# statistics_analysis.py
# calculates basic statistics on the output CSV from LLM_EPE.py

import pandas as pd
import os


#decorator to avoid division by zero
def try_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred: {e}")
    return wrapper

@try_decorator
def stats_divide(num_val, denom_val):
    return num_val/denom_val

def performance_statistics(output_file):
    # only CSV supported
    with open(output_file, 'a') as f:
        fname = output_file
        df = pd.read_csv(fname)
        df = df.reset_index() # make sure indexes pair with number of rows
        df.columns = df.columns.str.lower()

        #drop invalid rows
        df = df.drop(df[df.valid <1].index)

        true_EPE_positive = df.loc[df.epe == 1, "epe"].count()
        true_EPE_negative = df.loc[df.epe == 0, "epe"].count()
        test_EPE_positive = df.loc[df.llm_classification == "present", "llm_classification"].count()
        test_EPE_negative = df.loc[df.llm_classification == "absent", "llm_classification"].count()

        print("total EPE positive cases "+str(true_EPE_positive)+'\n')
        print("LLM EPE positive cases "+str(test_EPE_positive)+'\n')
        print("total EPE negative cases "+str(true_EPE_negative)+'\n')
        print("LLM EPE negative cases "+str(test_EPE_negative)+'\n')
        print("total cases "+str(true_EPE_negative+true_EPE_positive)+'\n')
        print("total LLM cases "+str(test_EPE_negative+test_EPE_positive)+'\n')
              
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for index, row in df.iterrows():
            llmval = row['llm_classification']
            epeval = row['epe']


            if (llmval == "present") and (epeval == 1):
                #true positive
                true_positive+=1
            elif (llmval == "present") and (epeval == 0):
                #false_positive
                false_positive+=1
            elif (llmval == "absent") and (epeval == 0):
                #true negative
                true_negative+=1
            elif (llmval == "absent") and (epeval == 1):
                #false negative
                false_negative+=1
            else:
                #some kind of error occurred
                print("ERROR " + str(row['accession'])+"\n")

        print("TP "+str(true_positive)+"\n")
        print("TN "+str(true_negative)+"\n")
        print("FP "+str(false_positive)+"\n")
        print("FN "+str(false_negative)+"\n")
        print()


        # to avoid division by zero, use the stats_divide function (see above), which has a decorator to use a try statement, and will skip over any attempt to divide by zero
        FPR_val =           stats_divide(false_positive,(false_positive+true_negative))
        FNR_val =           stats_divide(false_negative,(true_positive+false_negative))
        sensitivity_val =   stats_divide(true_positive,(true_positive+false_negative))
        specificity_val =   stats_divide(true_negative,(false_positive+true_negative))
        accuracy_val =      stats_divide((true_positive+true_negative),(true_EPE_negative+true_EPE_positive))
        PPV_val =           stats_divide(true_positive,(true_positive+false_positive))
        NPV_val =           stats_divide(true_negative,(true_negative+false_negative))


        print("accuracy "+str(accuracy_val)+"\n")

        print("sensitivity "+str(sensitivity_val)+"\n")

        print("specificity "+str(specificity_val)+"\n")

        print("FPR "+str(FPR_val)+"\n")

        print("FNR "+str(FNR_val)+"\n")

        print("PPV "+str(PPV_val)+"\n")

        print("NPV "+str(NPV_val)+"\n")


if __name__ == '__main__':

    # change filename to the output file to run on, for example output_gpt-4.csv
    filename = 'output.csv'
    performance_statistics(filename)