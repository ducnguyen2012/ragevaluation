# Modules to Import

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
import ast
#! ragas stuff
from dotenv import load_dotenv
import os
load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
def CSVpreprocessing(csvPath: str):
    df = pd.read_csv(csvPath)
    questionList = df['question']
    answerList = df['answer']
    contextList = df['contexts']
    contextListConvertedEleToList = [[context] for context in contextList]
    groundTruthList = df['groundtruth']
    return questionList, answerList, contextListConvertedEleToList, groundTruthList



def evaluateFunction(questionList: list, answerList: list, contextListConvertedEleToList: list, groundTruthList: list):

    data = {
        "question": questionList, #! replace with question list!
        "answer": answerList,  #! replace with response from chatbot
        "contexts": contextListConvertedEleToList,  
        "ground_truth": groundTruthList #! replace with response from EvaluateChatbot
    }

    # Define the features explicitly to ensure correct data types
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),  # Ensuring contexts is treated as a sequence of strings
        "ground_truth": Value("string")
    })

    # Convert the dictionary to a Dataset with the specified features
    dataset = Dataset.from_dict(data, features=features)

    # Perform the evaluation using the adjusted dataset
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall
        ],
    )

    return result



def Evaluate(CSVPath: str):
    #! if user use csv file:
    questionList, answerList, contextListConvertedEleToList, groundTruthList = CSVpreprocessing(CSVPath)
    evaluateFunction(questionList, answerList, contextListConvertedEleToList, groundTruthList)

print(Evaluate("../data/Testdata.csv"))