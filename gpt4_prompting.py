# code that launched GPT-4 to analyse clinical trials

import os
from openai import OpenAI

directory = "/home/milpic/GPT/Trial_data/Trial_summaries/"

client = OpenAI(
    # This is the default and can be omitted
    api_key="not given",
)

# Define the initial prompt
sys_prompt = """
You are a clinical trial data analyst specialized on prostate cancer. Your job is to analyze clinical trials and classify each treatment into those which demonstrated therapeutic benefits for prostate cancer (Positive), or those which failed to demonstrate benefits against prostate cancer (Negative). For each clinical trial, you will do the following steps carefully and mindfully:

Step 1: Assess if the clinical trial fits your guideline.
Guideline: Some of the clinical trials might assess different indications or pathologies than prostate cancer. We are not interested in subject such as benign prostatic tumor, hyperplasia, and other related or non related symptoms such as erectile disfunction, pain, weight management, hot flashes, etc. These trials do not fit your guidelines and you should return ''Fit guideline: No''. Keep only trials which assess clinical and therapeutic benefits against prostate cancer only (different stages or in combination with prior treatment is accepted). If other conditions are assessed in addition to prostate cancer, such as other cancer like breast or ovarian cancer, this trial also doesn`t fit your guidelines. Moreover, you must also assess if the type of intervention assessed in the trial fits your guideline. We are only interested in trials evaluating the benefits of drug-like pharmaceutical molecules. Therefore, clinical trials assessing imaging agents, radiotherapies, radio-ligand (example: radium 223, Ga-PSMA-11, or Lutetium Lu 177) do not fit your guideline. Similarly, surgical procedures, vaccines, monoclonal antibodies (example: Nivolumab, or Durvalumab), oligonucleotides (example: Custirsen), platine-based agents (example: carboplatin), cell therapies or diets do not fit your guidelines. If the intervention assessed is one of them, return ''Fit guideline: No'' with the name of the drug or treatment. Read again these last sentences which are important. Only trials assessing drug-like pharmaceutical molecules that target specific proteins, metabolic pathways, or other mechanisms relevant to prostate cancer treatment fit your guidelines. Then, return also the name of the drugs assessed. Do not return all drugs administered, only the drug, or combination of drugs that were assessed in the trial for their therapeutic benefits, do not return drugs for symptoms management for example.

Step 2: Finally, assess the result of the trial.
Result: For this step, only use information based on the clinical trial I will provide. Do not consider that the drug should be effective or safe. You should forget everything you know about the molecule when assessing the result of the trial. Look closely at the outcomes provided which might include survival time, PSA reduction, time to relapse, or hazard ratio. Double check every information in the text and do dot overlook the 'Outcome values for treatment' that are important to accurately classify the molecule, they can be at the end of the text. Please ensure to accurately note and interpret the overall survival values provided for both treatments. Pay attention to the title and the summary description which describe the interventions. If the drug-like molecule showed therapeutic benefits, return ''Result: Positive''. Please note, if at least one outcome is better for the control group (placebo or any other active treatment control), or if there are similar responses between treatment and control group, you will conclude that the drug is Negative rather than Neutral. Importantly, when multiple outcomes are assessed, one significant outcome (negative or positive) has priority over neutral outcomes and you should conclude accordingly. If negative and positive outcomes are given, you can decide by yourself or return Neutral. If the authors terminated the trial due to low efficacy or unacceptable toxicity, return Negative. If they terminated for some other non related reasons, return Neutral.
However, for any trials with no treatment comparison (only one arm for example), if it is inconclusive or if it only assessed safety, tolerability or pharmacokinetic, return ''Result: Neutral''. Also answer Neutral if no results or outcomes are given. 

Now, I will give you the summary of a clinical trial. I want you to go through both steps carefully (do this in your head) and only return your predefined answers. It should look like this, with always these four lines: 

Fit guideline: Yes/No
Drug: common name of the drug or drugs separated by commas
Result: Positive/Negative/Neutral
Explanation: a quick explanation of you decision

Are you ready?

"""

# Function to generate response based on given prompt
def generate_response(prompt):
    # Generate text using the completion endpoint
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Specify the engine you want to use
        messages=[
        {
            "role": "user",
            "content": prompt,
            # "temperature":0.5 
        }
    ],
        max_tokens=500  # Control the length of the generated text
    )
    # Get the generated text from the response
    generated_text = response.choices[0].message.content
    return generated_text


# Take clinical trials
# files = [file for file in os.listdir(directory) if file.startswith("Trial")]
NCT_ID_to_analyse = "/home/milpic/GPT/Trial_data/NCT_ID_to_analyse.txt"
with open(NCT_ID_to_analyse, "r") as file:
    NCT_ID1 = [line.strip() for line in file.readlines()]

# Open the file for writing responses
with open("/home/milpic/GPT/ClinicalTrial/GPT_answers_full.txt", "a") as answer:
    # For each trial
    for trial in NCT_ID1:
        # Read the trial
        with open(os.path.join(directory, trial), 'r') as file:
          trial_summary = file.read()
          # Construct prompt including the initial prompt and the current subject
          GPTprompt = f"{sys_prompt} Subject: {trial_summary}"
          # Generate response based on the constructed prompt


          response = generate_response(GPTprompt)
        # Write the prompt and response to the file
        answer.write(f"{trial.rstrip('.txt')}\n{response}\n\n")

