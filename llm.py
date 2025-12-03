
import json
import logging
import os
import re
import time
from typing import Dict, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor  
import ast
from my_secrets import GPT_API_KEY, PROMPT, GROQ_KEY
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, END
from utility_functions import (
    transcribe_folder_to_csv,
    process_folder_vad,
    save_num_speakers,
    process_audio_directory,
    transcript_quality,
    force_alignment_and_ctc_score,
    check_upsampling_folder,
    language_identification_indiclid,
    transliterate_file,
    get_required_inputs
)
import pandas as pd
logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
llm = ChatOpenAI(
    model_name="gpt-4.1-mini",
    openai_api_key= GPT_API_KEY,
)
# from langchain_groq import ChatGroq
# llm = ChatGroq(
#     model="llama-3.1-8b-instant", 
#     api_key= GROQ_KEY,
#     temperature=0.1,            
#     max_tokens=4096              
# )
def parse_prompt(prompt: str) -> Dict[str, Optional[str]]:
    result = {
        "audio_dir": None,
        "ground_truth_csv": None,
        "lang_code": None
    }
    path_pattern = r"['\"]?(/[^'\"]+(?:/[^'\"]+)*/?)['\"]?"
    csv_pattern = r"['\"]?(/[^'\"]+(?:/[^'\"]+)*\.csv)['\"]?"
    lang_pattern = r"\b(bn|gu|hi|kn|ml|mr|pa|sd|si|ta|te|ur)\b"
    audio_dir_match = re.search(path_pattern, prompt)
    if audio_dir_match:
        result["audio_dir"] = audio_dir_match.group(1)
    csv_match = re.search(csv_pattern, prompt)
    if csv_match:
        result["ground_truth_csv"] = csv_match.group(1)
    lang_match = re.search(lang_pattern, prompt, re.IGNORECASE)
    if lang_match:
        result["lang_code"] = lang_match.group(1).lower()
    if not result["audio_dir"] and not result["ground_truth_csv"]:
        llm_prompt = f"""Extract the following from the prompt:
        1. Audio directory path (e.g., /path/to/audio/)
        2. Ground truth CSV path (e.g., /path/to/file.csv)
        3. Language code (e.g., hi, te)
        If any are unclear, return None for that field.
        Prompt: {prompt}
        Return a JSON object with keys 'audio_dir', 'ground_truth_csv', 'lang_code'.
        """
        try:
            response = llm.invoke(llm_prompt)
            content = response.content.strip()
            if content.startswith('```json'):                                                                       #markdown format correction 
                content = content.replace('```json', '').replace('```', '').strip()
            llm_result = json.loads(content)
            for key in result:
                if not result[key]:
                    result[key] = llm_result.get(key)
        except Exception as e:
            logging.error(f"LLM prompt parsing failed: {e}")                                                         # LLM failsafe
            if audio_dir_match:
                result["audio_dir"] = audio_dir_match.group(1)
            elif "Audio_dir" in prompt:
                audio_dir_match = re.search(r"Audio_dir\s*=\s*['\"]?(/[^'\"]+/)['\"]?", prompt)
                if audio_dir_match:
                    result["audio_dir"] = audio_dir_match.group(1)
    
    if result["audio_dir"] and not os.path.isdir(result["audio_dir"]):
        logging.error(f"Invalid audio directory: {result['audio_dir']}")
        result["audio_dir"] = None
    if result["ground_truth_csv"] and not os.path.isfile(result["ground_truth_csv"]):
        logging.error(f"Invalid ground truth CSV: {result['ground_truth_csv']}")
        result["ground_truth_csv"] = None
    return result

class CombinedStateDict(TypedDict, total=False):
    audio_dir: str
    ground_truth_csv: str
    lang_code: str
    user_prompt: str
    A: str
    D: str
    E: str
    character_output: str
    vocab_output: str
    audio_length_output: str
    ctc_score_output: str
    language_verification_output: str
    upsampling_output: str
    valid_speaker_output: str
    domain_checker_output: str
    audio_transcript_matching_output: str
    language_identification_indiclid_output: str
    normalization_remove_tags_output: str
    llm_score_output: str
    transliteration_output: str
    corruption_output: str
    extension_output: str
    sample_rate_output: str
    english_word_count_output: str
    speaker_duration_output: str
    utterance_duplicate_checker_output: str
    wer_computation_agent_output: str

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
tools = [repl_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def select_tasks(user_prompt: str) -> str:
    prompt_1 = f"""You are given the following functions:
    1. ASR Transcription
    2. Number of Speakers calculation and duration per speaker
    3. Quality of Transcript
    4. Graphene or character calculation
    5. Vocab calculation
    6. Language verification (verify if transcriptions match an expected language)
    7. Audio length calculation
    8. Silence calculation (using VAD)
    9. English Word Counter
    10. CTC score calculation
    11. Upsampling Check
    12. Check if speakers are new or old
    13. Check the domain of the speech dataset
    14. Map transcriptions to audio files using forced alignment
    15. Language identification using ASR transcriptions and IndicLID
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check
    20. Audio extension and format check
    21. Audio sample rate check
    22. Speaker Durations
    23. utterance duplicate checker
    24. WER computation 
    Based on the prompt, identify the task numbers that must be executed to fulfill the request.
    - Inlcude task 1 whenever an audio_dir is provided.
    - If the prompt mentions 'Vocab calculation', include task 5.
    - If the prompt mentions 'Character calculation', include task 4.
    - If the prompt mentions 'new or old' or 'valid speaker', include tasks 2 and 12.
    - if the prompt mentions 'duplicate' or 'duplicates', include task 23.
    - Whenever prompt mentions "Durations" or "durations" , include both task 2 and task 22.
    - If the prompt mentions 'verify' and 'language' or 'expected language', include task 6.
    - If the prompt mentions 'language identification' or 'IndicLID', include tasks 1 and 15 (task 15 requires task 1 to generate transcriptions).
    - Include QC tasks (19, 20, 21) only if explicitly mentioned (e.g., 'corruption', 'extension', 'sample rate')
    - Only include tasks explicitly relevant to the prompt.
    Return the task numbers as a comma-separated string (e.g., '4,5,6').
    Prompt: {user_prompt}
    """
    resp_1 = llm.invoke(prompt_1).content
    task_numbers = re.findall(r'\b\d+\b', resp_1)
    expected_tasks = set()
    resp_1 = ','.join(task_numbers) if task_numbers else ''
    logging.info(f"Selected Tasks: {resp_1}")
    return resp_1

def prompt_checker_agent(user_prompt: str, selected_tasks: str) -> str:
    max_iterations = 3
    current_tasks = set(selected_tasks.split(',') if selected_tasks else [])
    iteration = 0
    task_definitions = """
    1. ASR Transcription
    2. Number of Speakers calculation and duration per speaker
    3. Quality of Transcript
    4. Graphene or character calculation
    5. Vocab calculation
    6. Language verification (verify if transcriptions match an expected language)
    7. Audio length calculation
    8. Silence calculation (using VAD)
    9. English Word Counter
    10. CTC score calculation
    11. Upsampling Check
    12. Check if speakers are new or old
    13. Check the domain of the speech dataset
    14. Map transcriptions to audio files using forced alignment
    15. Language identification using ASR transcriptions and IndicLID
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check
    20. Audio extension and format check
    21. Audio sample rate check
    22. Speaker Durations
    23. utterance duplicate checker
    24. WER computation 
    """
    rules = """
    - If the prompt mentions 'Vocab calculation', include task 5.
    - If the prompt mentions 'Character calculation', include task 4.
    - If the prompt mentions 'verify' and 'language' or 'expected language', include task 6.
    - If the prompt mentions 'language identification' or 'IndicLID', include tasks 1 and 15 (task 15 requires task 1 to generate transcriptions).
    - Include QC tasks (19, 20, 21) only if explicitly mentioned (e.g., 'corruption', 'extension', 'sample rate').
    - If the prompt mentions 'silence' or 'VAD', include task 8.
    - If the prompt mentions 'number of speakers' or 'speakers' or 'duration' or 'durations',  include both task 2 and task 22.
    - If the prompt mentions 'ASR' or 'transcription', include task 1.
    - if Task 9 or task 23 or task 24 are included , then definitely include task 16. 
    - Only include tasks explicitly relevant to the prompt.
    """
    while iteration < max_iterations:
        iteration += 1
        logging.info(f"Prompt Checker Iteration {iteration}: Current tasks: {','.join(current_tasks)}")
        checker_prompt = f"""
        You are a prompt analysis expert. Your task is to verify if the tasks selected for a given user prompt are correct and complete.
        **User Prompt**: {user_prompt}
        **Task Definitions**:
        {task_definitions}
        **Selection Rules**:
        {rules}
        **Currently Selected Tasks**: {','.join(current_tasks) if current_tasks else 'None'}
        **Your Task**:
        1. Analyze the user prompt in detail using natural language and in depth context understanding to identify all tasks that should be executed based on the task definitions and selection rules.
        2. Compare the expected tasks ( based on your analysis ) with the currently selected tasks
        3. Determine if the selected tasks are correct, if any are missing, or if any are extra:
        - 'Correct': All selected tasks match the expected tasks.
        - 'Missing': Some expected tasks are not in the selected tasks.
        - 'Extra': Some selected tasks are not relevant to the prompt.
        4. Strictly adhere to the the ouput format specified below.Do not deviate from this format even if your explaination is detailed
        **Output Format**:
        - Status: Correct or Missing or Extra (any one of these three)
        - Tasks: <comma-separated task numbers which are either missing or extra>
        - Explanation: <brief explanation if Missing or Extra, otherwise empty>
        **Important Notes**:
        - If there are both missing and extra tasks, prioritize 'Extra' as the Status and list only the extra tasks under 'Tasks'. Mention missing tasks (if any) in the explanation.
        - Ensure the 'Status' and 'Tasks' fields align with your final conclusion, not intermediate reasoning.
        - Do not include tasks in 'Tasks' field unless they are missing or extra.
        **Example Output**:
        Example1:
        Status: Missing
        Tasks: 1,2
        Explanation: The prompt mentions 'ASR transcription' (task 1) and 'number of speakers' (task 2), but these were not selected.

        Example2:
        Status: Extra
        Tasks: 3,4
        Explanation: The prompt does not mention 'Quality of Transcript' (task 3) or 'Graphene calculation' (task 4), but these were included in the selected tasks.
        """
        try:
            response = llm.invoke(checker_prompt).content.strip()
            logging.info(f"Prompt Checker Response: {response}")
            status_match = re.search(r'Status:\s*(Correct|Missing|Extra)', response)
            tasks_match = re.search(r'Tasks:\s*([\d,]+)', response)
            explanation_match = re.search(r'Explanation:\s*(.*)', response, re.DOTALL)
            if not status_match or not tasks_match:
                logging.error(f"Invalid prompt checker response format: {response}")
                break
            status = status_match.group(1)
            tasks = tasks_match.group(1).replace(' ', '')
            explanation = explanation_match.group(1).strip() if explanation_match else ''
            if status == 'Correct':
                logging.info(f"Prompt Checker: All tasks correct: {tasks}")
                return tasks
            if status == 'Missing': ####################3 WITH FEEDBACK
                logging.info(f"Prompt Checker with feedback: Missing/Extra tasks detected: {tasks}. Explanation: {explanation}")
                current_tasks.update(tasks.split(','))
                new_selected_tasks = select_tasks(user_prompt + f"\n\n here is the feedback for missing/extra tasks detected from your previous response: {explanation}\n\n")
                if new_selected_tasks:
                    current_tasks.update(new_selected_tasks.split(','))
                current_tasks = {t for t in current_tasks if t and t.isdigit() and 1 <= int(t) <= 24}
                logging.info(f"Updated tasks after iteration {iteration}: {','.join(sorted(current_tasks))}")

            if status == 'Extra': ###########################3 Additional code for extra tasks
                logging.info(f"Prompt Checker with feedback: Extra tasks detected: {tasks}. Explanation: {explanation}")
                current_tasks.difference_update(tasks.split(','))
                if not current_tasks:
                    logging.info("No tasks left after removing extra tasks.")
                    return ''    
        except Exception as e:
            logging.error(f"Prompt Checker failed: {e}")
            break
    final_tasks = ','.join(sorted(current_tasks)) if current_tasks else selected_tasks
    logging.info(f"Prompt Checker: Max iterations reached. Final tasks: {final_tasks}")
    return final_tasks

def topological_sort_tasks(resp_1: str) -> list[list[int]]:
    prompt_2 = f"""You are given a list of 24 tasks related to ASR, audio, and transcription processing:
        Tasks:
        1. ASR Transcription using audio files
        2. Number of Speakers calculation and duration per speaker using audio files
        3. Quality of Transcript using audio files and ground truth file
        4. Graphene or character calculation using ground truth file
        5. Vocab calculation using ground truth file
        6. Language verification (verify if transcriptions match an expected language) using ground truth file
        7. Audio length calculation using audio files
        8. Silence calculation (using VAD) audio files
        9. English Word Counter
        10. CTC score calculation using audio files and ground truth file
        11. Upsampling Check using audio files
        12. Check if speakers are new or old using the results from number of speakers calculation
        13. Check the domain of the speech dataset using transcriptions from ASR
        14. Map transcriptions to audio files using forced alignment, using ground truth transcriptions
        15. Language identification using ASR transcriptions and IndicLID, using transcriptions from ASR
        16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
        17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
        18. Transliteration - Convert Roman script words to Native script using Transliteration
        19. Audio corruption check using audio files
        20. Audio extension and format check using audio files
        21. Audio sample rate check using audio files
        22. Calculate durations of each speaker
        23. Utterance duplicate checker
        24. WER computation
        ## INPUT
        - You will receive a subset of the above task numbers (e.g., {resp_1}). Only consider these tasks for execution.
        ## TASK
        You must perform dependency-aware topological sorting of the requested tasks so that all dependencies are respected and tasks can be grouped for maximum concurrency where possible.
        ### General Instructions
        - Work only with the task numbers included in {resp_1}. Exclude any other tasks.
        - Group independent tasks together so they can execute concurrently (in parallel).
        - Arrange dependent tasks such that dependencies always run first and their dependents run strictly after.
        - Very strict dependency: Tasks 9, 23, and 24 depend on task 16. Task 16 must be completed before any of these are started.
        - Strict dependency: Tasks (3,4,5,6,10,13,14,15,16) all depend on task 1. Task 12 depends on task 2. Task 22 depends on task 2. Task 14 also depends on ground truth being available.
        - Place all tasks that depend on a particular task in subsequent groups, never in the same group as their dependency.
        - Use the provided audio directory for audio-related tasks.
        ### OUTPUT FORMAT
        - You MUST return only **one Python list of lists**.
        - Do NOT repeat the same group multiple times.
        - Do NOT add extra explanations or variations.
        - Return the topologically sorted tasks as a list of lists, where each sub-list contains tasks that can be run in parallel (e.g. [[1], [3,6], [12]]).
        - If the task subset has no dependencies or a valid structure can't be formed, group all tasks into a single list of list (e.g. [[6]]).
        ### EXAMPLES
        - If you receive [1,3,4], output should reflect that 3 and 4 depend on 1, e.g., [[1], [3,4]].
        - If you receive [2,12,22], output should reflect that 12 and 22 depend on 2, e.g., [[2],[12,22]].
        """


    import ast, re, logging

    try:
        resp_2 = llm.invoke(prompt_2).content
        resp_2 = resp_2.strip()

        # Remove code fences if any
        resp_2 = resp_2.replace('```python', '').replace('```', '').strip()

        # Remove any trailing commas in lists
        resp_2 = re.sub(r',(\s*[\]\}])', r'\1', resp_2)

        # Attempt to parse using ast.literal_eval
        try:
            structure = ast.literal_eval(resp_2)
        except:
            # fallback: extract all groups of numbers
            groups = re.findall(r'\[([0-9,\s]*)\]', resp_2)
            structure = [[int(t.strip()) for t in group.split(',') if t.strip()] for group in groups]

        # Validate structure
        if not isinstance(structure, list) or not all(isinstance(group, list) for group in structure):
            raise ValueError("Invalid structure: must be a list of lists")

        # Keep only requested tasks
        tasks_set = set(int(t) for t in resp_1.split(',') if t.strip().isdigit())
        structure = [[task for task in group if task in tasks_set] for group in structure]
        structure = [group for group in structure if group]

        # If empty after filtering, return all tasks as a single group
        if not structure:
            structure = [list(tasks_set)] if tasks_set else []

        logging.info(f"Topological Structure: {structure}")
        return structure

    except Exception as e:
        logging.error(f"Failed to parse topological structure: {e}")
        tasks = [int(t) for t in resp_1.split(',') if t.strip().isdigit()]
        return [tasks] if tasks else []

def corruption_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for corruption check: {audio_dir}")
        return {"corruption_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
    Write a Python script to do the following and then execute it using [python_repl]:
    - Import the 'os' module.
    - Create a CSV named 'corruption_check.csv' in the same directory.
    - The CSV should have columns: Filename, Status.
    - For each audio file in the folder:
        - Use os.path.getsize(file_path) to get the file size.
        - If the file size is 0 KB, mark its 'Status' column as 'corrupt'.
        - If the file size is of .mp3 format, mark its 'Status' column as 'corrupt'.
        - Otherwise, mark its 'Status' column as 'valid'.
    - Save the CSV.
    Finally, respond with "Success" if all files are valid, otherwise respond with "Invalid".
    """
    try:
        response = agent.invoke(task_prompt)
        return {"corruption_output": response.get("Success", "Invalid")}
    except Exception as e:
        logging.error(f"Corruption check failed: {e}")
        return {"corruption_output": f"Error: {e}"}

def extension_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for extension check: {audio_dir}")
        return {"extension_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
    Write a Python script to do the following and then execute it using [python_repl]:
    - Import the 'os' module and use it to iterate through files in the directory.

    Steps:
    1. Check that each file has a valid audio extension (only .wav ).
    2. Create a CSV named 'audio_format_check.csv' in the same directory.
    3. The CSV should have columns: Filename, Valid_Extension.
    4. For each file:
    - 'Valid_Extension' should be True if the filename ends with .wav (case insensitive), else False.
    5. Save the CSV.

    Finally, print "Success" if all files have Valid_Extension=True, otherwise print "Invalid".
    """

    try:
        response = agent.invoke(task_prompt)
        # Check for the expected output string
        if "Success" in response:
            return {"extension_output": "Success"}
        else:
            return {"extension_output": "Invalid"}
    except Exception as e:
        logging.error(f"Extension check failed: {e}")
        return {"extension_output": f"Error: {e}"}

def sample_rate_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for sample rate check: {audio_dir}")
        return {"sample_rate_output": "Invalid: No audio directory"}
    task_prompt = f"""You are given a folder with audio files at this path: {audio_dir}.
    Write a Python script to do the following and then execute it using [python_repl]:
    1. Check each audio file's sample rate
    2. Create a CSV named sample_rate_check.csv with columns: Filename, Sample_Rate, Status
    3. Store "Pass" in Status if sample rate is 16000 Hz, otherwise "Fail"
    4. Save the CSV  in the same directory
    Use libraries like librosa, soundfile, or wave to check the sample rate.
    Finally, respond with "Success" if all files have Status "Pass", otherwise respond with "Invalid".
    """
    try:
        response = agent.invoke(task_prompt)
        if "Success" in response:
            return {"sample_rate_output": "Success"}
        else:
            return {"sample_rate_output": "Invalid"}
    except Exception as e:
        logging.error(f"Sample rate check failed: {e}")
        return {"sample_rate_output": f"Error: {e}"}

def transcription_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcription: {audio_dir}")
        return {"A": "Error: Invalid audio directory"}
    logging.info("Running Transcription")
    result = transcribe_folder_to_csv(audio_dir, source_language="Hindi")
    return {"A": result} #, "audio_dir": audio_dir

def silence_vad_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for silence detection: {audio_dir}")
        return {"D": "Error: Invalid audio directory"}
    logging.info("Running Silence Detection")
    result = process_folder_vad(audio_dir)
    return {"D": result}

def num_speaker_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for speaker diarization: {audio_dir}")
        return {"E": "Error: Invalid audio directory"}
    logging.info("Running Speaker Diarization and Duration Calculation")
    result = save_num_speakers(audio_dir)
    return {"E": result}

def vocab_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))

    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for vocab calculation: {csv_path}")
        return {"vocab_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running vocab_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
    It has a column called 'Transcription' or 'Ground_Truth',(search case insensitively).
    Write a Python script to do the following and execute it using [python_repl] tool:
    1. For each row, extract a list of unique words (vocabulary) from the transcription and store it in a new column called 'vocab_list'.
    2. Save the updated CSV with the new column to the same directory as vocab_list.csv
    Respond with "Success" if the script completes
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "vocab_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"vocab_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"vocab_output": error_msg}
    except Exception as e:
        logging.error(f"Vocab calculation failed: {e}")
        return {"vocab_output": f"Error: {e}"}

def character_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = state.get('A', os.path.join(parent, 'results' ,"indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for character calculation: {csv_path}")
        return {"character_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running character_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
    It has a column called 'Transcription' or 'Ground_Truth',(search case insensitively), take it as transcription coloumn.
    Write a Python script to do the following and execute it using [python_repl] tool:
    1. For each row, extract a list of unique characters from the transcription coloumn and store it in a new column called 'character_list'.
    2. Save the updated CSV with the new column to the same directory as character_list.csv
    Respond with "Success" if the script completes
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "character_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"character_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"character_output": error_msg}
    except Exception as e:
        logging.error(f"Character calculation failed: {e}")
        return {"character_output": f"Error: {e}"}

def audio_length_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for audio length calculation: {audio_dir}")
        return {"audio_length_output": "Error: Invalid audio directory"}
    logging.info("Running audio_length_agent")
    task_prompt = f"""You are given audio files in the folder: {audio_dir}.
    Write a Python script to do the following and execute it using [python_repl] tool:
    1. Create a CSV with columns Filename and Audio_length.
    2. For each audio file, calculate its duration in seconds and store it in 'Audio_length' column.
    3. Save the CSV as audio_length.csv in the same folder.
    """
    try:
        response = agent.invoke(task_prompt)
        return {"audio_length_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Audio length calculation failed: {e}")
        return {"audio_length_output": f"Error: {e}"}

def language_verification_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for language verification: {csv_path}")
        return {"language_verification_output": f"Error: CSV file {csv_path} not found"}
    logging.info(f"Running Devanagari script verification with CSV: {csv_path}")
    task_prompt = f"""You are a script recognition expert.
    Your task is to determine whether the text in the 'ground_truth' column of a CSV file is written in the Devanagari script.
    Please follow these steps:
    Unicode Range for Devanagari Script:
    The Unicode range for the Devanagari script is from U+0900 to U+097F.
    This includes characters used in languages like Hindi, Sanskrit, Marathi, and others that use Devanagari as their writing system.
    use python and follow the Steps:
    1. Load the CSV file at this path: {csv_path} after import necessary python modules.
    2. Identify the 'ground_truth' column (case-insensitive, e.g., 'Ground_Truth', 'transcription').
    3. For each row in the 'ground_truth' column, check if all characters (excluding whitespace and punctuation) fall within the Unicode range U+0900 to U+097F.
    4. Add a new column 'Is_Devanagari' with True if all relevant characters are in the Devanagari range, False otherwise.
    5. If the transcription is empty or contains only whitespace/punctuation, set 'Is_Devanagari' to False.
    6. Save the updated CSV as 'language_verification.csv' in the same directory as the input CSV.
    7. Ensure the output CSV includes columns: 'Filename', 'Transcription' (the ground_truth text), and 'Is_Devanagari'.
    8. Handle errors gracefully.
    Example Text:
    "नमस्ते, आप कैसे हैं?" -> Is_Devanagari: True (all characters are in U+0900 to U+097F)
    "Hello" -> Is_Devanagari: False (characters are not in U+0900 to U+097F)
    Use the `python_repl` tool to execute a Python script that performs these steps.
    Respond with "Success" if the CSV is saved successfully, otherwise return an error message.
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "language_verification.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            logging.info(f"Devanagari verification CSV saved to: {output_path}")
            return {"language_verification_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Devanagari verification failed to generate {output_path}: {response.get('output', 'No output')}")
            return {"language_verification_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Devanagari verification failed: {e}")
        return {"language_verification_output": f"Error: {e}"}

def ctc_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for CTC score: {audio_dir}")
        return {"ctc_score_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for CTC score: {csv_path}")
        return {"ctc_score_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running CTC score calculation")
    try:
        output_path = os.path.join(os.path.dirname(csv_path), "ctc_scores.csv")
        results = process_audio_directory(audio_dir, csv_path, output_path)
        if results:
            df = pd.DataFrame(results)
            if df.empty:
                logging.error("No valid results generated from process_audio_directory")
                return {"ctc_score_output": "Error: No valid results generated"}
            grouped = df.groupby('filename').agg({
                'label': lambda x: ' '.join(x), 
                'average_ctc_score': 'first',  
            }).reset_index()
            grouped['Aligned_Segments'] = grouped['filename'].apply(
                lambda x: json.dumps([
                    {'label': row['label'], 'start': row['start_time'], 'end': row['end_time'], 'score': row['score']}
                    for _, row in df[df['filename'] == x].iterrows()
                ])
            )
            grouped.columns = ['Filename', 'Aligned_Transcript', 'CTC_Score', 'Aligned_Segments']
            grouped['CTC_Status'] = grouped['CTC_Score'].apply(
                lambda x: "Good" if float(x) > 0.7 else "Medium" if float(x) > 0.5 else "Poor"
            )
            grouped = grouped[['Filename', 'Aligned_Segments', 'Aligned_Transcript', 'CTC_Score', 'CTC_Status']]
            grouped.to_csv(output_path, index=False)
            logging.info(f"CTC scores saved to: {output_path}")
            return {"ctc_score_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"No results generated for CTC score calculation")
            return {"ctc_score_output": f"Error: No results generated"}
    except Exception as e:
        logging.error(f"CTC score calculation failed: {e}")
        return {"ctc_score_output": f"Error: {e}"}

def transcript_quality_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(audio_dir, "indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript quality: {csv_path}")
        return {"transcript_quality_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running transcript quality check")
    task_prompt = f"""You are given a CSV file at this path: {csv_path} containing transcriptions.
    Write a Python script to check the quality of each transcript using the transcript_quality function from utility_functions.
    Your script should:
    1. Load the CSV file
    2. Apply the transcript_quality function to each transcript
    3. Store the result in a new column called 'Quality_Check'
    4. Save the updated CSV with the new column to the same directory as transcript_quality.csv
    """
    try:
        response = agent.invoke(task_prompt)
        return {"transcript_quality_output": response.get("output", "Invalid")}
    except Exception as e:
        logging.error(f"Transcript quality check failed: {e}")
        return {"transcript_quality_output": f"Error: {e}"}

def upsampling_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for upsampling check: {audio_dir}")
        return {"upsampling_output": "Error: Invalid audio directory"}
    logging.info("Running upsampling check")
    result = check_upsampling_folder(audio_dir)
    return {"upsampling_output": result}

def valid_speaker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for valid speaker check: {audio_dir}")
        return {"valid_speaker_output": "Error: Invalid audio directory"}
    logging.info("Running valid speaker check")
    parent = os.path.dirname(audio_dir.rstrip("/"))
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'num_speakers.csv'. 
    The CSV has columns 'File Name', 'Number of Speakers', and 'Speaker Durations', where 'Speaker Durations' is a JSON string mapping speaker IDs to their speaking durations in hours.
    Write a Python script to:
    1. Load the 'num_speakers.csv' file.
    2. Create a dictionary to count how many files each speaker appears in.
    3. For each file in the CSV:
    - Parse the 'Number of Speakers' and 'Speaker Durations' columns.
    - If 'Number of Speakers' is 1 and 'SPEAKER_00' appears in more than one file, set Speaker_Status to 'Old' and Common_File to the current file name.
    - If 'Number of Speakers' > 1 and any speaker appears in more than one file, set Speaker_Status to 'Old' and Common_File to the current file name.
    - If 'Number of Speakers' = 'Error' , skip that row.
    - Otherwise, set Speaker_Status to 'New' and Common_File to an empty string.
    4. Create a new CSV with columns: 'Filename', 'Speaker_Status', 'Common_File'.
    5. Save the CSV as 'valid_speaker.csv' in the same directory.
    6. Handle errors gracefully.
    Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(parent, 'results', "valid_speaker.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"valid_speaker_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Valid speaker check failed to generate {output_path}")
            return {"valid_speaker_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Valid speaker check failed: {e}")
        return {"valid_speaker_output": f"Error: {e}"}

def domain_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for domain checker: {audio_dir}")
        return {"domain_checker_output": "Error: Invalid audio directory"}
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = os.path.join(parent, 'results' ,"indicconf_hypothesis.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"domain_checker_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'Indiconformer_Hypothesis' not in df.columns:
            logging.error("Column 'Indiconformer_Hypothesis' not found in CSV.")
            return {"domain_checker_output": "Error: 'Indiconformer_Hypothesis' column missing."}
        original_domains = ['Conversation', 'Keywords Spotting', 'Product Review','GK Questions', 'District Specific', 'KYP - Games',
       'Task of Fives', 'DOI - Religion', 'KYP - Cooking', 'Daily Life']
        domains = []
        for i, transcript in enumerate(df['Indiconformer_Hypothesis']):
            prompt = f"""You are a Hindi language expert. Analyze the following normalized Hindi transcript and determine the general domain of the speech dataset. some original domains to choose from are {original_domains}.
            ## OUTPUT FORMAT: Return the domain from original domains as a one word phrase suitable from original domains. no additional terms
            Now classify this transcript:
            Transcript: {transcript}
            Domain: 
    """
            try:
                response = llm.invoke(prompt)
                domains.append(response.content.strip())

            except Exception as e:
                logging.warning(f"LLM failed on row {i}: {e}")
                domains.append("Unknown")

        df['domain'] = domains
        output_path = os.path.join(parent, 'results', "domain_check.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"Domain check completed. Output saved to {output_path}")
        return {"domain_checker_output": f"CSV saved at: {output_path}"}

    except Exception as e:
        logging.error(f"Domain checker failed: {e}")
        return {"domain_checker_output": f"Error: {e}"}

def audio_transcript_matching_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcript matching: {audio_dir}")
        return {"audio_transcript_matching_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript matching: {csv_path}")
        return {"audio_transcript_matching_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running audio and transcript matching")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing audio files (.wav or .mp3) and a CSV file at {csv_path}. 
    The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains ground truth transcriptions.
    Write a Python script to:
    1. Load the CSV file.
    2. For each row in the CSV:
    - Get the audio file path by joining the folder path with the 'Filename'.
    - Use the force_alignment_and_ctc_score function from utility_functions to perform forced alignment.
    - Create an aligned transcript by joining tokens from aligned_segments, excluding special tokens.
    3. Create a CSV with columns 'Filename', 'Aligned_Segments' (JSON string), and 'Aligned_Transcript'.
    4. Save the CSV as 'audio_transcript_matching.csv' in the same directory.
    5. Handle errors gracefully.
    Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(parent,'results', "audio_transcript_matching.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"audio_transcript_matching_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"Transcript matching failed to generate {output_path}")
            return {"audio_transcript_matching_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Transcript matching failed: {e}")
        return {"audio_transcript_matching_output": f"Error: {e}"}


def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error("Invalid or missing audio directory")
        return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
    parent = os.path.dirname(audio_dir.rstrip("/"))
    output_path = os.path.join(parent, 'results', "indiclid_language_identification.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        logging.info(f"Skipping language identification: File already exists at {output_path}")
        return {"language_identification_indiclid_output": f"File already exists: {output_path}"}
    
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    if not os.path.exists(csv_path):
        logging.error(f"Transcription CSV not found at {csv_path}")
        return {"language_identification_indiclid_output": f"Error: Transcription CSV not found at {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
        if 'Indiconformer_Hypothesis' not in df.columns:
            logging.error("Missing 'Indiconformer_Hypothesis' column in CSV")
            return {"language_identification_indiclid_output": "Error: Missing 'Indiconformer_Hypothesis' column"}
        
        results = []
        for idx, row in df.iterrows():
            transcription = str(row['Indiconformer_Hypothesis'])
            filename = row.get('Filename', f"unknown_{idx}")
            
            if pd.isna(transcription) or transcription.strip() == "":
                logging.warning(f"Empty transcription for {filename}")
                results.append({
                    "Filename": filename,
                    "Transcription": transcription,
                    "Language_Code": "Unknown",
                    "Confidence": 0.0,
                    "Model_Used": "IndicLID"
                })
                continue
            
            try:
                lid_results = language_identification_indiclid(transcription)
                for _, lang_code, confidence, model_used in lid_results:
                    results.append({
                        "Filename": filename,
                        "Transcription": transcription,
                        "Language_Code": lang_code,
                        "Confidence": confidence,
                        "Model_Used": model_used
                    })
            except Exception as e:
                logging.error(f"Language identification failed for {filename}: {e}")
                results.append({
                    "Filename": filename,
                    "Transcription": transcription,
                    "Language_Code": "Error",
                    "Confidence": 0.0,
                    "Model_Used": "IndicLID"
                })
        
        if not results:
            logging.error("No language identification results generated")
            return {"language_identification_indiclid_output": "Error: No language identification results"}
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        logging.info(f"Language identification results saved to {output_path}")
        return {"language_identification_indiclid_output": output_path}
        
    except Exception as e:
        logging.error(f"Error processing language identification: {e}")
        return {"language_identification_indiclid_output": f"Error: {e}"}


def normalization_remove_tags_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    csv_path =  os.path.join(parent, 'results', "indicconf_hypothesis-gt.csv")
    output_path = os.path.join(parent, 'results', "normalized_list.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return {"normalization_remove_tags_output": f"CSV already exists at: {output_path}"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for character calculation: {csv_path}")
        return {"normalization_remove_tags_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running normalizing_agent")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'indicconf_hypothesis.csv'
    It has a column called 'Transcriptions' or 'ground_truth'(search case insensitively), take it as transcription coloumn.
    Write a Python script that reads a CSV file containing two columns: filename and ground_truth. The goal is to clean the ground_truth text and save the results to a new column called normalized_transcripts, then write the updated data to a new CSV.
    Execute the python script to do the following using [python_repl] tool.
    The cleaning steps are:
    -Remove HTML tags like <b> or </b>.
    -Remove any text enclosed in square brackets (e.g., [START]).
    -Remove all symbols like ['#','$','%']
    -The output CSV should preserve the original columns and include the cleaned transcript in the normalized_transcripts column.
    -Save the updated CSV with the new column to the same directory as normalized_list.csv
    Respond with "Success" if the script completes
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(os.path.dirname(csv_path), "normalized_list.csv")
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"normalization_remove_tags_output": f"CSV saved at: {output_path}"}
        else:
            error_msg = f"Failed to generate {output_path}"
            logging.error(error_msg)
            return {"normalization_remove_tags_output": error_msg}
    except Exception as e:
        logging.error(f"Character calculation failed: {e}")
        return {"normalization_remove_tags_output": f"Error: {e}"}

def llm_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for LLM score: {audio_dir}")
        return {"llm_score_output": "Error: Invalid audio directory"}
    logging.info("Running LLM score evaluation")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing a CSV file named 'indicconf_hypothesis.csv'. 
    The CSV has columns 'Filename'(search case insensitively) which contains filenames and either 'ground_truth' or 'transcriptions' coloumn (only one of 'ground_truth' or 'transcriptions' coloumn will be there) ,(search case insensitively), and that coloumn contains ASR transcription.
    Write a Python script to do the following and execute it using [pythom_repl] :
    1. Load the 'indicconf_hypothesis.csv' file.
    2. For each transcription(analyse each line carefully and dont just assign a single score to every row, assign scores only after understanding the fluency of the senetnce and its meaning):
    - Evaluate its coherence and fluency very strictly using the LLM as a judge.
    - Assign a score from 0 to 10(where 10 is given for meaningful sentences , and decrease the scores for as meaningless as they get and if any other language is detected other than one expected mark it 0).
    - Provide a brief comment explaining the score(We are evaluating the accuracy of hindi so be the judge and very strictly comment on the transcriptions)
    3. Create a CSV with columns 'Filename', 'Transcription', 'LLM_Score', and 'Evaluation_Comment'.
    4. Save the CSV as 'llm_scores.csv' in the same directory.
    5. Handle errors gracefully.
    Respond with "Success" if the script completes and the CSV is saved, otherwise "Invalid".
    """
    try:
        response = agent.invoke(task_prompt)
        output_path = os.path.join(parent, 'results', "llm_scores.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if response.get("output", "").strip() == "Success" and os.path.exists(output_path):
            return {"llm_score_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"LLM score evaluation failed to generate {output_path}")
            return {"llm_score_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"LLM score evaluation failed: {e}")
        return {"llm_score_output": f"Error: {e}"}

def transliteration_agent(state: CombinedStateDict) -> CombinedStateDict:
    csv_path = state.get('ground_truth_csv')
    lang_code = state.get('lang_code')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transliteration: {csv_path}")
        return {"transliteration_output": f"Error: CSV file {csv_path} not found"}
    if not lang_code:
        logging.error("No language code provided for transliteration")
        return {"transliteration_output": "Error: No language code provided"}
    logging.info("Running transliteration")
    try:
        result = transliterate_file(csv_path, lang_code)
        return {"transliteration_output": result}
    except Exception as e:
        logging.error(f"Transliteration failed: {e}")
        return {"transliteration_output": f"Error: {e}"}



def speaker_duration_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error("Invalid or missing audio directory")
        return {"speaker_duration_output": "Error: Invalid audio directory"}
    
    csv_path = os.path.join(parent, 'results', "num_speakers.csv")
    output_path = os.path.join(parent,'results', "speaker_durations.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        logging.error(f"Speaker CSV not found: {csv_path}")
        return {"speaker_duration_output": f"Error: Speaker CSV not found: {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['File Name', 'Number of Speakers', 'Speaker Durations']):
            logging.error("Missing required columns in num_speakers.csv")
            return {"speaker_duration_output": "Error: Missing required columns in num_speakers.csv"}
        
        speaker_durations = {}
        for _, row in df.iterrows():
            try:
                durations = json.loads(row['Speaker Durations'].replace('""', '"'))
                for speaker, duration in durations.items():
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
            except Exception:
                continue
        
        if not speaker_durations:
            logging.error("No speaker durations found")
            return {"speaker_duration_output": "Error: No speaker durations found"}
        
        results = [
            {'Speaker': speaker, 'Total_Duration_Hours': round(duration, 6)}
            for speaker, duration in sorted(speaker_durations.items())
        ]
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        logging.info(f"Speaker durations saved to {output_path}")
        return {"speaker_duration_output": output_path}
    
    except Exception as e:
        logging.error(f"Error processing speaker durations: {e}")
        return {"speaker_duration_output": f"Error: {e}"}


def english_word_count_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for word counter: {audio_dir}")
        return {"english_word_count_output": "Error: Invalid audio directory"}

    csv_path = os.path.join(parent, 'results', "normalized_list.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"english_word_count_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'ground_truth' not in df.columns:
            logging.error("Column 'ground_truth' not found in CSV.")
            return {"english_word_count_output": "Error: 'ground_truth' column missing."}
        word_counts = []
        for i, text in enumerate(df['ground_truth']):
            prompt = f"""You are a language expert. Count and return only the number of **English words** (case-insensitive) in the following text.
    Text:
    {text}
    Respond with just the number."""
            try:
                response = llm.invoke(prompt)
                count_str = response.content.strip()
                count = int(count_str)
                word_counts.append(count)
            except Exception as e:
                logging.warning(f"LLM failed on row {i}: {e}")
                word_counts.append(-1)

        df['english_word_count'] = word_counts
        output_path = os.path.join(parent, 'results', "english_word_count.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"English word count completed. Output saved to {output_path}")
        return {"english_word_count_output": f"CSV saved at: {output_path}"}

    except Exception as e:
        logging.error(f"English word count agent failed: {e}")
        return {"english_word_count_output": f"Error: {e}"}

def utterance_duplicate_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for duplicate checker: {audio_dir}")
        return {"utterance_duplicate_output": "Error: Invalid audio directory"}
    csv_path = os.path.join(parent, 'results',"normalized_list.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"utterance_duplicate_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        rows = []
        for column in df.columns:
            if df[column].dtype == 'object':
                duplicates = df[column][df[column].duplicated(keep=False)]
                if not duplicates.empty:
                    counts = duplicates.value_counts()
                    for utterance, count in counts.items():
                        rows.append({
                            "column_name": column,
                            "utterance": utterance,
                            "count": count
                        })
        if rows:
            dup_df = pd.DataFrame(rows)
            output_path = os.path.join(parent, 'results', "duplicate_utterances.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dup_df.to_csv(output_path, index=False)
            logging.info(f"Duplicate check completed. Output saved to {output_path}")
            return {"utterance_duplicate_output": f"CSV saved at: {output_path}"}
        else:
            return {"utterance_duplicate_output": "No duplicate utterances found."}
    except Exception as e:
        logging.error(f"Duplicate utterance checker failed: {e}")
        return {"utterance_duplicate_output": f"Error: {e}"}

import jiwer
def wer_computation_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get("audio_dir")
    parent = os.path.dirname(audio_dir.rstrip("/"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for WER computation: {audio_dir}")
        return {"wer_output": "Error: Invalid audio directory"}
    try:
        ref_csv_path = os.path.join(parent, 'results', "normalized_list.csv")
        hyp_csv_path = os.path.join(parent, 'results', "indicconf_hypothesis.csv")
        if not os.path.exists(ref_csv_path) or not os.path.exists(hyp_csv_path):
            return {"wer_output": "Error: One or both CSV files not found"}
        ref_df = pd.read_csv(ref_csv_path)
        hyp_df = pd.read_csv(hyp_csv_path)
        if len(ref_df) != len(hyp_df):
            return {"wer_output": "Error: CSVs do not have the same number of rows"}
        references = ref_df["normalized_transcripts"].astype(str)
        hypotheses = hyp_df["Indiconformer_Hypothesis"].astype(str)
        wer_rows = []
        for ref, hyp in zip(references, hypotheses):
            try:
                error = jiwer.wer(ref, hyp)
                wer_rows.append({
                    "Reference": ref,
                    "Hypothesis": hyp,
                    "WER": round(error, 4)
                })
            except Exception as e:
                wer_rows.append({
                    "Reference": ref,
                    "Hypothesis": hyp,
                    "WER": "Error"
                })
                logging.warning(f"WER computation failed for row: {e}")
        wer_df = pd.DataFrame(wer_rows)
        output_path = os.path.join(parent, 'results', "wer.csv")
        wer_df.to_csv(output_path, index=False)
        logging.info(f"WER computation completed. CSV saved to {output_path}")
        return {"wer_output": f"WER CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"WER computation failed: {e}")
        return {"wer_output": f"Error: {e}"}
node_map = {
    1: ("node_transcription", transcription_func, "A"),
    2: ("node_num_speaker", num_speaker_func, "E"),
    3: ("node_transcript_quality", transcript_quality_agent, "transcript_quality_output"),
    4: ("node_character", character_agent, "character_output"),
    5: ("node_vocab", vocab_agent, "vocab_output"),
    6: ("node_language_verification", language_verification_agent, "language_verification_output"),
    7: ("node_audio_length", audio_length_agent, "audio_length_output"),
    8: ("node_silence", silence_vad_func, "D"),
    9: ("node_code-mix-counter", english_word_count_agent, "english_word_count_output"),
    10: ("node_ctc_score", ctc_score_agent, "ctc_score_output"),
    11: ("node_upsampling", upsampling_agent, "upsampling_output"),
    12: ("node_valid_speaker", valid_speaker_agent, "valid_speaker_output"),
    13: ("node_domain_checker", domain_checker_agent, "domain_checker_output"),
    14: ("node_audio_transcript_matching", audio_transcript_matching_agent, "audio_transcript_matching_output"),
    15: ("node_language_identification_indiclid", language_identification_indiclid_agent, "language_identification_indiclid_output"),
    16: ("node_normalization_remove_tags", normalization_remove_tags_agent, "normalization_remove_tags_output"),
    17: ("node_llm_score", llm_score_agent, "llm_score_output"),
    18: ("node_transliteration", transliteration_agent, "transliteration_output"),
    19: ("node_corruption", corruption_agent, "corruption_output"),
    20: ("node_extension", extension_agent, "extension_output"),
    21: ("node_sample_rate", sample_rate_agent, "sample_rate_output"),
    22: ("speaker_duration",speaker_duration_agent, "speaker_duration_output"),
    23: ("node_utterance_duplicate_checker",utterance_duplicate_checker_agent,"utterance_duplicate_checker_output"),
    24: ("wer_computation",wer_computation_agent,"wer_computation_agent_output"),
    
}

def build_graph_from_structure(structure: list[list[int]], valid_tasks: set) -> StateGraph:
    graph_builder = StateGraph(CombinedStateDict)
    added_nodes = set()
    valid_structure = [[task for task in group if str(task) in valid_tasks] for group in structure]
    valid_structure = [group for group in valid_structure if group]
    if not valid_structure and valid_tasks:
        valid_structure = [[int(task) for task in valid_tasks if task.isdigit()]]
    for group in valid_structure:
        for task_id in group:
            if task_id in node_map and task_id not in added_nodes:
                node_name, func, _ = node_map[task_id]
                graph_builder.add_node(node_name, func)
                added_nodes.add(task_id)
    graph_builder.add_node("start", lambda state: state)
    for i in range(len(valid_structure)):
        current_group = valid_structure[i]
        if i == 0:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge("start", node_name)
        if i < len(valid_structure) - 1:
            next_group = valid_structure[i + 1]
            for curr_task in current_group:
                curr_node_name, _, _ = node_map[curr_task]
                for next_task in next_group:
                    next_node_name, _, _ = node_map[next_task]
                    graph_builder.add_edge(curr_node_name, next_node_name)
        if i == len(valid_structure) - 1:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge(node_name, END)
    if valid_structure:
        graph_builder.set_entry_point("start")
    else:
        raise ValueError("No valid tasks provided in structure")
    return graph_builder.compile()
def main(user_prompt: str):
    parsed_inputs = parse_prompt(user_prompt)
    if not parsed_inputs["audio_dir"] and not parsed_inputs["ground_truth_csv"]:
        logging.error("No valid audio directory or ground truth CSV provided in prompt")
        print("Error: Please provide audio directory and/or ground truth CSV in the prompt")
        return
    resp_1 = select_tasks(user_prompt)
    
    verified_tasks = prompt_checker_agent(user_prompt, resp_1)
    valid_tasks = set(verified_tasks.split(',')) if verified_tasks else set()
    structure = topological_sort_tasks(verified_tasks)
    print("Using structure:", structure)

    main_graph = build_graph_from_structure(structure, valid_tasks)
    main_graph.get_graph().print_ascii()

    initial_state = {
        "audio_dir": parsed_inputs["audio_dir"],
        "ground_truth_csv": parsed_inputs["ground_truth_csv"],
        "lang_code": parsed_inputs["lang_code"],
        "user_prompt": user_prompt
    }
    final_result = main_graph.invoke(initial_state)
    print("\nFinal Pipeline Results:")
    for key, value in final_result.items():
        if key not in ["audio_dir", "ground_truth_csv", "lang_code", "user_prompt"]:
            print(f"{key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"{key}: {value}")

if __name__ == "__main__":
    
    user_prompt = PROMPT
    # "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification,calculate speaker duration in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt = "I want to check the number of speakers, calculate the duration for each speaker, verify if speakers are new or old, compute WER, count English words, check for utterance duplicates, and clean the transcriptions by removing HTML tags in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt= "verify language as Hindi (' hi '),CTC SCORE,   WER computation , normalization , calculate LLM score , check domain ,english word counter, and check for utterance duplicate  in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    # user_prompt= "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification, calculate speaker duration, verify if speakers are new or old in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/' "
    # user_prompt = "I need to analyze my Hindi-English conversational dataset to assess the quality of dialogue flow, contextual consistency across speaker turns, and conversational coherence for training our dialogue state tracking model. The data contains multi-turn conversations where speakers ask questions and provide answers in alternating fashion. audios present in directory: '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    
    main(user_prompt)
    
