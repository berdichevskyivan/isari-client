import requests
import threading
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")
workerKey = os.getenv('WORKER_KEY')
environment = os.getenv('WORKER_ENVIRONMENT')

# Set the base URL based on the environment
if environment == 'work':
    base_url = 'https://isari.ai/'
elif environment == 'local':
    base_url = 'http://localhost/'

model = None
tokenizer = None

client_script_hash = None

def load_model_and_tokenizer(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quantization_4bit_config = BitsAndBytesConfig(load_in_4bit=True)
    quantization_8bit_config = BitsAndBytesConfig(load_in_8bit=True)

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    # Quantize the model when loading it for better inference times, but sadly, less performance AKA Intelligence
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", quantization_config=quantization_8bit_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def run_inference(input_text, temperature):
    from transformers import pipeline
    try:
        global model, tokenizer
        if model is None or tokenizer is None:
            model_path = "microsoft/Phi-3-mini-4k-instruct"
            model, tokenizer = load_model_and_tokenizer(model_path)

        messages = [
            {"role": "system", "content": "You are an expert problem solver and analyzer. You use science, engineering, technology and math to identify or solve problems."},
            {"role": "system", "content": "Follow the user instructions, SPECIALLY regarding the subjects or fields you need to exclude from your output. Focus solely on practical scientific outputs."},
            {"role": "user", "content": input_text},
        ]

        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
        )

        generation_args = { 
            "max_new_tokens": 1000, 
            "return_full_text": False, 
            "temperature": temperature, 
            "do_sample": temperature > 0, 
        }
        
        output = pipe(messages, **generation_args)
        generated_text = output[0]['generated_text']

        return generated_text
    except Exception as e:
        return e

def check_for_tasks():
    try:
        check_task_url = f'{base_url}checkForTask'
        store_completed_task_url = f'{base_url}storeCompletedTask'
        response = requests.post(check_task_url, json={'workerKey': workerKey, 'scriptHash': client_script_hash})
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            task_id = data.get('result').get('task').get('id')

            # If we have a file cache.json and the task_id matches the task_id from the dictionary, then we bypass inference
            # and send the output to the gateway. If the task_id doesn't match what is being sent, we delete the file.
            # Check if cache.json exists
            if os.path.exists('cache.json'):
                # Load the cached data
                with open('cache.json', 'r') as cache_file:
                    cache_data = json.load(cache_file)
                
                # Check if the task_id matches
                if cache_data.get('task_id') == task_id:
                    # Bypass inference and send the cached output to the gateway
                    print({'success': True, 'output': cache_data.get('output')})
                    print("We have the correct cached file. Sending output to the Gateway server...")
                    output = cache_data.get('output')
                    response = requests.post(store_completed_task_url, json={'output': output, 'workerKey': workerKey, 'scriptHash': client_script_hash, 'task_id': task_id})
                    response.raise_for_status()
                    # Wait for response
                    data = response.json()
                    print('data returned from storeCompletedTask', data)
                    print('Removing cache file...')
                    os.remove('cache.json')
                    # If successful, we run the function again, looking for another task
                    if data.get('success'):
                        print('Successfully sent cached output to Gateway. Will retrieve another task in 5 seconds...')
                        # Create a Timer object that will run 'my_function' after 'delay' seconds
                        timer = threading.Timer(5, check_for_tasks, args=[])
                        # Start the timer
                        timer.start()
                    else:
                        print("Error:", data.get('message'))
                else:
                    # If the task_id doesn't match, delete the cache file
                    print('Cache file exists but it doesnt match the task id. Removing cache file...')
                    os.remove('cache.json')
            else:

                # Generation args
                # Temperature varies according to the task type
                # Currently, only the task type Extrapolation, has sampling and thus increased temperature
                # This is for the purpose of obtaining more unconventional and novel results, less deterministic ones.
                # For some tasks, we need deterministic outputs. For others, we don't.
                temperature = data.get('result').get('taskType').get('temperature')
                task_type = data.get('result').get('taskType').get('id')
                role = data.get('result').get('taskType').get('role')
                task_name = data.get('result').get('taskType').get('name')
                task_description = data.get('result').get('taskType').get('description')
                instructions = next((inst.get('instruction') for inst in data.get('result').get('instructions') if inst.get('instruction_type') == 'output'), '')
                negative_prompt = data.get('result').get('negativePrompt')
                
                input_text = ''

                # If task type is 1, we need to add |Issue Title| and |Issue Context|
                if task_type == 1:
                    issue_title = data.get('result').get('userInput').get('issue_title')
                    issue_context = data.get('result').get('userInput').get('issue_context')
                    input_text = (f"Assume this role: {role}\n"
                                f"You must perform this task: {task_name}\n"
                                f"This task consists of: {task_description}\n"
                                f"This is the |Issue Title|: {issue_title}\n"
                                f"This is the |Issue Context|: {issue_context}\n"
                                f"These are your output instructions: {instructions}\n\n"
                                f"|| Exclude this from your output: {negative_prompt}\n")
                else:
                    # Here we prepare the input text
                    subject_name = data.get('result').get('issue').get('name')
                    subject_description = data.get('result').get('issue').get('description')
                    context = data.get('result').get('issue').get('context')
                    input_text = (f"Assume this role: {role}\n"
                                f"You must perform this task: {task_name}\n"
                                f"This task consists of: {task_description}\n"
                                f"This is the subject: {subject_name}\n"
                                f"This is a brief description of the subject: {subject_description}\n"
                                f"This is a brief context of the subject: {context}\n"
                                f"Exclude this from your output: {negative_prompt}\n"
                                f"These are your output instructions: {instructions}\n\n"
                                f"|| Exclude this from your output: {negative_prompt}\n")
                
                metrics = data.get('result').get('metrics')
                if metrics and len(metrics) > 0:
                    print('There are metrics attached')
                    input_text += "These are the metrics:\n"
                    
                    # Loop through each metric
                    for metric in metrics:
                        metric_name = metric.get('name')
                        metric_description = metric.get('description')
                        input_text += f"Metric name: {metric_name}\n"
                        input_text += f"This is a description of the metric: {metric_description}\n"
                        input_text += "These are the criteria for this metric, to be used ONLY as context:\n"
                        
                        # Loop through each criterion within the metric
                        for criterion in metric.get('criteria', []):
                            criterion_name = criterion.get('name')
                            criterion_description = criterion.get('description')
                            input_text += f"This is the name for this criteria: {criterion_name}\n"
                            input_text += f"This is the description for this criteria: {criterion_description}\n"
                        
                    input_text += "Do NOT provide a value for the criteria. ONLY provide a value to the metrics.\n"
                    input_text += "This is an example output, for guidance: { complexity: 99, scope: 99 }\n"
                    input_text += "Each metric should NOT contain a JSON but rather, a single integer.\n"

                print(input_text)
                print("Current temperature for this task type is: ", temperature)
                generated_text = run_inference(input_text, temperature)

                cleaned_text = generated_text.replace("\\n", "").replace("```json", "").replace("```", "")
                print("Cleaned text is: ", cleaned_text)
                json_object = json.loads(cleaned_text)
                formatted_output = json.dumps(json_object)

                # If output is generated successfully, we cache it.
                cache_data = {
                    "task_id": task_id,
                    "output": formatted_output
                }

                with open('cache.json', 'w') as cache_file:
                    json.dump(cache_data, cache_file)
                
                # Once we have a generated text, we need to store it on a local file with the id of the task
                # if the task id ( data.get('result').get('task').get('id') matches the stored id , the we use that Stored output, and delete the cache once we send it to the Gateway )
                # We send the output to the Gateway
                response = requests.post(store_completed_task_url, json={'output': formatted_output, 'workerKey': workerKey, 'scriptHash': client_script_hash, 'task_id': task_id})
                response.raise_for_status()
                # Wait for response
                data = response.json()
                # Only if a specific error is returned, we do NOT delete the cache
                # In most instances, we do.
                print('data returned from storeCompletedTask', data)
                print('Removing cache file...')
                os.remove('cache.json')
                # If successful, we run the function again, looking for another task
                if data.get('success'):
                    print('Successfully sent output to Gateway. Will retrieve another task in 5 seconds...')
                    # Create a Timer object that will run 'my_function' after 'delay' seconds
                    timer = threading.Timer(5, check_for_tasks, args=[])
                    # Start the timer
                    timer.start()
                else:
                    print("Error:", data.get('message'))
        else:
            if data.get('error_code') == 'NO_MORE_TASKS':
                print("There are no more tasks. Stopping execution.")
            else:
                print("Error:", data.get('message'))
    except requests.exceptions.RequestException as error:
        print('Error checking for tasks:', error)
        if os.path.exists('cache.json'):
            os.remove('cache.json')

if __name__ == "__main__":
    # We dont set the worker id here anymore. 
    # That is set in the database
    print("WORKER_KEY: ", workerKey)
    if workerKey:
        try:
            # Read the content of client.py
            with open('client.py', 'r') as file:
                script_content = file.read()
            # We first send the script to the server and hash it in order to validate it
            validate_script_url = f'{base_url}validateScript'
            response = requests.post(validate_script_url, json={'workerKey': workerKey, 'script': script_content})
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                print('Script was validated successfully. Proceeding to check for tasks...')
                client_script_hash = data.get('hash')
                check_for_tasks()
            else:
                print("Error:", data.get('message'))
        except requests.exceptions.RequestException as error:
            print('Error validating script:', error)
            if os.path.exists('cache.json'):
                os.remove('cache.json')
    else:
        print("Worker Key was not found. Please add a WORKER_KEY to your .env file")