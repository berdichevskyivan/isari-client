import requests
import threading
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")
workerKey = os.getenv('WORKER_KEY')
environment = os.getenv('WORKER_ENVIRONMENT')

if environment:
    base_url = 'http://localhost/'
else:
    base_url = 'https://isari.ai/'

model = None
tokenizer = None

client_script_hash = None

def load_model_and_tokenizer(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quantization_4bit_config = BitsAndBytesConfig(load_in_4bit=True)
    quantization_8bit_config = BitsAndBytesConfig(load_in_8bit=True)

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    # Quantize the model when loading it for better inference times
    # Better models and more resources will allow us to process tasks more efficiently
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
            print('Successfully received Task.')
            task_id = data.get('response').get('task_id')
            temperature = data.get('response').get('temperature')
            input_text = data.get('response').get('input_text')

            print('Starting inference...')
            generated_text = run_inference(input_text, temperature)
            
            print('Inference completed. Cleansing output and sending back to Gateway...')
            cleaned_text = generated_text.replace("\\n", "").replace("```json", "").replace("```", "")
            json_object = json.loads(cleaned_text)
            formatted_output = json.dumps(json_object)

            response = requests.post(store_completed_task_url, json={'output': formatted_output, 'workerKey': workerKey, 'scriptHash': client_script_hash, 'task_id': task_id})
            response.raise_for_status()

            data = response.json()
            print('Response from Gateway: ', data)

            # If successful, we run the function again, looking for another task
            if data.get('success'):
                print('Successfully sent output to Gateway. Will retrieve another task in 5 seconds...')
                timer = threading.Timer(5, check_for_tasks, args=[])
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

def check_for_workflow_tasks():
    try:
        check_workflow_task_url = f'{base_url}checkForWorkflowTask'
        store_completed_workflow_task_url = f'{base_url}storeCompletedWorkflowTask'
        response = requests.post(check_workflow_task_url, json={'workerKey': workerKey, 'scriptHash': client_script_hash})
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            print('Successfully received Task.')
            task_id = data.get('response').get('task_id')
            temperature = data.get('response').get('temperature')
            input_text = data.get('response').get('input_text')

            print('Input text: ', input_text)
            print('Temperature: ', temperature)
            print('Starting inference...')
            generated_text = run_inference(input_text, temperature)
            
            print('Inference completed. Cleansing output and sending back to Gateway...')
            cleaned_text = generated_text.replace("\\n", "").replace("```json", "").replace("```", "")
            json_object = json.loads(cleaned_text)
            formatted_output = json.dumps(json_object)

            print('Generated output: ', cleaned_text)

            response = requests.post(store_completed_workflow_task_url, json={'output': formatted_output, 'workerKey': workerKey, 'scriptHash': client_script_hash, 'task_id': task_id})
            response.raise_for_status()

            data = response.json()
            print('Response from Gateway: ', data)

            # If successful, we run the function again, looking for another task
            if data.get('success'):
                print('Successfully sent output to Gateway. Will retrieve another task from the workflow in 5 seconds...')
                timer = threading.Timer(5, check_for_workflow_tasks, args=[])
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

if __name__ == "__main__":
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

                # while True:
                #     choice = input("Choose between your personal workflow(1) or the global workflow(2): [1-2] ")
                    
                #     if choice == '1':
                #         check_for_workflow_tasks()
                #         break
                #     elif choice == '2':
                #         check_for_tasks()
                #         break
                #     else:
                #         print("Invalid choice. Please enter 1 or 2.")

                # For now, we just check the personal workflow
                check_for_workflow_tasks()

            else:
                print("Error:", data.get('message'))
        except requests.exceptions.RequestException as error:
            print('Error validating script:', error)
    else:
        print("Worker Key was not found. Please add a WORKER_KEY to your .env file")