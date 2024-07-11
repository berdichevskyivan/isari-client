import requests
import threading
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

model = None
tokenizer = None

quantization_4bit_config = BitsAndBytesConfig(load_in_4bit=True)
quantization_8bit_config = BitsAndBytesConfig(load_in_8bit=True)

def load_model_and_tokenizer(model_path):

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    # Quantize the model when loading it for better inference times, but sadly, less performance AKA Intelligence
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", quantization_config=quantization_8bit_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def run_inference(input_text):
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
            "max_new_tokens": 600, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        }
        
        output = pipe(messages, **generation_args)
        generated_text = output[0]['generated_text']

        # # Split the text by "},"
        # parts = generated_text.split("},")

        # # Check if there are four or more occurrences
        # if len(parts) >= 4:
        #     # Join the first four parts with "},", then add the closing bracket
        #     limited_text = "},".join(parts[:4]) + "}]"
        # else:
        #     # If fewer than four parts, join all parts and add the closing bracket
        #     limited_text = "},".join(parts) + "}]"

        return generated_text
    except Exception as e:
        return e

def check_for_tasks(worker_id):
    try:
        check_task_url = 'http://localhost/checkForTask'
        response = requests.post(check_task_url, json={'workerId': worker_id})
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
                    response = requests.post('http://localhost/storeCompletedTask', json={'output': output, 'worker_id': worker_id, 'task_id': task_id})
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
                        timer = threading.Timer(5, check_for_tasks, args=[worker_id])
                        # Start the timer
                        timer.start()
                else:
                    # If the task_id doesn't match, delete the cache file
                    print('Cache file exists but it doesnt match the task id. Removing cache file...')
                    os.remove('cache.json')
            else:
                role = data.get('result').get('taskType').get('role')
                task_name = data.get('result').get('taskType').get('name')
                task_description = data.get('result').get('taskType').get('description')
                subject_name = data.get('result').get('issue').get('name')
                subject_description = data.get('result').get('issue').get('description')
                instructions = next((inst.get('instruction') for inst in data.get('result').get('instructions') if inst.get('instruction_type') == 'output'), '')

                # Here we prepare the input text
                input_text = (f"Assume this role: {role}\n"
                            f"You must perform this task: {task_name}\n"
                            f"This task consists of: {task_description}\n"
                            f"This is the subject: {subject_name}\n"
                            f"This is a brief description of the subject: {subject_description}\n"
                            f"These are your output instructions: {instructions}\n")
                
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
                generated_text = run_inference(input_text)

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
                response = requests.post('http://localhost/storeCompletedTask', json={'output': formatted_output, 'worker_id': worker_id, 'task_id': task_id})
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
                    timer = threading.Timer(5, check_for_tasks, args=[worker_id])
                    # Start the timer
                    timer.start()
        else:
            if data.get('error_code') == 'ACTIVE_TASK':
                print('Worker already has a task assigned. Checking for the cache.json file...')

                task_id = data.get('task_id')

                if os.path.exists('cache.json'):
                    # Load the cached data
                    with open('cache.json', 'r') as cache_file:
                        cache_data = json.load(cache_file)
                    
                    # Check if the task_id matches
                    if cache_data.get('task_id') == task_id:
                        # Bypass inference and send the cached output to the Gateway
                        print({'success': True, 'output': cache_data.get('output')})
                        print("We have the correct cached file. Sending output to the Gateway server...")
                        output = cache_data.get('output')
                        response = requests.post('http://localhost/storeCompletedTask', json={'output': output, 'worker_id': worker_id, 'task_id': task_id})
                        response.raise_for_status()
                        # Wait for response
                        data = response.json()
                        print('data returned from storeCompletedTask', data)
                    else:
                        print('Task Id received doesn\'t match the task id in the cache.json file. Deleting cache.json file')
                        os.remove('cache.json')
                else:
                    print('There is no cache.json file. Please ask the admins to change the status of this task.')
            elif data.get('error_code') == 'NO_MORE_TASKS':
                print("There are no more tasks. Stopping execution.")
            else:
                print("Error:", data.get('message'))
    except requests.exceptions.RequestException as error:
        print('Error checking for tasks:', error)
        if os.path.exists('cache.json'):
            os.remove('cache.json')

if __name__ == "__main__":
    worker_id = 20
    check_for_tasks(worker_id)