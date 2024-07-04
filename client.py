import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model = None
tokenizer = None

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
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
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        }
        
        output = pipe(messages, **generation_args)
        generated_text = output[0]['generated_text']
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
            # Here we prepare the input text
            role = data.get('result').get('taskType').get('role')
            task_name = data.get('result').get('taskType').get('name')
            task_description = data.get('result').get('taskType').get('description')
            subject_name = data.get('result').get('issue').get('name')
            subject_description = data.get('result').get('issue').get('description')
            instructions = next((inst.get('instruction') for inst in data.get('result').get('instructions') if inst.get('instruction_type') == 'output'), '')

            input_text = (f"Assume this role: {role}\n"
                        f"You must perform this task: {task_name}\n"
                        f"This task consists of: {task_description}\n"
                        f"This is the subject: {subject_name}\n"
                        f"This is a brief description of the subject: {subject_description}\n"
                        f"These are your output instructions: {instructions}")

            print(input_text)
            generated_text = run_inference(input_text)
            print({'success': True, 'output': generated_text})
        else:
            print("Error:", data.get('message'))
    except requests.exceptions.RequestException as error:
        print('Error checking for tasks:', error)

if __name__ == "__main__":
    worker_id = 20
    check_for_tasks(worker_id)
