import argparse
import ast
import datetime
import json
import os
import re
import sys
import time
import yaml
import tqdm

import prompts
from model import parse_explore_rsp, get_openai_response, get_qwen_response, parse_multi_explore_rsp, get_gpt4v_response, get_qwenmax_response, get_model_response_qwen, get_InternVL_response
from ..utils import *

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--task_file', type=str, help='task_file', default=r'simple_tasks_sample.json')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, help='config path', default=r'config.yaml')
    parser.add_argument('--task_type', type=str, help='task type', default=r'multi_simple')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--max_rounds', type=int, help='max rounds', default=20)
    return parser.parse_args()

args = get_parse()
data_dir = args.data_dir
MAX_ROUNDS = args.max_rounds
#读取config文件
with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

if args.model_type == 'Qwen':
    API_url = config[args.model_name]['QWEN_API_BASE']
#得到API_url
elif args.model_type == 'OpenAI':
    API_url = config[args.model_name]['OPENAI_API_BASE']
    token = config[args.model_name]['OPENAI_API_KEY']
    temperature = config[args.model_name]['TEMPERATURE']
    max_tokens = config[args.model_name]['MAX_TOKENS']

task_path = os.path.join(args.data_dir, args.task_file)

with open(task_path, "r") as f:
    tasks = json.load(f)
    
all_time = 0

for i, task in tqdm.tqdm(enumerate(tasks[args.start_num:], start=args.start_num)):
    final_page_name = task['name']
    task_info = task['task']

    task_desc = task_info.split('\n')[-1]

    #get the task description step by step
    multi_task_desc = []
    steps = re.findall(r'^\d+\.\s(.*)', task_info, re.MULTILINE)
    for step in steps:
        multi_task_desc.append(step)

    #get the path of data
    init_data_path = find_dir_with_prefix(data_dir, final_page_name.split('0')[0]+'_')
    task_dir = os.path.join(data_dir, init_data_path)

    if not os.path.exists(task_dir):
        continue

    #initial settings
    round_count = 0
    last_act = "None"
    task_complete = False

    all_action_id_file = os.path.join(task_dir, 'all_action_id.json')
    #get all action id 
    with open(all_action_id_file, "r", encoding='UTF-8') as fp:
        all_action_ids = json.load(fp)
    all_action_ids = json.loads(all_action_ids)

    #convert the action id to a dictionary
    all_actions_list = []
    for k, v in all_action_ids.items():
        action = {'action_info': k, 'action_id': v}
        all_actions_list.append(action)

    #get all page actions
    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    all_page_actions_data = all_page_actions['data']

    with open(os.path.join(task_dir, 'all_page_id.json')) as fp:
        all_page_ids = json.load(fp)
    all_page_ids = json.loads(all_page_ids)

    with open(os.path.join(task_dir, 'all_triple.json')) as fp:
        all_page_triples = json.load(fp)['data']
    all_page_convert = {}
    for all_page_triple in all_page_triples:
        all_page_convert[all_page_triple[0]+'act'+str(all_page_triple[1])] =all_page_triple[2]

    current_page_actions = {}

    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    for current_page_data in all_page_actions['data']:
        current_page_actions[current_page_data['name']] = current_page_data['action_valid']

    current_page_name = final_page_name.split('_')[0]

    action_history = []
    page_history = []

    ans_action_id = []
    ans_action_info = []
    ans_history_pages = []
    gt_page_name = []
    task_time = 0
    before_time = time.time()

    while round_count < MAX_ROUNDS:
        round_count += 1
        print_with_color(f"Round {round_count}", "yellow")

        screenshot_path = os.path.join(task_dir, current_page_name, current_page_name + '-screen.png')
        html_file = os.path.join(task_dir, current_page_name, current_page_name + '-html.txt')
        xml_file = os.path.join(task_dir, current_page_name, current_page_name + '-xml.txt')

        with open(html_file, 'r', encoding='utf-8') as f:  # html
            html_content = f.read()
        with open(xml_file, 'r', encoding='utf-8') as f:  # xml
            xml_content = f.read()
        current_page_all_action_ids = current_page_actions[current_page_name]
        current_action_infos = []
        for id in current_page_all_action_ids:
            current_action_infos.append(all_actions_list[int(id)]['action_info'])

        click_actions, input_actions_pre, scroll_actions, current_page_all_actions = actions_generate(html_content, xml_content)
        click_actions, input_actions, scroll_actions, scroll_action_bounds = current_actions_generate(current_action_infos, click_actions, input_actions_pre, scroll_actions, current_page_all_actions)

        drawn_screenshot = os.path.join(task_dir, current_page_name, current_page_name + f'_labeled_multi.png')

        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)


        imgcv = cv2.imread(drawn_screenshot)
        imgcv = cv2.resize(imgcv, (1080, 2400))
        cv2.imwrite(drawn_screenshot, imgcv)

        prompt = re.sub(r"<ui_document>", "", prompts.multipath_task_template)
        prompt = re.sub(r"<task_description>", task_desc, prompt)

        
        task_count = current_page_name.count('_')
        if task_count < len(multi_task_desc) - 1:
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[task_count], prompt)
        else:
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[len(multi_task_desc)-1], prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)
        print_with_color("Thinking about what to do in the next step...", "yellow")

        if args.model_name == 'qwen272B' or args.model_name == 'qwen272B_1':
            status, rsp = get_model_response_qwen(prompt, [drawn_screenshot])
        elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
            status, rsp = get_openai_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'gpt-4v' or args.model_name == 'gpt-4v-1':
            status, rsp = get_gpt4v_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'qwen-vl-max':
            status, rsp = get_qwenmax_response(prompt, [drawn_screenshot], args.model_name, API_url)
        elif args.model_name == 'InternVL2':
            status, rsp = get_InternVL_response(prompt, drawn_screenshot)
        if status:

            res = parse_multi_explore_rsp(rsp)
            act_name = res[0]
            
            last_act = res[-1]
            res = res[:-1]

            if act_name == "click":
                _, area = res

                current_page_name, action_info, action_id = action_click(click_actions, area, current_page_all_actions, all_action_ids, current_page_name, all_page_convert)

                if action_info == "ERROR":
                    ans_action_info.append(area + "click error")
                    ans_action_id.append(action_id)  
                    print_with_color("ERROR: click execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "scroll":
                _, area, direction = res
                print(area)
                print(len(scroll_actions))
                current_page_name, action_info, action_id = action_scroll(scroll_action_bounds, area, direction, all_action_ids, current_page_name, all_page_convert)
               
                if action_info == "ERROR":
                    ans_action_info.append(area + direction + "scroll error")
                    ans_action_id.append(action_id)  
                    print_with_color("ERROR: scroll execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "input":
                _, text = res
                current_page_name, action_info, action_id = action_input(text, input_actions, current_page_all_actions, all_action_ids, current_page_name, all_page_convert)

                if action_info == "ERROR":
                    ans_action_info.append('input' + text + "error")
                    ans_action_id.append(action_id)
                    print_with_color("ERROR: input execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "back":
                current_page_name = current_page_name.rsplit('_', 1)[0]
                ans_action_id.append(-1)
                ans_action_info.append('back')
            else:
                ans_action_id.append(-3)
                ans_action_info.append('no such action error')
            ans_history_pages.append(current_page_name)
        else:
            ans_history_pages.append(current_page_name)
            ans_action_info.append(rsp)
            ans_action_id.append(-3)
            print_with_color('error', "red")
        if current_page_name == final_page_name:
            break

    action_and_page = zip(ans_action_id, ans_action_info, ans_history_pages)

    result_path = os.path.join(args.save_path, args.task_type + args.model_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    with open(os.path.join(data_dir, result_path, f'{final_page_name}_{i}.txt'), "w", encoding="utf-8") as f:
        for action_id, action_info, history_page in action_and_page:
            f.write(f"{action_id}: {action_info}:{history_page}\n")
        if task_complete:
            f.write("Task completed successfully")
        elif round_count == MAX_ROUNDS:
            f.write("Task finished due to reaching max rounds")
        else:
            f.write("Task failed")

    if task_complete:
        print_with_color("Task completed successfully", "yellow")
    elif round_count == MAX_ROUNDS:
        print_with_color("Task finished due to reaching max rounds", "yellow")
    else:
        print_with_color("Task finished unexpectedly", "red")

