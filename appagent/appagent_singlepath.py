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

from model import parse_explore_rsp, get_qwen_response, get_openai_response, get_qwenmax_response, get_llama_response, parse_llama_explore_rsp, get_gpt4v_response, get_cogagent_response, get_InternVL_response, get_llava_response
from ..utils import *


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, help='config path', default=r'config.yaml')
    parser.add_argument('--task_type', type=str, help='task type', default=r'single_simple')
    parser.add_argument('--task_file', type=str, help='task file', default=r'complex_tasks_sample.json')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default='Qwen')
    parser.add_argument('--save_path', type=str, help='save results path')
    return parser.parse_args()


args = get_parse()
with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

if args.model_type == 'Qwen':
    API_url = config[args.model_name]['QWEN_API_BASE']
elif args.model_type == 'OpenAI':
    API_url = config[args.model_name]['OPENAI_API_BASE']
    token = config[args.model_name]['OPENAI_API_KEY']
    temperature = config[args.model_name]['TEMPERATURE']
    max_tokens = config[args.model_name]['MAX_TOKENS']
elif args.model_type == 'LLAMA':
    API_url = config[args.model_name]['LLAMA_API_BASE']
elif args.model_type == 'cogagent':
    API_url = config[args.model_name]['COAGENT_API_BASE']
else:
    API_url = None

all_time = 0
task_path = os.path.join(args.data_dir, args.task_file)

with open(task_path, "r") as f:
    tasks = json.load(f)

for i, task in tqdm.tqdm(enumerate(tasks[args.start_num:], start=args.start_num)):
    final_page_name = task['name']
    task_info = task['task']

    task_desc = task_info.split('\n')[-1]

    multi_task_desc = []
    steps = re.findall(r'^\d+\.\s(.*)', task_info, re.MULTILINE)
    for step in steps:
        multi_task_desc.append(step)

    init_data_path = find_dir_with_prefix(args.data_dir, final_page_name.split('0')[0] + '_')
    task_dir = os.path.join(args.data_dir, init_data_path)

    if not os.path.exists(task_dir):
        continue

    round_count = 0
    last_act = "None"
    gt_act = "None"
    task_complete = False

    all_action_id_file = os.path.join(task_dir, 'all_action_id.json')

    with open(all_action_id_file, "r", encoding='UTF-8') as fp:
        all_action_ids = json.load(fp)
    all_action_ids = json.loads(all_action_ids)

    all_actions_list = []
    for k, v in all_action_ids.items():
        action = {'action_info': k, 'action_id': v}
        all_actions_list.append(action)

    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    all_page_actions_data = all_page_actions['data']

    current_page_actions = {}

    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    for current_page_data in all_page_actions['data']:
        current_page_actions[current_page_data['name']] = current_page_data['action_valid']

    with open(os.path.join(task_dir, 'all_page_id.json')) as fp:
        all_page_ids = json.load(fp)
    all_page_ids = json.loads(all_page_ids)

    with open(os.path.join(task_dir, 'all_triple.json')) as fp:
        all_page_triples = json.load(fp)['data']
    all_page_convert = {}
    for all_page_triple in all_page_triples:
        all_page_convert[all_page_triple[0] + 'act' + str(all_page_triple[1])] = all_page_triple[2]

    current_page_name = final_page_name.split('_')[0]

    action_history = []
    page_history = []

    ans_action_id = []
    ans_action_info = []

    gt_page_name = []
    task_time = 0
    before_time = time.time()

    for round_count, page_id in enumerate(final_page_name.split('_')[1:]):
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

        click_actions, input_actions, scroll_actions, current_page_all_actions = actions_generate(html_content,
                                                                                                  xml_content)
        input_actions = input_generate(input_actions, current_action_infos, current_page_all_actions)

        drawn_screenshot = os.path.join(task_dir, current_page_name, current_page_name + f'_labeled.png')
        scroll_action_bounds = get_scroll_bounds(scroll_actions)

        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)


        imgcv = cv2.imread(drawn_screenshot)
        imgcv = cv2.resize(imgcv, (1080, 2400))
        cv2.imwrite(drawn_screenshot, imgcv)


        prompt = re.sub(r"<ui_document>", "", prompts.singlepath_task_template)
        prompt = re.sub(r"<task_description>", task_desc, prompt)
        if round_count > len(multi_task_desc):
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[-1], prompt)
        else:
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[round_count - 1], prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)

        print_with_color("Thinking about what to do in the next step...", "yellow")

        if args.model_name == 'qwen272B':
            status, rsp = get_qwen_response(prompt, [drawn_screenshot], API_url)
        elif args.model_name == 'gpt-4o':
            status, rsp = get_openai_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'llama90B':
            status, rsp = get_llama_response(prompt, [drawn_screenshot], args.model_name, API_url)
        elif args.model_name == 'qwen-vl-max':
            status, rsp = get_qwenmax_response(prompt, [drawn_screenshot], args.model_name, API_url)
        elif args.model_name == 'gpt-4v':
            status, rsp = get_gpt4v_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'cogagent':
            status, rsp = get_cogagent_response(prompt, drawn_screenshot)
        elif args.model_name == 'InternVL2':
            status, rsp = get_InternVL_response(prompt, drawn_screenshot)
        elif args.model_name == 'llava':
            status, rsp = get_llava_response(prompt, drawn_screenshot)
        

        if status:

            if args.model_name == 'llama90B':
                res = parse_llama_explore_rsp(rsp)
            # elif args.model_name == 'coagent':
            #     res = pasrse_coagent_rsp(rsp)
            else:
                res = parse_explore_rsp(rsp)
            
            act_name = res[0]

            if act_name == "click":
                area = res[1]
                try:
                    area_idx = int(re.findall(r'(\d+)', area)[0])
                    action_info = current_page_all_actions[click_actions[area_idx - 1]]
                    click_action = 'click(' + action_info + ')'
                    action_id = all_action_ids[click_action]
                except:
                    action_info = area + 'click_error'
                    action_id = -2
                ans_action_info.append(click_action)
                ans_action_id.append(action_id)
                current_page_name = current_page_name + f'_{page_id}'
                gt_page_name.append(current_page_name)

            elif act_name == "scroll":
                area, direction = res[1], res[2]
                
                try:
                    area_idx = int(re.findall(r'(\d+)', area)[0])
                    action_info = dict_scroll_parameters(scroll_actions[area_idx - 1], direction)
                    scroll_action = "scroll(" + str((action_info)) + ")"
                    action_info = scroll_action
                    action_id = all_action_ids[scroll_action]
                except:
                    action_info = area + direction + 'scroll_error'
                    action_id = -2
                ans_action_info.append(action_info)
                ans_action_id.append(action_id)
                current_page_name = current_page_name + f'_{page_id}'
                gt_page_name.append(current_page_name)

            elif act_name == "input":
                text = res[1]
                input_action = None
                _, action_info, action_id = action_input(text, input_actions, current_page_all_actions, all_action_ids,
                                                         current_page_name, all_page_convert)
                ans_action_info.append(action_info)
                ans_action_id.append(action_id)
                current_page_name = current_page_name + f'_{page_id}'
                gt_page_name.append(current_page_name)
            else:
                ans_action_info.append('no such action')
                ans_action_id.append(-3)
                current_page_name = current_page_name + f'_{page_id}'
                gt_page_name.append(current_page_name)
            # time.sleep(30)
        
        else:
            print_with_color(rsp, "red")
            continue
    
    result_path = os.path.join(args.save_path, args.task_type + args.model_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('task completed')
    action_and_page = zip(ans_action_id, ans_action_info, gt_page_name)
    with open(os.path.join(result_path, f'{final_page_name}_{i}.txt'), "w", encoding="utf-8") as f:
        for action_id, action_info, page_name in action_and_page:
            f.write(f"{action_id}: {action_info}: {page_name}\n")
