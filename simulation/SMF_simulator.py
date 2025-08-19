import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import random
import numpy as np
import torch
from utils import random_seed, sampled_indices, ConfigLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from abc import abstractmethod
from typing import Dict, Any
from typing import Union, List, Tuple, NamedTuple, TYPE_CHECKING
import ujson as json
import pandas as pd
import time
from string import Template
import re
from tqdm import tqdm
import numpy as np
from typing import List
from deffuant import Deffuant
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# simulation model
config = ConfigLoader("config.yaml")
model_name = config.get('model.model_path')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  
embedding_model_path = config.get('model.embedding_model')
embedding_model = SentenceTransformer(embedding_model_path)  

def batch_generate(prompts, max_length, max_new_tokens, temperature):
    all_texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_texts.append(text)


    model_inputs = tokenizer(
        all_texts,
        padding=True,          # 启用动态填充
        truncation=True,       # 启用截断
        max_length=768,       # 设置最大长度
        return_tensors="pt",
        pad_to_multiple_of=8,  # 对齐显存优化
        return_attention_mask=True
    ).to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,  # 明确指定填充token
        use_cache=True,                        # 启用KV缓存加速
        temperature=1.0,         # 温度越高，随机性越大（默认1.0，建议0.5~2.0）
        # top_k=0,                 # 禁用top_k过滤（允许所有词参与采样）
        # top_p=0.95,              # 使用核采样（nucleus sampling），累积概率阈值
        # num_beams=1,             # 禁用束搜索（beam search）
        # repetition_penalty=1.2,  # 惩罚重复内容（>1.0时降低重复）
    )

    generated_ids = [
        output_ids[input_len:] 
        for input_len, output_ids in zip(
            model_inputs.attention_mask.sum(dim=1),  # 用attention_mask获取真实长度
            generated_ids
        )
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def calculate_sampling_probabilities(post: str, users: List[str]) -> List[int]:

    all_texts = [post] + users
    embeddings = list(embedding_model.encode(all_texts))  

    post_embedding = embeddings[0]
    user_embeddings = embeddings[1:]
    
    post_vector = np.array(post_embedding)
    user_matrix = np.array(user_embeddings)
    similarities = np.dot(user_matrix, post_vector) / (
        np.linalg.norm(user_matrix, axis=1) * np.linalg.norm(post_vector)
    )
    
    exp_scores = np.exp(similarities - np.max(similarities))  
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities

def main():
    # parameter
    agent_num = config.get('simulation.agent_num')
    sample_num= config.get('simulation.sample_num')
    epoch_num = config.get('simulation.epoch_num')
    seed = config.get('simulation.random_seed')
    random_seed(seed)

    # data
    sim_data_path = config.get('data.sim_data')
    train_data = json.load(open(sim_data_path,'r'))
    profile_path = config.get('data.profile_data')
    agent_profiles = pd.read_csv(profile_path)
    role_descriptions = agent_profiles['role_description'].to_list()
    agent_names = agent_profiles['name'].to_list()
    relation_graph = agent_profiles['relation'].to_list()
    relation_graph = [eval(i) for i in relation_graph]
    # print(agent_profiles)
    
    abm_profiles = []
    following_info = {}
    ids = range(len(relation_graph))
    name2id = {}
    
    for agent_id, agent_name, agent_relation in zip(ids,agent_names,relation_graph):
        abm_profiles.append({
            "id": agent_id, 
            "name": agent_name, 
            "init_att": 0,
            "relation": agent_relation
        })
        following_info[agent_name] = agent_relation
        name2id[agent_name] = agent_id

    # simulation
    PROMPT = config.get('simulation.prompt')
    MF_PROMPT = config.get('simulation.mf_prompt')
    
    timestamp = 1457068974
    timeArray = time.localtime(timestamp)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    # output_parser = TwitterParser() 
    
    btz = config.get('model.batch_size')
    max_length = config.get('model.max_length')
    max_new_tokens = config.get('model.max_new_tokens')
    temperature = config.get('model.temperature')
    alpha = config.get('model.abm_alpha')
    bc_bound = config.get('model.abm_bc_bound')
    
    simulation_results = []
    MF_states = []
    max_retries = config.get('simulation.max_retries')  # 设置最大重试次数
    user_actions = ''
    abm_model = Deffuant(abm_profiles, alpha=alpha, bc_bound=bc_bound,following_info = following_info)
    
    for i_t, sample_data in tqdm(enumerate(train_data), total=len(train_data)):
        start_all_time = time.time()
        trigger_news = f"[Post: {sample_data['Title']}; Category: {sample_data['Category']}; Concept: {sample_data['Concept']}; Subcategory: {sample_data['Subcategory']}]\n"
        pid, uid = sample_data['Pid'], sample_data['Uid']
        
        probabilities = calculate_sampling_probabilities(trigger_news, role_descriptions[:agent_num])
        MF_state = 'None'
        results = []
        tmp_MF_states = []
        agent_memories= {}
        abm_model.reset()
        for a_n in range(agent_num):
            agent_memories[a_n] = []
        
        for ep in range(epoch_num):
            start_epoch_time = time.time()
            sampled_agents = sampled_indices(probabilities, all_num=agent_num, sample_num=sample_num)
            processed_prompts = []
            core_agent_names = []
            core_agent_atts = []
            for a_t in sampled_agents:
                processed_prompt = Template(PROMPT)
                data = {"agent_name": agent_names[a_t], 
                        "role_description": role_descriptions[a_t], 
                        # "current_time": datetime,
                        "memory": ';'.join(agent_memories[a_t][-3:]),
                        "trigger_news": trigger_news,
                        "MF": MF_state,
                        # "chat_history": "None",
                        # "tweet_page": "None",
                        # "info_box": "None"
                       }
                processed_prompt = processed_prompt.substitute(data)
                processed_prompts.append(processed_prompt)
                core_agent_names.append(agent_names[a_t])
                
                
            
            
            for step in range(int(sample_num/btz)+1):
                retry_count = 0
                success = False
                tmp_results = []
                btz_processed_prompts = processed_prompts[step*btz:(step+1)*btz]
                # print(btz_processed_prompts[0])
        
                while not success and retry_count < max_retries:
                    try:
                        
                        batch_output = batch_generate(btz_processed_prompts, max_length, max_new_tokens, temperature)
                        for output in batch_output:
                            # print(output)
                            # print("***")
                            output = output.lower()
                            reaction_type = ""
                            reaction = ""
                            interest_score = 0
                            # print(output)
                            # print("###")
            
                            thought = ''
                            try:
                                thought = output.split("thought:")[-1].split("action:")[0].strip()
                            except Exception as e:
                                print(f"fail to get thought: {str(e)} \n output:{output}")
            
                                
                            if "like(" in output:
                                reaction = ""
                                reaction_type = 'like'          
                            elif "do_nothing(" in output:
                                reaction, reaction_type = "", 'nothing'
                            elif "post(" in output:
                                action = output.split("action:")[-1].split("score:")[0].strip()  # 取 "Action: " 之后的部分
                                start = action.find('post("') + len('post("')
                                end = action.find('")', start)
                                reaction = action[start:end]
                                reaction_type = 'post'
                            elif "retweet(" in output:
                                action = output.split("action:")[-1].split("score:")[0].strip()  # 取 "Action: " 之后的部分
                                start = action.find('retweet("') + len('retweet("')
                                end = action.find('")', start)
                                reaction = action[start:end]
                                reaction_type = 'retweet'
                            elif "reply(" in output:
                                action = output.split("action:")[-1].split("score:")[0].strip()  # 取 "Action: " 之后的部分
                                start = action.find('reply("') + len('reply("')
                                end = action.find('")', start)
                                reaction = action[start:end]
                                reaction_type = 'reply'
                            else:
                                raise Exception(
                                    f"no valid parsed_response detected, "
                                    f"cur response {output}"
                                )
                            if(reaction_type!='nothing'):
                                for neighbor in relation_graph[a_t]:
                                    # if(agent_memories[name2id[neighbor]].count(';')>=3):
                                    #     continue
                                    agent_memories[name2id[neighbor]].append(f"{agent_names[sampled_agents[step]]} {reaction_type} the post: {reaction}")
                                
                            score_part = output.split('score:')[-1].strip()
                            match = re.search(r'^\d+', score_part)  # 匹配开头的连续数字
                            if not match:
                                raise ValueError(f"Score格式错误: {score_part}")
                            interest_score = int(match.group())
                            
                            result = {
                                "order_id": i_t,
                                "pid": pid,
                                "uid": uid,
                                "thought": thought,
                                "reaction_type": reaction_type,
                                "reaction_text": reaction,
                                "interest_score": interest_score
                            }
                            tmp_results.append(result)
                            core_agent_atts.append(interest_score)
                        success = True  # 标记为成功
                    except Exception as e:
                        print(f"Step {step} failed (retry {retry_count+1}/{max_retries}): {str(e)} \n output:{output}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"Failed after {max_retries} attempts; please check the model output format.")
                        
                        
                    if retry_count >= max_retries:
                        tmp_results = []
                        for i in range(len(btz_processed_prompts)):
                            tmp_results.append({
                                "order_id": i_t,
                                "pid": pid,
                                "uid": uid,
                                "reaction_type": 'nothing',
                                "reaction_text": '',
                                "interest_score": 0
                            })
                        core_agent_atts = [-1]*len(btz_processed_prompts)
                results.extend(tmp_results)
            # if(ep==epoch_num-1):
            #     break
            # 更新文本社交平均场
            user_actions = ''
            for i_r, tmp_re in enumerate(results[-sample_num:]):
                user_actions+=f'''
                user {i_r}:\n
                reaction_type: {tmp_re['reaction_type']}\n
                reaction_text: {tmp_re['reaction_text']}\n
                interest_score: {tmp_re['interest_score']}\n
                '''
            processed_MF_prompt = Template(MF_PROMPT)
            data = {"trigger_news": trigger_news,
                    "user_actions": user_actions,
                   }
            processed_MF_prompt = processed_MF_prompt.substitute(data)
            MF_state = batch_generate([processed_MF_prompt], max_length, max_new_tokens, temperature)[0]
    
            # 更新数值社交平均场
            for i in range(len(sampled_agents)):
                if(core_agent_atts[i]!=-1):
                    abm_model.update_mirror(core_agent_names[i], core_agent_atts[i])
                
            abm_model.step(sampled_agents)
            all_atts = abm_model.get_attitudes()
            
            tmp_MF_states.append({"pid": pid,
                                 "epoch": ep,
                                "MF_state": MF_state,
                                "all_atts": all_atts,
                                "mean_atts:": sum(all_atts) / len(all_atts)})
            end_epoch_time = time.time()
            # print("epoch time:",end_epoch_time-start_epoch_time)
            # if(ep==epoch_num-1):
            #     print("PROMPT:", processed_MF_prompt)
            #     print("tmp_MF_states:", tmp_MF_states)
            #     print('*'*20)
        simulation_results.append(results)
        MF_states.append(tmp_MF_states)
        # print(trigger_news)
        # print(MF_states)
        end_all_time = time.time()
    
        # print("all time:",end_all_time-start_all_time)
        # xxx
        if(i_t%2500==0):
            with open(f"simulation_result/MF_simulation_results_{i_t}.json", "w", encoding="utf-8") as f:
                json.dump(simulation_results, f, ensure_ascii=False, indent=4) 
            with open(f"simulation_result/MF_simulation_states_{i_t}.json", "w", encoding="utf-8") as f:
                json.dump(MF_states, f, ensure_ascii=False, indent=4)  
    
    
    # In[ ]:
    
    
    with open(f"simulation_result/MF_simulation_results_all.json", "w", encoding="utf-8") as f:
        json.dump(simulation_results, f, ensure_ascii=False, indent=4) 
    with open(f"simulation_result/MF_simulation_states_all.json", "w", encoding="utf-8") as f:
        json.dump(MF_states, f, ensure_ascii=False, indent=4) 

if __name__ == "__main__":
    main()