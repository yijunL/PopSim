import ujson as json
EPO=2
data_state = json.load(open("./MF_simulation_states_all.json",'r'))

def MF_data_con(train_data,output_file):
    train_data_new = []
    for i in train_data:
        i['input'] = i['input'].split('The popularity score of this post is: ')[0]+'The social network state about this post is: '+data_state[i['pid']][EPO]['MF_state']+' With the average intresting score is: '+f"{data_state[i['pid']][EPO]['mean_atts:']:.2f}. "+'The popularity score of this post is: '
        train_data_new.append(i)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(train_data_new, f, ensure_ascii=False, indent=4)  # indent 用于美化格式

train_data_mini = json.load(open("./dataset_time/SMP_train_mini_mm_time.json",'r'))
train_data = json.load(open("./dataset_time/SMP_train_mm_time.json",'r'))
val_data = json.load(open("./dataset_time/SMP_val_mm_time.json",'r'))
test_data = json.load(open("./dataset_time/SMP_test_mm_time.json",'r'))

MF_data_con(train_data_mini,"./dataset_MF_time/SMP_train_mini_mm_MF_time.json")
MF_data_con(train_data,"./dataset_MF_time/SMP_train_mm_MF_time.json")
MF_data_con(val_data,"./dataset_MF_time/SMP_val_mm_time.json")
MF_data_con(test_data,"./dataset_MF_time/SMP_test_mm_MF_time.json")