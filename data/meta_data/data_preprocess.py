import os
import tqdm
import ujson as json

train_all = []

# prefix = 'test_allmetadata_json/test'
prefix = 'train_allmetadata_json/train'
train_label = open(prefix+'_label.txt','r')
train_labels = train_label.readlines()

train_text = json.load(open(prefix+'_text.json','r'))
train_user_data = json.load(open(prefix+'_user_data.json','r'))
train_category = json.load(open(prefix+'_category.json','r'))
train_additional_information = json.load(open(prefix+'_additional_information.json','r'))
train_temporalspatial_information = json.load(open(prefix+'_temporalspatial_information.json','r'))

train_img_filepath = open(prefix+'_img_filepath.txt','r')
train_img_filepaths = train_img_filepath.readlines()


for i_t, (exp_text, exp_category, exp_add_info, exp_temp_info) in enumerate(zip(train_text, train_category, train_additional_information, train_temporalspatial_information)):
    data_sample = {}
    for key in exp_text:
        if(key not in data_sample):
            data_sample[key] = exp_text[key]
        else:
            assert data_sample[key] == exp_text[key]

    for key in exp_category:
        if(key not in data_sample):
            data_sample[key] = exp_category[key]
        else:
            assert data_sample[key] == exp_category[key]

    for key in exp_add_info:
        if(key not in data_sample):
            data_sample[key] = exp_add_info[key]
        else:
            assert data_sample[key] == exp_add_info[key]

    for key in exp_temp_info:
        if(key not in data_sample):
            data_sample[key] = exp_temp_info[key]
        else:
            assert data_sample[key] == exp_temp_info[key]

    assert train_user_data['Uid'][str(i_t)] == data_sample['Uid']

    # print(train_user_data['Pid'][str(i_t)], data_sample['Pid'])
    assert str(train_user_data['Pid'][str(i_t)]) == str(data_sample['Pid'])


    
    for key in train_user_data:
        if(key not in data_sample):
            data_sample[key] = train_user_data[key][str(i_t)]

    data_sample['img_filepaths'] = train_img_filepaths[i_t]
    # data_sample['label'] = train_labels[i_t]
    train_all.append(data_sample)
    # print(i_t)
    # print(exp_text, exp_category, exp_add_info, exp_temp_info)
    # xxx
    
f2 = open(prefix+"_all.json",'w',encoding='utf8')
json.dump(train_all,f2,indent=4,ensure_ascii=False)
f2.close()
print(len(train_all))