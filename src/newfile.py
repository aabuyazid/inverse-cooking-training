import pickle
import os

aux_data_dir = "/scratch/ntallapr/inversecooking/data"

for split in ['train', 'test', 'val']:
    test = os.path.join(aux_data_dir,"testing_"+split+".pkl")
    dataset = pickle.load(open(os.path.join(aux_data_dir, 'recipe1m_'+split+'.pkl'), 'rb+'))

    thelist = []
    with open(os.path.join(aux_data_dir, 'recipe1m_'+split+'.pkl'), 'rb+') as dataset:
        db = pickle.load(dataset)
        for keys in db:
            id = keys['id']
            id_str = str(id)
            if(id_str[0:2] == "00" or id_str[0:2] == "01"):
                thelist.append(keys)
    print('the number of images for {} is {}'.format(split, len(thelist)))
    #for i in range(len(thelist)):
    #    print(thelist[i])
    #    pickle.dump(thelist[i],f)
    #else:
    #    f.close()
    with open(test, 'wb+') as f:
        pickle.dump(thelist,f)

print('check testing.pkl')
