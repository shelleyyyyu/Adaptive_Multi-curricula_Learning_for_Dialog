import numpy as np
from scipy.spatial import distance
import pickle
import torch.nn.functional as F
import torch
import sys


def main(fname):
        embedding_1d = []
        embedding = []
        with open(fname, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            print(len(data))
            print(type(data[0]))
            print((data[0]).size())
            for d in data:
                embedding.append(d)
                embedding_1d.append(torch.sum(d, 0))

        with open('vec.tsv', 'w', encoding='utf-8') as vec_file, open('label.tsv', 'w', encoding='utf-8') as label_file:
            for idx, emb in enumerate(embedding_1d):
                tmp_list = []
                for e in np.array(emb.cpu()):
                    tmp_list.append(str(e))
                vec_file.write('\t'.join(tmp_list)+'\n')
                label_file.write(str(idx)+'\n')


        # for index in range(len(embedding)):
        #     if index == len(embedding)-1:
        #         continue
        #     sim = F.cosine_similarity(embedding[index], embedding[index+1]).abs().mean()
        #     print(index, '-', index+1, sim)

        # print('-'*10)

        # sim_array = []
        # for index in range(len(embedding)):
        #     if index == 0:
        #         continue
        #     sim = distance.cosine(embedding_1d[0], embedding_1d[index])
        #     sim_array.append(float('%.6f'%sim))
        #     print(0, '-', index, float('%.6f'%sim))

        # print(np.argsort(sim_array))

if "__main__" == __name__:
    main(sys.argv[1])
