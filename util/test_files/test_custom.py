import sys
sys.path.append('../')
import workflow as wf

scores, pred, test_set = wf.Run.custom_validation(train_dataset = '../data/train_set.csv', test_dataset = '../data/test_set.csv', model = 'GC')

print(scores)
