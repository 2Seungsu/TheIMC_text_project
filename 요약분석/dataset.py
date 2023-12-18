import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels)


# __init__(self, encodings, labels):
# 초기화
# encodings: 피쳐(문장)을 토큰화하고 처리된 입력 데이터.
# labels: 해당 라벨(타겟) 즉 요약문. 


# __getitem__ :
# 주어진 인덱스(idx)에서 단일 데이터 샘플을 검색하는 데 사용
# 인코딩 키에서 키를 가져오는 사전(항목)을 생성합니다.
# 각 키에 대해 인코딩에 있는 해당 값의 idx 번째 요소에서 텐서를 생성합니다.
# 사전에 'labels' 키를 추가하고, 값은 labels에 있는 'input_ids' 키의 idx 번째 요소에서 생성된 텐서.


# __len__ :
# 데이터 세트의 총 샘플 수를 반환하며 labels의 'input_ids' 키 길이에 따라 결정.