import os

def tmp_write(file_num, lst):
    file_path = f'/nouns_to_generate_{file_num}.txt'
    f = open(directory_path + file_path, 'w')
        
    for idx, noun in enumerate(lst, 1):
        if idx != len(lst):
            f.write(noun + '\n')
        else: # 마지막 줄에 줄바꿈 들어가지 않게 필터링
            f.write(noun)
        
    f.close()

def make_file():
    '''
    26366 / 32 = 823.xxx지만 
    823으로 32개를 만들 경우 남아있는 단어들이 생기므로 
    824로 나눠 31개로 나눈 후 마지막에 남은 단어들로 32번째 파일 생성
    '''

    tmp = [] # 단어 담아둘 임시 리스트
    
    for idx, noun in enumerate(nouns, 1):
        tmp.append(noun)
        
        if (idx % 824 == 0): 
            tmp_write(idx // 824, tmp)
            tmp = []
            print(f'{idx // 824}번째 파일 생성 완료')

    tmp_write(32, tmp) # 남아있는 단어들로 마지막 파일 생성
    print('32번째 파일 생성 완료')
    

if __name__ == "__main__":    
    with open('nouns_to_generate.txt', 'r') as file:
            nouns = list(map(lambda x: x.strip(), file.readlines()))
    
    directory_path = './nouns_batch' # 32개씩 나눈 배치 저장할 폴더 경로

    if not os.path.isdir(directory_path): # 폴더 없으면 생성
        os.mkdir(directory_path)
        
    make_file()



