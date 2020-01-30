import pandas as pd
import requests
import time

df = pd.read_csv('insurance_qna_dataset.csv', delimiter='\t')

questions = df['Question']

print(len(questions))

base_url = 'https://quillbot.com/api/singleFlip?userID=N/A&text='

resposnes = []

for q in questions:
    q2 = "%20".join(q.split(' '))
    req = f'{base_url}' + q2 + '&strength=4&fthresh=1'

    r = requests.get(url = req)

    data = r.json()

    time.sleep(0.5)

    resposnes.append(data['flipped_alt'])

    print(data['flipped_alt'])

ser = pd.Series(resposnes).to_csv('paraphrazed.csv')

# print(ser[1])

# df2 = pd.DataFrame(ser)

# df2.to_csv('paraphrazed.csv')
