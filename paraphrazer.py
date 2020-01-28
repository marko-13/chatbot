import time

import requests
import csv

def send_req(question):
    question_parts = question.split(" ")
    text = ""
    brojac = 0
    for q_p in question_parts:

        if q_p != '?':
            if len(q_p) !=brojac:
                text = text + q_p + '%20'
                brojac += brojac
            else:
                text = text + q_p
                brojac += brojac

    cookie = {'cookie': '__cfduid=d545ae89701cd204d72f0a3102c09a6641580215561; _ga=GA1.2.1920374839.1580215562; _gid=GA1.2.399999151.1580215562; connect.sid=s%3AaZCmzz5N2X0x-Ri5n6Rl3_ogO-GlGx7W.zk%2BkqW9OOLxFRIWvWRSdP%2Flh6uFdeUhTzXYnjSS35nE; __stripe_mid=d437c0c2-bedc-417a-bd5e-ffc80394f899; userIDToken=eyJhbGciOiJSUzI1NiIsImtpZCI6IjI1OTc0MmQyNjlhY2IzNWZiNjU3YzBjNGRkMmM3YjcyYWEzMTRiNTAiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vcGFyYXBocmFzZXItNDcyYzEiLCJhdWQiOiJwYXJhcGhyYXNlci00NzJjMSIsImF1dGhfdGltZSI6MTU4MDIxNzY5NCwidXNlcl9pZCI6ImhqUDd3RUhWUE9lVnpHdEFMaVg5VWFrVE5MNTIiLCJzdWIiOiJoalA3d0VIVlBPZVZ6R3RBTGlYOVVha1ROTDUyIiwiaWF0IjoxNTgwMjE3Njk1LCJleHAiOjE1ODAyMjEyOTUsImVtYWlsIjoiZDEzNzcyNDVAdXJoZW4uY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImVtYWlsIjpbImQxMzc3MjQ1QHVyaGVuLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6InBhc3N3b3JkIn19.ado0Z6z0wtQprWjldwipj2ugVv6TXUeqYjIESh4Y5ExNSU7yve7xzliSQ2FkI06hj_zM2eRHNSeJHGWY5_UhqCiPtViqvI5M_B5Qy8xdMVvIloAatT2JkCEZRQ2YvBZivt-iNUcw6dlR-juMaXREqenDdOjCQUdDTOakWmQdDuiOxSDC4Mur2bzVpP7-YOpsiIsBXcPONYMfmZUOeC4X9k__4ZKkEwghz6LlzxOr7MpBtioJ6m9mS9yrC-PuB52yaU_3kYrr43yV0qsRQNg8RRKr23_WFBDLSD8gEnV3Vt8rP4djHo52hSGXIUVOWYOul8hvpHMagftsQKl4guJm_Q; authenticated=true; premium=false; quid=hjP7wEHVPOeVzGtALiX9UakTNL52; prioritize=4; fthresh=0; _gat=1; amplitude_id_6e403e775d1f5921fdc52daf8f866c66quillbot.com=eyJkZXZpY2VJZCI6IjQxMThmMjUwLWYzNzgtNGM1ZS05ZjE4LTI2MWY2YmQ2NTI2OFIiLCJ1c2VySWQiOiJkMTM3NzI0NUB1cmhlbi5jb20iLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE1ODAyMTk2MzkwNjksImxhc3RFdmVudFRpbWUiOjE1ODAyMTk2MzkwNjksImV2ZW50SWQiOjE1LCJpZGVudGlmeUlkIjoxLCJzZXF1ZW5jZU51bWJlciI6MTZ9'}

    r = requests.get('https://quillbot.com/api/singleFlip?userID=N/A&text=' + text + '&strength=4&autoflip=false&wikify=false&fthresh=-1',
                     cookies = cookie)


    print(text)
    ret = r.json()

    print(ret['flipped_alt'])
    print('\n\n')

    with open('log/para.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow([question, ret['flipped_alt']])

    time.sleep(1)
