from selenium import webdriver
import pandas as pd
import time

driver = webdriver.Chrome('/home/mihailo/chrome-driver/chromedriver')

# TODO: Basic petlja radi. Ako recaptcha iskace stalno, promeniti mejl.
#       Password je mail. Trebalo bi cuvati podatke u neki csv

url = 'https://quillbot.com/'

garbage_mail = 'd1382372@urhen.com'

driver.get(url)

# Click login button
login_btn_class = 'login-btn'

login_btn = driver.find_elements_by_class_name(login_btn_class)[0]


login_btn.click()

# Enter mail/pw into fields
email_field = driver.find_element_by_id('username')

email_field.send_keys(garbage_mail)

password_field = driver.find_element_by_id('password')

password_field.send_keys(garbage_mail)

# Click login

submit_login_btn = driver.find_element_by_id('login-button')
submit_login_btn.click()

# Select creative

# Wait for login to happen
time.sleep(5)

creative_radio = driver.find_element_by_xpath('//*[@id="everythingContainer"]/div/div/div[2]/div[3]/div[1]/div[1]/div/div[2]/div[2]/div/div[2]/div[2]/div[4]')
creative_radio.click()

# Select the input field, output field and action button

action_btn = driver.find_element_by_id('paraphraseBtn')

input_field = driver.find_element_by_id('inputText')

# All of the output words are in SPAN's with the class 'word'

# import dataset

df = pd.read_csv('insurance_qna_dataset.csv', delimiter='\t')

questions = df['Question']

paraphrazed_questions = []

for question in questions[:100]:
    input_field.clear()
    input_field.send_keys(question)

    action_btn.click()

    time.sleep(5)

    words = []
    for span in driver.find_elements_by_class_name('word'):
        final_span_el = span.find_elements_by_tag_name('span')[0]

        print(final_span_el)
        words.append(final_span_el.get_attribute('innerHTML'))


    words = " ".join(words)
    print('Question: ' + question)
    print(words)