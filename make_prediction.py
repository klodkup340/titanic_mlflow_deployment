import requests

payload = '[{"Pclass_1":1,"Pclass_2":0,"Pclass_3":0,"Sex_male":1,"Sex_female":0,"Age_category_Missing":0,"Age_category_Infant":0,"Age_category_Child":0,"Age_category_Teenager":0,"Age_category_Young Adult":0,"Age_category_Adult":1,"Age_category_Senior":0}, ' \
          ' {"Pclass_1":1,"Pclass_2":0,"Pclass_3":0,"Sex_male":0,"Sex_female":1,"Age_category_Missing":0,"Age_category_Infant":0,"Age_category_Child":0,"Age_category_Teenager":0,"Age_category_Young Adult":0,"Age_category_Adult":1,"Age_category_Senior":0}, ' \
          ' {"Pclass_1":0,"Pclass_2":0,"Pclass_3":1,"Sex_male":1,"Sex_female":0,"Age_category_Missing":0,"Age_category_Infant":1,"Age_category_Child":0,"Age_category_Teenager":0,"Age_category_Young Adult":0,"Age_category_Adult":0,"Age_category_Senior":0}, ' \
          ' {"Pclass_1":1,"Pclass_2":0,"Pclass_3":0,"Sex_male":1,"Sex_female":0,"Age_category_Missing":0,"Age_category_Infant":0,"Age_category_Child":0,"Age_category_Teenager":0,"Age_category_Young Adult":0,"Age_category_Adult":1,"Age_category_Senior":0}, ' \
          ' {"Pclass_1":0,"Pclass_2":0,"Pclass_3":1,"Sex_male":0,"Sex_female":1,"Age_category_Missing":1,"Age_category_Infant":0,"Age_category_Child":0,"Age_category_Teenager":0,"Age_category_Young Adult":0,"Age_category_Adult":0,"Age_category_Senior":0}]'
headers = {'Content-Type': 'application/json; format=pandas-records'}
requests_uri = 'http://127.0.0.1:5000/invocations'

if __name__ == '__main__':
   try:
      response = requests.post(requests_uri, data=payload, headers=headers)
      print(response.content)
   except Exception as ex:
      raise(ex)