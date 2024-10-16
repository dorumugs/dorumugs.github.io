# Summary
EBS 오디오 어학당에 들어가면 당연하게도 구독을 해야 강의를 들을 수 있어요.  
영어 공부를 위해 저는 Power English를 선택했어요.  
그런데... 책은 사고 싶지 않았답니다.  

그래서 PDF 있는 것만 해보자 라고 생각했어요.  
PDF있는 강의가 무려 1250개나 있더군요.  
하나씩 다운을 받는 중에 현타가 왔습니다.  

그래서 자동으로 다운 받는 코드를 짰습니다.  
함께 해요!

## 패키지 설치
저는 맥에서 크롤을 진행합니다. 윈도우 코드는 없으니 참고 부탁드려요.  
먼저 패키지를 설치합니다. selenium으로 크롤을 하고 webdriver_manager로 크롬 버전을 자동으로 맞춥니다.  
pyperclip는 네이버 로그인할 때 사용합니다. 다른 SNS도 한번 값을 찾아 구현해보세요.  


```python
!pip install selenium webdriver_manager pyperclip
```

## 코드의 시작
코드는 아래와 같은 순서로 진행되요.  
네이버 로그인 > EBS 어학당 로그인 > EBS 어학당 Power English 이동 > PDF 있는 강의 내려받기  
<br>
참고로!!!! 코드시작 전에 PE라는 폴더를 코드와 같은 경로에 생성해 두셔야 합니다.  
테스타하다가 날아갈까봐 두려워 저도 수동생성했어요.

### 00 라이브러리 선언


```python
from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import time
import pyperclip
import requests
```

### 01 크롬 드라이버 지정 후 네이버 방문


```python
driver = webdriver.Chrome(service= Service(ChromeDriverManager().install()))

url = "https://nid.naver.com/nidlogin.login?mode=form&url=https://www.naver.com/"
driver.maximize_window()
driver.get(url)
```

### 02 네이버 로그인
pyperclip를 사용하여 값을 복사해서 붙여넣는 방식을 사용하면 네이버 로그인시 자동방지를 회피할 수 있어요.  
그래서 코드가 약간 길어졌습니다.  

웹페이지가 하나씩 변경되는걸 기다리면서 진행되야 하므로 time.sleep()을 주기적으로 사용해야 해요.  
또 "기기 등록" 으로 문제가 발생할 수 있으니 기기 등록이 나오면 바로 클릭하게 대비합니다.


```python

naver_id = "네이버 ID"
naver_pw = "네이버 PW"

# 아이디 입력
id_input = driver.find_element(By.CSS_SELECTOR, "#id")
id_input.click()
pyperclip.copy(naver_id)
actions = ActionChains(driver)
actions.key_down(Keys.COMMAND).send_keys('v').key_up(Keys.COMMAND).perform()
time.sleep(1) # 입력 후 잠시 대기

# 패스워드 입력
pw_input = driver.find_element(By.CSS_SELECTOR, "#pw")
pw_input.click()
pyperclip.copy(naver_pw)
actions = ActionChains(driver)
actions.key_down(Keys.COMMAND).send_keys('v').key_up(Keys.COMMAND).perform()
time.sleep(1) # 입력 후 잠시 대기

# 로그인 버튼 클릭
driver.find_element(By.CSS_SELECTOR, "#log\.login").click()

# 로그인 후 '새로운 환경' 알림에서 '등록완료' 버튼 클릭
try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(By.CSS_SELECTOR, "span.btn_cancel")
    )
    element.click()
except:
    print("기기 등록 '등록완료' 버튼을 찾을 수 없습니다.")
```

    기기 등록 '등록완료' 버튼을 찾을 수 없습니다.


### 02 EBS 어학당 로그인
네이버에 로그인인 되었다면, 어학당은 SNS 로그인으로 바로 진입이 가능합니다.  
로그인 버튼으로 바로 로그인하고 EBS Power English 페이지로 이동합니다.  
PDF가 존재하는 강의만 필터가 가능하게 Radio 버튼을 제공하는데 이부분도 처리합니다.


```python
url = "https://5dang.ebs.co.kr/login"
driver.maximize_window()
driver.get(url)

time.sleep(3) # 입력 후 잠시 대기

# 로그인 버튼 클릭
login = driver.find_element(By.CSS_SELECTOR, "#frm > div.left > div.btn_sns > div.lg_sns_list > ul > li.lg_sns01.first_child.first_item > a")
login.click()

time.sleep(3) # 입력 후 잠시 대기

# EBS 어학당 Power English로 이동
url = "https://5dang.ebs.co.kr/auschool/sub/replay?prodId=191&courseId=BK0KAKC0000000005&stepId=01BK0KAKC0000000005&lectId=20269296&situ="
driver.get(url)

time.sleep(3) # 입력 후 잠시 대기

# PDF 있는 강의만 필터
button = driver.find_element(By.ID, 'chk_pdf_only')
driver.execute_script("arguments[0].click();", button)
```

### 03 PDF 강의 개수 확인


```python
table_rows = driver.find_elements(By.CSS_SELECTOR, 'table tr')  # 테이블의 행들 찾기

for index, row in enumerate(table_rows):
    print(f"Row {index + 1}: {row.text}")  # 각 행의 텍스트 출력
    if index + 1 == 4:
        total_cnt = row.text.split(' ')
        total_cnt = total_cnt[0]
        page_cnt = int(total_cnt)//10
    
print('total count :', total_cnt)
print('page count :', page_cnt)
```

    Row 1: 
    Row 2: 1 Walking for Exercise: Why Aren't You Walking Today? 2023.08.31 26316 70
    Row 3: 
    Row 4: 1252 African Safari: I’ll Never Forget This 2024.04.30 4087 29
    Row 5: 1251 Meal Kits: I Finally Placed a Full Order 2024.04.29 2934 30
    Row 6: 1250 It Only Feels like a Dangerous Time to Travel 2024.04.27 2581 30
    Row 7: 1249 Starting a T-Shirt Business: I Owe It All to You! 2024.04.26 2284 23
    Row 8: 1248 Getting a “Touch-Up”: It Harnesses the Power of Collagen 2024.04.25 2406 29
    Row 9: 1247 Commuting by E-Bike: I Always Comply with the Law 2024.04.24 2424 30
    Row 10: 1246 African Safari: My Heart Is in My Throat! 2024.04.23 2417 23
    Row 11: 1245 Meal Kits: It Boosts Your Confidence for Cooking 2024.04.22 2467 25
    Row 12: 1244 Video Games 101 2024.04.20 2434 26
    Row 13: 1243 Starting a T-Shirt Business: A Status Symbol 2024.04.19 2137 24
    total count : 1252
    page count : 125


### 04 Selenium 아닌 BS4
Selenium으로 간단하게 처리하려고 했는데, 사이트 구조가 그렇게는 불가능해 보였어요.  
그래서 BS4를 사용하여 replayAjax에 payload를 넣어 호출하는 방식으로 리스트를 가져왔답니다.  
첫페이지 부터 끝까지 전부 리스트를 가져왔어요. 


```python
# 로그인 후, 세션 쿠키를 가져옴
selenium_cookies = driver.get_cookies()
# 2. requests 세션에 Selenium의 쿠키 적용
session = requests.Session()
# 쿠키를 requests 세션에 설정
for cookie in selenium_cookies:
    session.cookies.set(cookie['name'], cookie['value'])

audio_list = []
pdf_list = []
for i in range(int(page_cnt)): # 요청할 URL
    page_num = i + 1
    print('Page Number :', page_num)
    url = "https://5dang.ebs.co.kr/auschool/replayAjax"

    # POST 요청에 사용할 payload
    payload = {
        'prodId': '',
        'courseId': 'BK0KAKC0000000005',
        'stepId': '01BK0KAKC0000000005',
        'lectId': '20269296',
        'pageNum': {i},
        'orderby': 'NEW',
        'pdfOnly': 'Y',
        'situ': '',
        'startDate': '',
        'endDate': '',
        'date': '',
        'pageSize': 10,
        'subMenuId': '',
        'prodChrgClsNm': '유료'
    }

    # 요청을 보낼 때 추가적으로 필요한 헤더 설정
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # POST 요청 보내기
    response = session.post(url, data=payload, headers=headers)
    time.sleep(2) # 입력 후 잠시 대기

    # '/auschool/download/atchfile?'로 시작하는 모든 <a> 태그의 href 속성만 추출
    soup = BeautifulSoup(response.text, 'html.parser')
    download_links = soup.find_all('a', href=True)


    # 해당 패턴의 링크만 출력
    for link in download_links:
        if link['href'].startswith('/auschool/sub/replay?'):
            audio_link = 'https://5dang.ebs.co.kr' + link['href']
            audio_title = link.text.strip()  # 텍스트 앞뒤 공백 제거

            # URL 파싱을 사용하여 쿼리스트링 분석
            parsed_url = urlparse(audio_link)
            query_params = parse_qs(parsed_url.query)
            lectId = query_params.get('lectId', [''])[0]

            # 하나의 딕셔너리 생성
            audio_dict = {
                'lectId': lectId,
                'audio_title': audio_title,
                'audio_link': audio_link
            }

            # audio_list에 추가
            audio_list.append(audio_dict)
    
        elif link['href'].startswith('/auschool/download/atchfile?'):
            pdf_link = 'https://5dang.ebs.co.kr' + link['href']

            # URL 파싱을 사용하여 쿼리스트링 분석
            parsed_url = urlparse(pdf_link)
            query_params = parse_qs(parsed_url.query)
            lectId = query_params.get('lectId', [''])[0]

            # 하나의 딕셔너리 생성
            pdf_dict = {
                'lectId': lectId,
                'pdf_link': pdf_link
            }

            # pdf_list에 추가
            pdf_list.append(pdf_dict)
    

```

    Page Number : 1
    Page Number : 2
    Page Number : 3
    Page Number : 4
    Page Number : 5
    Page Number : 6
    Page Number : 7
    Page Number : 8
    Page Number : 9
    Page Number : 10
    Page Number : 11
    Page Number : 12
    Page Number : 13
    Page Number : 14
    Page Number : 15
    Page Number : 16
    Page Number : 17
    Page Number : 18
    Page Number : 19
    Page Number : 20
    Page Number : 21
    Page Number : 22
    Page Number : 23
    Page Number : 24
    Page Number : 25
    Page Number : 26
    Page Number : 27
    Page Number : 28
    Page Number : 29
    Page Number : 30
    Page Number : 31
    Page Number : 32
    Page Number : 33
    Page Number : 34
    Page Number : 35
    Page Number : 36
    Page Number : 37
    Page Number : 38
    Page Number : 39
    Page Number : 40
    Page Number : 41
    Page Number : 42
    Page Number : 43
    Page Number : 44
    Page Number : 45
    Page Number : 46
    Page Number : 47
    Page Number : 48
    Page Number : 49
    Page Number : 50
    Page Number : 51
    Page Number : 52
    Page Number : 53
    Page Number : 54
    Page Number : 55
    Page Number : 56
    Page Number : 57
    Page Number : 58
    Page Number : 59
    Page Number : 60
    Page Number : 61
    Page Number : 62
    Page Number : 63
    Page Number : 64
    Page Number : 65
    Page Number : 66
    Page Number : 67
    Page Number : 68
    Page Number : 69
    Page Number : 70
    Page Number : 71
    Page Number : 72
    Page Number : 73
    Page Number : 74
    Page Number : 75
    Page Number : 76
    Page Number : 77
    Page Number : 78
    Page Number : 79
    Page Number : 80
    Page Number : 81
    Page Number : 82
    Page Number : 83
    Page Number : 84
    Page Number : 85
    Page Number : 86
    Page Number : 87
    Page Number : 88
    Page Number : 89
    Page Number : 90
    Page Number : 91
    Page Number : 92
    Page Number : 93
    Page Number : 94
    Page Number : 95
    Page Number : 96
    Page Number : 97
    Page Number : 98
    Page Number : 99
    Page Number : 100
    Page Number : 101
    Page Number : 102
    Page Number : 103
    Page Number : 104
    Page Number : 105
    Page Number : 106
    Page Number : 107
    Page Number : 108
    Page Number : 109
    Page Number : 110
    Page Number : 111
    Page Number : 112
    Page Number : 113
    Page Number : 114
    Page Number : 115
    Page Number : 116
    Page Number : 117
    Page Number : 118
    Page Number : 119
    Page Number : 120
    Page Number : 121
    Page Number : 122
    Page Number : 123
    Page Number : 124
    Page Number : 125


### 05 가져온 리스트 전처리
먼저 타이틀 없는 것들은 지웠습니다.  
그리고 저장 경로에서 문제를 일이키는 "/" 를 담고 있는 타이틀은 " "로 변경했어요.  
audio와 pdf의 개수가 1240으로 동일한 거 보니 잘 가져온 것이 맞아 보이네요.  


```python
audio_list = [item for item in audio_list if item['audio_title'].strip() != '바로듣기']
for item in audio_list:
    item['audio_title'] = item['audio_title'].replace('/', ' ')
print("audio_list :",len(audio_list), "|| pdf_list :", len(pdf_list))
```

    audio_list : 1240 || pdf_list : 1240


audio_list, pdf_list 리스트에 각각 가져온 정보를 담고 있어요.  
이 데이터는 lectId 라는 키로 묶을 수 있더라고요. 그래서 merge를 진행했습니다.


```python
# lectId를 기준으로 병합된 결과를 담을 딕셔너리
merged_dict = {}

# a_list에 있는 데이터를 lectId 기준으로 merged_dict에 추가
for a in audio_list:
    merged_dict[a['lectId']] = a

# b_list에 있는 데이터를 lectId 기준으로 merged_dict에 병합
for b in pdf_list:
    lectId = b['lectId']
    if lectId in merged_dict:
        # 이미 존재하는 lectId에 대해 두 딕셔너리 병합
        merged_dict[lectId].update(b)
    else:
        # b_list에만 있는 lectId의 경우, 새로운 항목으로 추가
        merged_dict[lectId] = b

# 병합된 결과를 리스트로 변환
merged_list = list(merged_dict.values())

# 결과 출력
print(len(merged_list), type(merged_list), merged_list[0]) # 1240
```

정확하게 Merge가 되었는지 확인해 보니, 딱 원하는 모양으로 된것을 확인할 수 있어요.


```python
new_list = merged_list[1101:]
print(new_list[0])
print(new_list[0]['audio_title'])
print(new_list[0]['audio_link'])
print(new_list[0]['pdf_link'])
```

    {'lectId': '20380558', 'audio_title': 'We Need to Get Our Sales Up. Any Ideas?', 'audio_link': 'https://5dang.ebs.co.kr/auschool/sub/replay?prodId=191&lectId=20380558&pageNum=111&orderby=NEW&situ=&startDate=&endDate=&pdfOnly=Y&subMenuId=', 'pdf_link': 'https://5dang.ebs.co.kr/auschool/download/atchfile?filePath=/public/lectures/2024/09/12/13/pdf/7a3d717e-8178-407c-b041-b12c826c9b93.pdf&fileName=Pe202010_23.pdf&courseId=BK0KAKC0000000005&stepId=01BK0KAKC0000000005&lectId=20380558&multiYn=Y'}
    We Need to Get Our Sales Up. Any Ideas?
    https://5dang.ebs.co.kr/auschool/sub/replay?prodId=191&lectId=20380558&pageNum=111&orderby=NEW&situ=&startDate=&endDate=&pdfOnly=Y&subMenuId=
    https://5dang.ebs.co.kr/auschool/download/atchfile?filePath=/public/lectures/2024/09/12/13/pdf/7a3d717e-8178-407c-b041-b12c826c9b93.pdf&fileName=Pe202010_23.pdf&courseId=BK0KAKC0000000005&stepId=01BK0KAKC0000000005&lectId=20380558&multiYn=Y


### 06 이제 다운로드 시작
title, audio 링크, pdf 링크 를 통해서 가져온 정보로 오디오와 PDF를 싹싹 긁어옵니다.  
전부 가져오니 20기가가 조금 넘었어요. 시간도 좀 오래걸립니다. 이걸 손으로 했다면.... 상상하고 싶지 않네요.


```python
for i in range(len(new_list)):
    title = new_list[i]['audio_title']
    audio = new_list[i]['audio_link']
    pdf = new_list[i]['pdf_link']

    driver.get(audio)
    time.sleep(3) # 입력 후 잠시 대기
    video_element = driver.find_element(By.XPATH, '//video[@playerclassname="imgtech.media.VideoPlayer"]')
    video_download_src = video_element.get_attribute('src')  # href 속성 가져오기

    # 파일을 저장할 경로 설정
    file_path = f"./PE/{title}.m4a"

    # 파일 다운로드
    response = requests.get(video_download_src)

    # 파일 저장
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print("파일 다운로드 완료:", file_path)

    # 파일을 저장할 경로 설정
    file_path = f"./PE/{title}.pdf"

    # 파일 다운로드
    response = requests.get(pdf)

    # 파일 저장
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print("파일 다운로드 완료:", file_path)
```

    파일 다운로드 완료: ./PE/We Need to Get Our Sales Up. Any Ideas?.m4a
    파일 다운로드 완료: ./PE/We Need to Get Our Sales Up. Any Ideas?.pdf
    파일 다운로드 완료: ./PE/Going to the Chiropractor.m4a
    파일 다운로드 완료: ./PE/Going to the Chiropractor.pdf
    파일 다운로드 완료: ./PE/Visiting Grandma at the Senior Home.m4a
    파일 다운로드 완료: ./PE/Visiting Grandma at the Senior Home.pdf
    파일 다운로드 완료: ./PE/Travel: Kenya ? Giraffe Manor, Nairobi.m4a
    파일 다운로드 완료: ./PE/Travel: Kenya ? Giraffe Manor, Nairobi.pdf
    파일 다운로드 완료: ./PE/My Chicken Is Undercooked!.m4a
    파일 다운로드 완료: ./PE/My Chicken Is Undercooked!.pdf
    파일 다운로드 완료: ./PE/How to Make a Good First Impression.m4a
    파일 다운로드 완료: ./PE/How to Make a Good First Impression.pdf
    파일 다운로드 완료: ./PE/The Company Website Needs an Overhaul.m4a
    파일 다운로드 완료: ./PE/The Company Website Needs an Overhaul.pdf
    파일 다운로드 완료: ./PE/HIIT: Short Workouts, Big Results.m4a
    파일 다운로드 완료: ./PE/HIIT: Short Workouts, Big Results.pdf
    파일 다운로드 완료: ./PE/Getting an Extension for My Research Paper.m4a
    파일 다운로드 완료: ./PE/Getting an Extension for My Research Paper.pdf
    파일 다운로드 완료: ./PE/Travel: Kenya ? Climbing Mount Kenya.m4a
    파일 다운로드 완료: ./PE/Travel: Kenya ? Climbing Mount Kenya.pdf
    파일 다운로드 완료: ./PE/I think you drink too much coffee!.m4a
    파일 다운로드 완료: ./PE/I think you drink too much coffee!.pdf
    파일 다운로드 완료: ./PE/Salt-Fat-Sugar: the Secret to Fast Food.m4a
    파일 다운로드 완료: ./PE/Salt-Fat-Sugar: the Secret to Fast Food.pdf
    파일 다운로드 완료: ./PE/Congratulations, You’re Employee of the Year!.m4a
    파일 다운로드 완료: ./PE/Congratulations, You’re Employee of the Year!.pdf
    파일 다운로드 완료: ./PE/Is Space Tourism Coming?.m4a
    파일 다운로드 완료: ./PE/Is Space Tourism Coming?.pdf
    파일 다운로드 완료: ./PE/I Got Dumped Via Email!.m4a
    파일 다운로드 완료: ./PE/I Got Dumped Via Email!.pdf
    파일 다운로드 완료: ./PE/Kenya ? the Masai Mara.m4a
    파일 다운로드 완료: ./PE/Kenya ? the Masai Mara.pdf
    파일 다운로드 완료: ./PE/Eating Garlic to Fight Colds.m4a
    파일 다운로드 완료: ./PE/Eating Garlic to Fight Colds.pdf
    파일 다운로드 완료: ./PE/My Dream Journal.m4a
    파일 다운로드 완료: ./PE/My Dream Journal.pdf
    파일 다운로드 완료: ./PE/Rescheduling the Department meeting.m4a
    파일 다운로드 완료: ./PE/Rescheduling the Department meeting.pdf
    파일 다운로드 완료: ./PE/Taking Cold Showers for Health.m4a
    파일 다운로드 완료: ./PE/Taking Cold Showers for Health.pdf
    파일 다운로드 완료: ./PE/My Boyfriend Has the Worst Fashion Sense!.m4a
    파일 다운로드 완료: ./PE/My Boyfriend Has the Worst Fashion Sense!.pdf
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Arriving in Moscow.m4a
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Arriving in Moscow.pdf
    파일 다운로드 완료: ./PE/There’s Nothing Quite Like Late Night Street Food.m4a
    파일 다운로드 완료: ./PE/There’s Nothing Quite Like Late Night Street Food.pdf
    파일 다운로드 완료: ./PE/Why Do Home Remedies Work?.m4a
    파일 다운로드 완료: ./PE/Why Do Home Remedies Work?.pdf
    파일 다운로드 완료: ./PE/The Life of a Personal Shopper Stylist.m4a
    파일 다운로드 완료: ./PE/The Life of a Personal Shopper Stylist.pdf
    파일 다운로드 완료: ./PE/We Aren’t Totally Human?.m4a
    파일 다운로드 완료: ./PE/We Aren’t Totally Human?.pdf
    파일 다운로드 완료: ./PE/City Life Vs. Country Life.m4a
    파일 다운로드 완료: ./PE/City Life Vs. Country Life.pdf
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Papers, Please!.m4a
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Papers, Please!.pdf
    파일 다운로드 완료: ./PE/Cravings While Pregnant.m4a
    파일 다운로드 완료: ./PE/Cravings While Pregnant.pdf
    파일 다운로드 완료: ./PE/What Will the Internet Look Like in 10 Years?.m4a
    파일 다운로드 완료: ./PE/What Will the Internet Look Like in 10 Years?.pdf
    파일 다운로드 완료: ./PE/Working from Home.m4a
    파일 다운로드 완료: ./PE/Working from Home.pdf
    파일 다운로드 완료: ./PE/You’re Never Too Old to Skateboard!.m4a
    파일 다운로드 완료: ./PE/You’re Never Too Old to Skateboard!.pdf
    파일 다운로드 완료: ./PE/I Regret Sending That Email! Help!.m4a
    파일 다운로드 완료: ./PE/I Regret Sending That Email! Help!.pdf
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Getting from Irkutsk to Kultuk.m4a
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Getting from Irkutsk to Kultuk.pdf
    파일 다운로드 완료: ./PE/You’ve Never Had a Fresh Bagel? No Way!.m4a
    파일 다운로드 완료: ./PE/You’ve Never Had a Fresh Bagel? No Way!.pdf
    파일 다운로드 완료: ./PE/Are We Losing Online Privacy?.m4a
    파일 다운로드 완료: ./PE/Are We Losing Online Privacy?.pdf
    파일 다운로드 완료: ./PE/Getting My Luxury Car Detailed.m4a
    파일 다운로드 완료: ./PE/Getting My Luxury Car Detailed.pdf
    파일 다운로드 완료: ./PE/Should You Use a Tablet Computer to Keep Your Kids Busy?.m4a
    파일 다운로드 완료: ./PE/Should You Use a Tablet Computer to Keep Your Kids Busy?.pdf
    파일 다운로드 완료: ./PE/Breaking Bad Habits.m4a
    파일 다운로드 완료: ./PE/Breaking Bad Habits.pdf
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? This Train Ride Is Never Ending!.m4a
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? This Train Ride Is Never Ending!.pdf
    파일 다운로드 완료: ./PE/Food Bloggers.m4a
    파일 다운로드 완료: ./PE/Food Bloggers.pdf
    파일 다운로드 완료: ./PE/Homeschooling vs. Public School.m4a
    파일 다운로드 완료: ./PE/Homeschooling vs. Public School.pdf
    파일 다운로드 완료: ./PE/Learning to Play Golf to Help Make Sales.m4a
    파일 다운로드 완료: ./PE/Learning to Play Golf to Help Make Sales.pdf
    파일 다운로드 완료: ./PE/I Don’t Take Medicine If I Can Avoid It.m4a
    파일 다운로드 완료: ./PE/I Don’t Take Medicine If I Can Avoid It.pdf
    파일 다운로드 완료: ./PE/My Fiancee Wants 5 Kids!.m4a
    파일 다운로드 완료: ./PE/My Fiancee Wants 5 Kids!.pdf
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Vladivostok.m4a
    파일 다운로드 완료: ./PE/The Trans-Siberian Railway ? Vladivostok.pdf
    파일 다운로드 완료: ./PE/How Do You Choose a Restaurant?.m4a
    파일 다운로드 완료: ./PE/How Do You Choose a Restaurant?.pdf
    파일 다운로드 완료: ./PE/Having an Online Fundraiser for Your Birthday.m4a
    파일 다운로드 완료: ./PE/Having an Online Fundraiser for Your Birthday.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Part-time Wedding Planner.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Part-time Wedding Planner.pdf
    파일 다운로드 완료: ./PE/Preserving Dying Languages.m4a
    파일 다운로드 완료: ./PE/Preserving Dying Languages.pdf
    파일 다운로드 완료: ./PE/Worst Blind Date of My Life!.m4a
    파일 다운로드 완료: ./PE/Worst Blind Date of My Life!.pdf
    파일 다운로드 완료: ./PE/The View of Montreal from Mount Royal Park.m4a
    파일 다운로드 완료: ./PE/The View of Montreal from Mount Royal Park.pdf
    파일 다운로드 완료: ./PE/Poutine in Montreal.m4a
    파일 다운로드 완료: ./PE/Poutine in Montreal.pdf
    파일 다운로드 완료: ./PE/Using Boredom as a Tool.m4a
    파일 다운로드 완료: ./PE/Using Boredom as a Tool.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Personal Chef.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Personal Chef.pdf
    파일 다운로드 완료: ./PE/My Electric Car Is Out of Juice!.m4a
    파일 다운로드 완료: ./PE/My Electric Car Is Out of Juice!.pdf
    파일 다운로드 완료: ./PE/Childhood Now Versus the “Old Days”.m4a
    파일 다운로드 완료: ./PE/Childhood Now Versus the “Old Days”.pdf
    파일 다운로드 완료: ./PE/Montreal: Habitat 67 ? Futuristic Housing.m4a
    파일 다운로드 완료: ./PE/Montreal: Habitat 67 ? Futuristic Housing.pdf
    파일 다운로드 완료: ./PE/Who Is a Better Cook, Your Mother or Your Father?.m4a
    파일 다운로드 완료: ./PE/Who Is a Better Cook, Your Mother or Your Father?.pdf
    파일 다운로드 완료: ./PE/The Svalbard Global Seed Vault.m4a
    파일 다운로드 완료: ./PE/The Svalbard Global Seed Vault.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Hot Air Balloon Chase Crew.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Hot Air Balloon Chase Crew.pdf
    파일 다운로드 완료: ./PE/Are Standard IQ Tests Accurate or Culturally Biased?.m4a
    파일 다운로드 완료: ./PE/Are Standard IQ Tests Accurate or Culturally Biased?.pdf
    파일 다운로드 완료: ./PE/Would You Rather Be Rich or Famous?.m4a
    파일 다운로드 완료: ./PE/Would You Rather Be Rich or Famous?.pdf
    파일 다운로드 완료: ./PE/Walking through Old Montreal.m4a
    파일 다운로드 완료: ./PE/Walking through Old Montreal.pdf
    파일 다운로드 완료: ./PE/Why Do the Smells of Food Bring Back Certain Memories?.m4a
    파일 다운로드 완료: ./PE/Why Do the Smells of Food Bring Back Certain Memories?.pdf
    파일 다운로드 완료: ./PE/Why We LOVE Baby Animals.m4a
    파일 다운로드 완료: ./PE/Why We LOVE Baby Animals.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Do I Have What It Takes?.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Do I Have What It Takes?.pdf
    파일 다운로드 완료: ./PE/Should Students Be Allowed to Use Calculators on Tests?.m4a
    파일 다운로드 완료: ./PE/Should Students Be Allowed to Use Calculators on Tests?.pdf
    파일 다운로드 완료: ./PE/Moving Your Elderly Parents to Live With You.m4a
    파일 다운로드 완료: ./PE/Moving Your Elderly Parents to Live With You.pdf
    파일 다운로드 완료: ./PE/Why Don’t Many People Speak English Here?.m4a
    파일 다운로드 완료: ./PE/Why Don’t Many People Speak English Here?.pdf
    파일 다운로드 완료: ./PE/The Dirty Dozen.m4a
    파일 다운로드 완료: ./PE/The Dirty Dozen.pdf
    파일 다운로드 완료: ./PE/Is It Too Late to Stop Climate Change?.m4a
    파일 다운로드 완료: ./PE/Is It Too Late to Stop Climate Change?.pdf
    파일 다운로드 완료: ./PE/How About a Food Tour of My City?.m4a
    파일 다운로드 완료: ./PE/How About a Food Tour of My City?.pdf
    파일 다운로드 완료: ./PE/Is Genetic Editing Ethical?.m4a
    파일 다운로드 완료: ./PE/Is Genetic Editing Ethical?.pdf
    파일 다운로드 완료: ./PE/“Modern Art Is Just…not Art.”.m4a
    파일 다운로드 완료: ./PE/“Modern Art Is Just…not Art.”.pdf
    파일 다운로드 완료: ./PE/The Great Geyser.m4a
    파일 다운로드 완료: ./PE/The Great Geyser.pdf
    파일 다운로드 완료: ./PE/Fermented Shark or Sheep’s Head? Tough Choice..m4a
    파일 다운로드 완료: ./PE/Fermented Shark or Sheep’s Head? Tough Choice..pdf
    파일 다운로드 완료: ./PE/How Long do You Want to Live?.m4a
    파일 다운로드 완료: ./PE/How Long do You Want to Live?.pdf
    파일 다운로드 완료: ./PE/I Handwrite Letters for People.m4a
    파일 다운로드 완료: ./PE/I Handwrite Letters for People.pdf
    파일 다운로드 완료: ./PE/Do We Rely on Computers Too Much?.m4a
    파일 다운로드 완료: ./PE/Do We Rely on Computers Too Much?.pdf
    파일 다운로드 완료: ./PE/I Think I Saw a UFO!.m4a
    파일 다운로드 완료: ./PE/I Think I Saw a UFO!.pdf
    파일 다운로드 완료: ./PE/Iceland: I’m Not Going to “Fly Lake!” (Lake Myvatn).m4a
    파일 다운로드 완료: ./PE/Iceland: I’m Not Going to “Fly Lake!” (Lake Myvatn).pdf
    파일 다운로드 완료: ./PE/The Problem of Food Deserts.m4a
    파일 다운로드 완료: ./PE/The Problem of Food Deserts.pdf
    파일 다운로드 완료: ./PE/A Photographic Memory.m4a
    파일 다운로드 완료: ./PE/A Photographic Memory.pdf
    파일 다운로드 완료: ./PE/Opening a Workspace for Virtual Workers.m4a
    파일 다운로드 완료: ./PE/Opening a Workspace for Virtual Workers.pdf
    파일 다운로드 완료: ./PE/Are You Ever Too Old to Learn a Language?.m4a
    파일 다운로드 완료: ./PE/Are You Ever Too Old to Learn a Language?.pdf
    파일 다운로드 완료: ./PE/What Advice Would You Give Your Younger Self?.m4a
    파일 다운로드 완료: ./PE/What Advice Would You Give Your Younger Self?.pdf
    파일 다운로드 완료: ./PE/Iceland: Whale Watching in Olafsik.m4a
    파일 다운로드 완료: ./PE/Iceland: Whale Watching in Olafsik.pdf
    파일 다운로드 완료: ./PE/I’m Learning to Cook Online!.m4a
    파일 다운로드 완료: ./PE/I’m Learning to Cook Online!.pdf
    파일 다운로드 완료: ./PE/Is Love Real or Just Chemical Reactions?.m4a
    파일 다운로드 완료: ./PE/Is Love Real or Just Chemical Reactions?.pdf
    파일 다운로드 완료: ./PE/A Professional Matchmaker.m4a
    파일 다운로드 완료: ./PE/A Professional Matchmaker.pdf
    파일 다운로드 완료: ./PE/How Color Affects One’s Mood.m4a
    파일 다운로드 완료: ./PE/How Color Affects One’s Mood.pdf
    파일 다운로드 완료: ./PE/Your Home Is So Cozy!.m4a
    파일 다운로드 완료: ./PE/Your Home Is So Cozy!.pdf
    파일 다운로드 완료: ./PE/Iceland: Nightlife in Reykjavik.m4a
    파일 다운로드 완료: ./PE/Iceland: Nightlife in Reykjavik.pdf
    파일 다운로드 완료: ./PE/I Have Food in My Teeth, and Nobody Said Anything!.m4a
    파일 다운로드 완료: ./PE/I Have Food in My Teeth, and Nobody Said Anything!.pdf
    파일 다운로드 완료: ./PE/Is the “5-second Rule” Based on Science?.m4a
    파일 다운로드 완료: ./PE/Is the “5-second Rule” Based on Science?.pdf
    파일 다운로드 완료: ./PE/I’m Going to Self-Publish My Book.m4a
    파일 다운로드 완료: ./PE/I’m Going to Self-Publish My Book.pdf
    파일 다운로드 완료: ./PE/Can You Be Too Clean?.m4a
    파일 다운로드 완료: ./PE/Can You Be Too Clean?.pdf
    파일 다운로드 완료: ./PE/I’m the Oldest Person in the Office!.m4a
    파일 다운로드 완료: ./PE/I’m the Oldest Person in the Office!.pdf
    파일 다운로드 완료: ./PE/Miami: A Baseball Game at Marlins Park.m4a
    파일 다운로드 완료: ./PE/Miami: A Baseball Game at Marlins Park.pdf
    파일 다운로드 완료: ./PE/Bugs will be the new source of protein.m4a
    파일 다운로드 완료: ./PE/Bugs will be the new source of protein.pdf
    파일 다운로드 완료: ./PE/“Global Weirding” Is Here.m4a
    파일 다운로드 완료: ./PE/“Global Weirding” Is Here.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Piano Teacher.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Piano Teacher.pdf
    파일 다운로드 완료: ./PE/Planting Trees to Help the Environment.m4a
    파일 다운로드 완료: ./PE/Planting Trees to Help the Environment.pdf
    파일 다운로드 완료: ./PE/Daily Journaling.m4a
    파일 다운로드 완료: ./PE/Daily Journaling.pdf
    파일 다운로드 완료: ./PE/Miami: Ocean Drive Art Deco Buildings.m4a
    파일 다운로드 완료: ./PE/Miami: Ocean Drive Art Deco Buildings.pdf
    파일 다운로드 완료: ./PE/Do Food Expiration Dates Matter?.m4a
    파일 다운로드 완료: ./PE/Do Food Expiration Dates Matter?.pdf
    파일 다운로드 완료: ./PE/Panning for Gold.m4a
    파일 다운로드 완료: ./PE/Panning for Gold.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Hosting “Watercolor and Wine” parties.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Hosting “Watercolor and Wine” parties.pdf
    파일 다운로드 완료: ./PE/Make Your Own Vinyl Records.m4a
    파일 다운로드 완료: ./PE/Make Your Own Vinyl Records.pdf
    파일 다운로드 완료: ./PE/Stress Baking.m4a
    파일 다운로드 완료: ./PE/Stress Baking.pdf
    파일 다운로드 완료: ./PE/Miami: Day Trip to Key West.m4a
    파일 다운로드 완료: ./PE/Miami: Day Trip to Key West.pdf
    파일 다운로드 완료: ./PE/Stress Baking.m4a
    파일 다운로드 완료: ./PE/Stress Baking.pdf
    파일 다운로드 완료: ./PE/Want to Get Healthy? Dance!.m4a
    파일 다운로드 완료: ./PE/Want to Get Healthy? Dance!.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Professional Audience Member.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Professional Audience Member.pdf
    파일 다운로드 완료: ./PE/Drinking Water to Avoid Headaches.m4a
    파일 다운로드 완료: ./PE/Drinking Water to Avoid Headaches.pdf
    파일 다운로드 완료: ./PE/Getting a Cast Off at the Doctor’s Office.m4a
    파일 다운로드 완료: ./PE/Getting a Cast Off at the Doctor’s Office.pdf
    파일 다운로드 완료: ./PE/Miami: Everglades Park.m4a
    파일 다운로드 완료: ./PE/Miami: Everglades Park.pdf
    파일 다운로드 완료: ./PE/Alligator Steak? No Way!.m4a
    파일 다운로드 완료: ./PE/Alligator Steak? No Way!.pdf
    파일 다운로드 완료: ./PE/The Creation of National Parks.m4a
    파일 다운로드 완료: ./PE/The Creation of National Parks.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Online Language Teacher.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Online Language Teacher.pdf
    파일 다운로드 완료: ./PE/The Dangers of Commercial Sunscreens.m4a
    파일 다운로드 완료: ./PE/The Dangers of Commercial Sunscreens.pdf
    파일 다운로드 완료: ./PE/Camping with Bears.m4a
    파일 다운로드 완료: ./PE/Camping with Bears.pdf
    파일 다운로드 완료: ./PE/Miami: Biscayne Bay Dinner Cruise.m4a
    파일 다운로드 완료: ./PE/Miami: Biscayne Bay Dinner Cruise.pdf
    파일 다운로드 완료: ./PE/Can I get something instead of carrots?.m4a
    파일 다운로드 완료: ./PE/Can I get something instead of carrots?.pdf
    파일 다운로드 완료: ./PE/The Dangers of Blue Light.m4a
    파일 다운로드 완료: ./PE/The Dangers of Blue Light.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: YouTube Product Reviewer.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: YouTube Product Reviewer.pdf
    파일 다운로드 완료: ./PE/There’s an app for that!.m4a
    파일 다운로드 완료: ./PE/There’s an app for that!.pdf
    파일 다운로드 완료: ./PE/Dating a Co-Worker.m4a
    파일 다운로드 완료: ./PE/Dating a Co-Worker.pdf
    파일 다운로드 완료: ./PE/Los Angeles: Movie Studio Tour.m4a
    파일 다운로드 완료: ./PE/Los Angeles: Movie Studio Tour.pdf
    파일 다운로드 완료: ./PE/What’s So Great About Bagels?.m4a
    파일 다운로드 완료: ./PE/What’s So Great About Bagels?.pdf
    파일 다운로드 완료: ./PE/The Power of Random Acts of Kindness.m4a
    파일 다운로드 완료: ./PE/The Power of Random Acts of Kindness.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Scalping Tickets.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: Scalping Tickets.pdf
    파일 다운로드 완료: ./PE/The Jellyfish That Lives Forever.m4a
    파일 다운로드 완료: ./PE/The Jellyfish That Lives Forever.pdf
    파일 다운로드 완료: ./PE/A Tough Job Market for Graduates.m4a
    파일 다운로드 완료: ./PE/A Tough Job Market for Graduates.pdf
    파일 다운로드 완료: ./PE/Los Angeles: Korea Town.m4a
    파일 다운로드 완료: ./PE/Los Angeles: Korea Town.pdf
    파일 다운로드 완료: ./PE/I Hate cooking, But l love Cooking Shows!.m4a
    파일 다운로드 완료: ./PE/I Hate cooking, But l love Cooking Shows!.pdf
    파일 다운로드 완료: ./PE/The Perfect air Purifier.m4a
    파일 다운로드 완료: ./PE/The Perfect air Purifier.pdf
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: The Life of a Virtual Assistant.m4a
    파일 다운로드 완료: ./PE/Entrepreneur’s Life: The Life of a Virtual Assistant.pdf

