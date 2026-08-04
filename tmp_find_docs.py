import re, requests
urls = ['https://www.rmit.edu.vn/study-at-rmit/tuition-fees','https://www.rmit.edu.vn/study-at-rmit/scholarships/','https://www.rmit.edu.vn/students/my-studies/fees-and-payments']
for u in urls:
    print('URL:', u)
    try:
        r = requests.get(u, timeout=30, headers={'User-Agent':'Mozilla/5.0'})
        print('status', r.status_code)
        text = r.text
        links = sorted(set(re.findall(r'https?://[^\s"\']+\.(?:pdf|doc|docx)', text, flags=re.I)))
        for link in links:
            print(link)
        print('---')
    except Exception as e:
        print('ERR', e)
        print('---')
