import requests

HOST = '127.0.0.1:9092'

def test_ui_infer_api(img_url):
    ui_infer_url = f'http://{HOST}/vision/ui-infer'
    params = {'url': img_url}
    response = requests.post(url=ui_infer_url, json=params)
    result = response.json()
    return result


if __name__ == '__main__':
    
    img_url = 'https://github.com/Meituan-Dianping/vision-ui/blob/master/image/1_0.png?raw=true'
    result = test_ui_infer_api(img_url)
    assert(result['code'] == 0)

    # image url not found
    img_url = 'https://github.com/Meituan-Dianping/vision-ui/blob/master/image/er.png?raw=true'
    result = test_ui_infer_api(img_url)
    assert(result['code'] == 4)

