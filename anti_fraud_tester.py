import requests
import threading
import time

url = "https://ttqsr.top/c98aaWwP"  # 你的测试链接
num_requests = 50   # 总请求数
threads = 10        # 并发线程数

def send_request():
    try:
        resp = requests.get(url, timeout=5)
        print(f"Status: {resp.status_code}, IP: {resp.raw._connection.sock.getpeername()[0] if resp.raw._connection else 'unknown'}")
    except Exception as e:
        print(f"Error: {e}")

def worker():
    while True:
        with lock:
            if counter[0] >= num_requests:
                break
            counter[0] += 1
        send_request()
        time.sleep(0.1)  # 避免太猛

lock = threading.Lock()
counter = [0]
thread_pool = []
for _ in range(threads):
    t = threading.Thread(target=worker)
    t.start()
    thread_pool.append(t)
for t in thread_pool:
    t.join()
print("Done")