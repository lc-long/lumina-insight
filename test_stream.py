import requests
import json

url = "http://127.0.0.1:8000/api/chat/stream"
payload = {
    "question": "解释一下hpakt论文中的热力图，分析这个热力图是什么意思，表现了模型的什么?",
    "user_id": "User_A"
}

print("====== 🚀 发起流式请求 ======\n")

# 发起 POST 请求，开启 stream 模式
with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            # 剔除 SSE 协议前面的 "data: " 前缀
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                try:
                    # 解析后端的结构化 JSON
                    event = json.loads(data_str)
                    
                    if event["type"] == "sources":
                        print("📚 [第一时间获取到引用来源]:")
                        for idx, source in enumerate(event["data"]):
                            print(f"   {idx+1}. 文件: {source['source']}, 页码: {source['page']}, 相关度: {source['score']}")
                        print("\n🤖 [AI 开始作答]:\n", end="")
                        
                    elif event["type"] == "content":
                        # 像打字机一样打印字符
                        print(event["data"], end="", flush=True)
                        
                    elif event["type"] == "done":
                        print("\n\n✅ [输出完毕]")
                        
                    elif event["type"] == "error":
                        print(f"\n❌ [发生错误]: {event['data']}")
                        
                except json.JSONDecodeError:
                    print(f"\n解析失败的原始数据: {data_str}")