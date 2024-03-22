def find_dog(sound):
    if sound == "멍멍":
        return("개가 짖네")

    else:
        return("다른 동물이구나")

sound = "야옹"
find_result = find_dog(sound)

print(find_result)

import torch

print(torch.cuda.is_available())
"""
- Variables: 변수
- WATCH: 조사식
- CALL STACK: 호출 스택

1. break point 지정: 빨간 점으로 표시한 라인 전까지 코드 실행
2. run
"""