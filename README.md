# 환승연애4 유튜브 숏폼 vs 롱폼 참여 분석 대시보드

본 프로젝트는 TVING 오리지널 시리즈 **[환승연애4]**의 유튜브 콘텐츠를 대상으로,  
동일한 방송 클립이 숏폼과 롱폼 포맷으로 업로드되었을 때  
사용자 참여(댓글, 좋아요) 양상이 어떻게 달라지는지를 분석하고 시각화한 Streamlit 기반 대시보드이다.

---

## 📌 프로젝트 배경
환승연애4는 이례적으로 유튜브 댓글창을 개방하고,  
동일한 회차·동일한 장면을 기반으로 한 숏폼과 롱폼 콘텐츠를 병행 업로드하고 있다.  
숏폼은 알고리즘 기반 노출이 강한 반면, 롱폼은 사용자의 ‘선택’에 의해 시청된다는 점에서  
두 포맷 간 참여 양상에 차이가 존재할 가능성이 있다.

본 프로젝트는 이러한 차이가 실제 데이터 상에서 어떻게 나타나는지를 탐색하는 것을 목표로 한다.

---

## 📊 사용 데이터
- 유튜브에 업로드된 환승연애4 숏폼 및 롱폼 콘텐츠 데이터
- 주요 컬럼
  - `video_type` : shorts / long
  - `duration` : 영상 길이 (초)
  - `comments_engagement` : 댓글 참여율
  - `likes_engagement` : 좋아요 참여율

---

## 🛠 사용 기술
- Python
- Streamlit
- pandas
- matplotlib
- seaborn

---

## 🚀 실행 방법 (로컬)
```bash
pip install streamlit pandas matplotlib seaborn openpyxl
streamlit run app.py
