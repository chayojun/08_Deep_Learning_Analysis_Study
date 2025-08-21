import streamlit as st

# st.title("Hello Streamlit")
# st.write("스트림릿 사이트")

# name = st.text_input("이름을 입력하세요")

# if st.button("실행"):
#     st.success(f"안녕하세요 : {name}님 반갑습니다.")

# st.title("스트림릿 제목 ----- 1")
# st.header("헤더")
# st. subheader("서브헤더")
# st.text("일반텍스트")
# st.markdown(" ** 마크다운 지원 ** 정말 좋아요")
# st.markdown("*마크다운 지원* 정말 좋아요")
# #st.markdown ("! [O|0||](https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQVgr
# st.code("print('Hello Python!')", language="python")

st.set_page_config(page_title="레이아웃",layout="wide")

if "counter" not in st.session_state:
    st.session_state.counter = 0

with st.sidebar:
    userid = st.text_input("ID를 입력하세요", key="id")
    
    st.write("sesstion_state :", st.session_state.id)

    st.header("옵션")
    data = st.date_input("날짜")
    cls = st.selectbox("클래스", ["A","B","C"])

st.title("대시보드")
st.write("----------------------------------------------------")


tab1,tab2,tab3 = st.tabs(["개요","지표","결과"])

with tab1:
    # st.header("원티드 매출분석")
    # container = st.container()
    # container.write("컨테이너 안에 들어가는 글")
    # st.write("컨테이너 밖")
    if st.button("증가"):
        st.session_state.counter += 1

    if st.button("감소"):
        st.session_state.counter -= 1

    st.write("현재값:",st.session_state.counter)


    with st.expander("더보기"):
        st.write("이안에 숨겨진 기능이 있습니다.")



with tab2:
    col1,col2,col3 = st.columns([2,5,3])
    with col1:
        st.subheader("요약지표")
        st.metric("Auccuracy", "94.3%", "+0.7%")
    with col2:
        st.subheader("라인차트")
        st.line_chart({"acc":[0.8,1.2, 0.4,0.9]})
    with col3:
        st.subheader("세부옵션")
        st.checkbox("원티드")
with tab3:
    st.metric("최종 매출", "999,999,999,999", "+999.7%")